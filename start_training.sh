#!/bin/bash
# SO101 训练启动器（在训练服务器上跑，不接机械臂和相机）
#
# 用 POLICY 选策略：
#   pi05     （默认）从 models/pi05_base 微调：VLM 冻结，只训动作专家，bf16，开梯度检查点，batch 32 单卡峰值约 19GB。
#            全量微调要 70GB 以上，大显存机器上可以加 --policy.train_expert_only=false。
#   smolvla  从 models/smolvla_base 微调：视觉编码器冻结，只训动作专家。另需本地缓存里有 SmolVLM2（见下）。
# 默认用上所有可见显卡（CUDA_VISIBLE_DEVICES 可限定），多卡时用 accelerate 做数据并行。
# 数据集名、任务文本读 station.env；数据集默认取 data/ 下唯一的那个。
#
# 用法（先开 tmux，SSH 断了训练不会停）：
#   SMOKE=1 bash start_training.sh                                   # 冒烟测试：跑 20 步并存一次，确认数据、显存、保存都正常
#   PRECHECK_ONLY=1 bash start_training.sh                           # 只做前置检查，打印将要执行的命令
#   bash start_training.sh                                           # 正式训练 π0.5，用全部可见显卡
#   POLICY=smolvla bash start_training.sh                            # 训练 SmolVLA
#   CUDA_VISIBLE_DEVICES=0,1 POLICY=smolvla bash start_training.sh   # 只用 0、1 号卡
#   RESUME=outputs/train/<名字> bash start_training.sh               # 中断后从最近一次 checkpoint 接着训
#
# 准备 SmolVLA 底座（在项目目录执行，走 hf-mirror）：
#   hf download lerobot/smolvla_base --revision 5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6 --local-dir models/smolvla_base
#   hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct
#
# 环境变量：
#   POLICY       pi05 / smolvla，默认 pi05；续训时以 checkpoint 里记录的为准
#   DATA_ROOT    数据集目录，留空=自动找 data/ 下唯一的数据集
#   BASE_MODEL   底座模型目录，默认 models/pi05_base 或 models/smolvla_base
#   NUM_GPUS     用几张卡，默认全部可见的卡
#   BATCH_SIZE   每张卡的 batch，默认 π0.5 32、SmolVLA 64；实际 batch = BATCH_SIZE × 卡数
#   STEPS        训练步数。默认按「和单卡训练一样多的样本量」折算：π0.5 单卡 10000 步 × 32、
#                SmolVLA 单卡 20000 步 × 64（50 集约 2.3 万帧，分别约 14 轮、56 轮），再除以实际 batch
#   SAVE_FREQ    每多少步存一次 checkpoint，默认整个训练存两次；启动前按数据盘剩余空间检查
#   NUM_WORKERS  每张卡的数据加载进程数，默认 min(8, CPU 核数 / 卡数)
#   GRAD_CKPT    π0.5 是否开梯度检查点，默认 1；设 0 更快，但显存要 30GB 以上，只适合 48GB 及以上的卡
#   SMOLVLA_RENAME_MAP  SmolVLA 的相机改名，默认 observation.images.top=observation.images.camera1
#                       （station.env 里的 RENAME_MAP 是给 π0.5 的）
#   MASTER_PORT  多卡通信端口，默认自动找空闲端口，同时跑两个训练也不冲突
#   JOB_NAME     输出目录名（outputs/train/ 下），默认 <POLICY>_<时间>
# 其余参数原样传给 lerobot-train，例如：bash start_training.sh --policy.optimizer_lr=5e-5
source "$(dirname "${BASH_SOURCE[0]}")/common.sh" || exit 1
set -uo pipefail
so101_activate_env || exit 1
cd "$ROOT" || exit 1

STAMP=$(date +%Y%m%d_%H%M%S)
SMOKE=${SMOKE:-0}
PRECHECK_ONLY=${PRECHECK_ONLY:-0}
RESUME=${RESUME:-}
DATA_ROOT=${DATA_ROOT:-}
GRAD_CKPT=${GRAD_CKPT:-1}

# 续训时策略类型以 checkpoint 为准，不看 POLICY
if [ -n "$RESUME" ]; then
  RESUME_CFG=$RESUME/checkpoints/last/pretrained_model/train_config.json
  [ -f "$RESUME_CFG" ] || { echo "!! 找不到 $RESUME_CFG，RESUME 要指向 outputs/train/<名字>"; exit 1; }
  POLICY=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["policy"]["type"])' "$RESUME_CFG") || exit 1
  POLICY_CFG=$RESUME/checkpoints/last/pretrained_model/config.json
fi
POLICY=${POLICY:-pi05}
# 每种策略：默认底座、单卡时的步数和 batch（步数折算的基准），以及一个 checkpoint 的大致体积
case "$POLICY" in
  pi05)    BASE_DEFAULT=$ROOT/models/pi05_base;    BASE_STEPS=10000; BASE_BATCH=32; CKPT_GB=10 ;;  # bf16 模型约 9.4GB + 优化器状态
  smolvla) BASE_DEFAULT=$ROOT/models/smolvla_base; BASE_STEPS=20000; BASE_BATCH=64; CKPT_GB=3 ;;   # 模型约 0.9GB + 优化器状态
  *) echo "!! POLICY 只能是 pi05 或 smolvla，现在是 $POLICY"; exit 1 ;;
esac
BASE_MODEL=${BASE_MODEL:-$BASE_DEFAULT}
[ -z "$RESUME" ] && POLICY_CFG=$BASE_MODEL/config.json

VISIBLE_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null) || VISIBLE_GPUS=0
NUM_GPUS=${NUM_GPUS:-$VISIBLE_GPUS}
case "$NUM_GPUS" in ''|*[!0-9]*) echo "!! NUM_GPUS 要是正整数，现在是 $NUM_GPUS"; exit 1 ;; esac
if [ "$NUM_GPUS" -lt 1 ]; then
  echo "!! torch 看不到 CUDA 显卡（nvidia-smi 能看到的话，检查 CUDA_VISIBLE_DEVICES 和 torch 的 CUDA 版本）"; exit 1
fi
if [ "$NUM_GPUS" -gt "$VISIBLE_GPUS" ]; then
  echo "!! NUM_GPUS=$NUM_GPUS 超过了可见的 $VISIBLE_GPUS 张卡"; exit 1
fi

BATCH_SIZE=${BATCH_SIZE:-$BASE_BATCH}
TOTAL_BATCH=$(( BATCH_SIZE * NUM_GPUS ))
STEPS=${STEPS:-$(( (BASE_STEPS * BASE_BATCH + TOTAL_BATCH - 1) / TOTAL_BATCH ))}
SAVE_FREQ=${SAVE_FREQ:-$(( (STEPS + 1) / 2 ))}
LOG_FREQ=${LOG_FREQ:-$(( STEPS / 100 > 10 ? STEPS / 100 : 10 ))}
WORKERS_DEFAULT=$(( $(nproc) / NUM_GPUS )); [ "$WORKERS_DEFAULT" -gt 8 ] && WORKERS_DEFAULT=8; [ "$WORKERS_DEFAULT" -lt 1 ] && WORKERS_DEFAULT=1
NUM_WORKERS=${NUM_WORKERS:-$WORKERS_DEFAULT}
JOB_NAME=${JOB_NAME:-${POLICY}_$STAMP}
if [ "$SMOKE" = "1" ]; then
  STEPS=20; SAVE_FREQ=20; LOG_FREQ=5; JOB_NAME=smoke_${POLICY}_$STAMP
fi

# 底座模型、分词器、数据集都在本地，禁止联网：缺文件时立刻报错，而不是卡在下载超时上
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
if [ "$NUM_GPUS" -gt 1 ]; then
  # 消费级显卡（GeForce）不支持卡间 P2P，NCCL 默认去试会卡住；A800 这类数据中心卡不受影响
  if [ -z "${NCCL_P2P_DISABLE:-}" ] && nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q GeForce; then
    export NCCL_P2P_DISABLE=1
  fi
  MASTER_PORT=${MASTER_PORT:-$(python -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')}
fi

mkdir -p "$ROOT/logs" "$ROOT/outputs/train"
LOG=$ROOT/logs/train_$STAMP.log

# ---- 1. 定位数据集 ----
if [ -z "$RESUME" ] && [ -z "$DATA_ROOT" ]; then
  shopt -s nullglob
  CANDIDATES=("$ROOT"/data/*/meta/info.json)
  shopt -u nullglob
  case ${#CANDIDATES[@]} in
    0) echo "!! data/ 下没有数据集（找不到 */meta/info.json），先把数据集下载到 data/<数据集>"; exit 1 ;;
    1) DATA_ROOT=${CANDIDATES[0]%/meta/info.json} ;;
    *) echo "!! data/ 下有多个数据集，请用 DATA_ROOT 指定其中一个："
       printf '   %s\n' "${CANDIDATES[@]%/meta/info.json}"
       exit 1 ;;
  esac
fi

# ---- 2. 前置检查：文件、依赖、显卡、任务文本、磁盘 ----
if [ -z "$RESUME" ]; then
  [ -f "$BASE_MODEL/model.safetensors" ] || { echo "!! 底座模型不在：$BASE_MODEL/model.safetensors"; exit 1; }
fi
if [ "$POLICY" = pi05 ]; then
  [ -f "$ROOT/models/paligemma_tokenizer/tokenizer.model" ] || { echo "!! 分词器不在：models/paligemma_tokenizer/"; exit 1; }
fi

python - "${DATA_ROOT:-}" "${TASK:-}" "$POLICY" "$POLICY_CFG" "$NUM_GPUS" <<'PY' || exit 1
import json
import sys
from pathlib import Path

data_root, task, policy, policy_cfg, num_gpus = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
ok = True
try:
    import accelerate  # noqa: F401
except ImportError:
    print('!! 缺 accelerate（lerobot 的训练依赖），在项目目录执行：pip install -e "./runtime/lerobot[training]"')
    ok = False
import torch

for i in range(num_gpus):
    p = torch.cuda.get_device_properties(i)
    print(f"显卡 {i}: {p.name}  {p.total_memory / 2**30:.0f}GB")
if policy == "smolvla":
    # SmolVLA 加载时要从本地缓存读 VLM 骨干的配置、处理器和权重
    vlm = json.load(open(policy_cfg)).get("vlm_model_name")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(vlm, local_files_only=True)
    except Exception:
        print(f"!! 本地缓存里没有 {vlm}，先执行：hf download {vlm}")
        ok = False
if data_root:
    import pandas as pd

    tasks = pd.read_parquet(Path(data_root) / "meta" / "tasks.parquet").index.tolist()
    if task and task not in tasks:
        print(f"注意：station.env 的 TASK 不在数据集里。训练照常用数据集里的任务文本，推理时 TASK 必须改成其中之一：")
        for t in tasks:
            print(f"   {t}")
sys.exit(0 if ok else 1)
PY

if [ -z "$RESUME" ]; then
  N_CKPT=$(( (STEPS + SAVE_FREQ - 1) / SAVE_FREQ ))
  NEED_GB=$(( N_CKPT * CKPT_GB ))
  FREE_GB=$(df -BG --output=avail "$ROOT/outputs/train" | tail -1 | tr -dc 0-9)
  if [ "$FREE_GB" -lt "$NEED_GB" ]; then
    echo "!! outputs/ 所在磁盘剩 ${FREE_GB}GB，本次会存 ${N_CKPT} 个 checkpoint，约需 ${NEED_GB}GB。"
    echo "   清理磁盘，或调大 SAVE_FREQ 少存几个。"
    exit 1
  fi
fi

# ---- 3. 组装命令 ----
# 相机改名是 k=v 写法，这里转成 lerobot 的 {k: v} 写法（与 start_inference.sh 相同）
if [ "$POLICY" = smolvla ]; then
  MAP=${SMOLVLA_RENAME_MAP:-observation.images.top=observation.images.camera1}
else
  MAP=${RENAME_MAP:-}
fi
RENAME_MAP_DICT=""
if [ -n "$MAP" ]; then
  RENAME_MAP_DICT="{$(printf '%s' "$MAP" | sed -e 's/[[:space:]]*=[[:space:]]*/: /g')}"
fi

# 多卡：每张卡一个进程，各自 batch BATCH_SIZE，梯度在卡间平均；单卡直接用 python 跑
if [ "$NUM_GPUS" -gt 1 ]; then
  LAUNCH=(python -m accelerate.commands.launch --multi_gpu --num_processes="$NUM_GPUS" --num_machines=1
          --mixed_precision=no --dynamo_backend=no --main_process_port="$MASTER_PORT")
else
  LAUNCH=(python)
fi

# 视频用 pyav 解码（自带 FFmpeg）：torchcodec 在数据加载进程里会和已加载的 libjpeg 冲突，读不了视频
if [ -n "$RESUME" ]; then
  OUTPUT_DIR=$(cd "$RESUME" && pwd)
  CMD=("${LAUNCH[@]}" "$ROOT/train_policy.py" --resume=true --config_path="$RESUME_CFG")
else
  OUTPUT_DIR=$ROOT/outputs/train/$JOB_NAME
  CMD=("${LAUNCH[@]}" "$ROOT/train_policy.py"
    --policy.path="$BASE_MODEL"
    --policy.device=cuda
    --policy.push_to_hub=false
    --dataset.repo_id="${DATASET_ID:?station.env 里要有 DATASET_ID}"
    --dataset.root="$DATA_ROOT"
    --dataset.video_backend=pyav
    --output_dir="$OUTPUT_DIR"
    --job_name="$JOB_NAME"
    --steps="$STEPS"
    --batch_size="$BATCH_SIZE"
    --save_freq="$SAVE_FREQ"
    --log_freq="$LOG_FREQ"
    --num_workers="$NUM_WORKERS"
    --wandb.enable=false)
  if [ "$POLICY" = pi05 ]; then
    CMD+=(--policy.dtype=bfloat16 --policy.train_expert_only=true
          --policy.gradient_checkpointing=$([ "$GRAD_CKPT" = 1 ] && echo true || echo false))
  fi
  [ -n "$RENAME_MAP_DICT" ] && CMD+=(--rename_map="$RENAME_MAP_DICT")
fi
CMD+=("$@")

echo "=================================================="
echo "SO101 $POLICY 训练   $(date '+%F %T')$([ "$SMOKE" = "1" ] && echo '   【冒烟测试】')"
echo "显卡: ${NUM_GPUS} 张$([ "$NUM_GPUS" -gt 1 ] && echo "（accelerate 多卡，端口 $MASTER_PORT${NCCL_P2P_DISABLE:+，NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE}）")"
if [ -n "$RESUME" ]; then
  echo "续训: $RESUME"
else
  echo "数据: $DATA_ROOT"
  echo "底座: $BASE_MODEL"
  echo "参数: ${STEPS} 步 | 每卡 batch ${BATCH_SIZE} × ${NUM_GPUS} 卡 = ${TOTAL_BATCH} | 每 ${SAVE_FREQ} 步存一次 | 每卡 ${NUM_WORKERS} 个数据进程${MAP:+ | rename_map=$MAP}"
fi
echo "输出: $OUTPUT_DIR"
echo "日志: $LOG"
echo "=================================================="
if [ -z "${TMUX:-}" ] && [ -z "${STY:-}" ]; then
  echo "提示：当前不在 screen/tmux 里，SSH 一断训练就停。正式训练前先 tmux new -s train 再跑。"
fi
echo "将执行："
printf '  %q' "${CMD[@]}"; echo; echo
[ "$PRECHECK_ONLY" = "1" ] && { echo "PRECHECK_ONLY=1：前置检查通过，未启动训练。"; exit 0; }

{
  echo "# SO101 train $(date '+%F %T')"
  echo "# cmd: $(printf '%q ' "${CMD[@]}")"
} > "$LOG"

"${CMD[@]}" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "lerobot_train 退出码: $RC"
echo "日志: $LOG"

# 把训练时的 station.env 存进输出目录，推理机拿到模型时能查到任务文本和相机改名
[ -d "$OUTPUT_DIR" ] && [ -z "$RESUME" ] && cp "$STATION_ENV" "$OUTPUT_DIR/station_$STAMP.env"

if [ "$RC" -eq 0 ]; then
  echo "checkpoint 在: $OUTPUT_DIR/checkpoints/"
  ls "$OUTPUT_DIR/checkpoints/" 2>/dev/null | sed 's/^/   /'
  if [ "$SMOKE" = "1" ]; then
    echo "冒烟测试通过。这次的输出没用，删掉腾空间：rm -rf $OUTPUT_DIR"
  fi
fi
exit "$RC"
