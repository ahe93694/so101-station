#!/bin/bash
# SO101 π0.5 训练启动器（在云服务器上跑，不接机械臂和相机）
#
# 从 models/pi05_base 微调：VLM 冻结，只训动作专家，bf16。32GB 显存放得下；
# 全量微调要 70GB 以上，换大显存机器后可以加 --policy.train_expert_only=false。
# 数据集名、相机改名、任务文本读 station.env；数据集默认取 data/ 下唯一的那个。
#
# 用法（先开 screen 或 tmux，SSH 断了训练不会停）：
#   SMOKE=1 bash start_training.sh                      # 冒烟测试：跑 20 步并存一次，确认数据、显存、保存都正常
#   bash start_training.sh                              # 正式训练
#   RESUME=outputs/train/<名字> bash start_training.sh  # 中断后从最近一次 checkpoint 接着训
#
# 环境变量：
#   DATA_ROOT    数据集目录，留空=自动找 data/ 下唯一的数据集
#   BASE_MODEL   底座模型目录，默认 models/pi05_base
#   STEPS        训练步数，默认 10000（50 集约 2.3 万帧，batch 32 约 14 轮）
#   BATCH_SIZE   默认 32，爆显存就减半
#   SAVE_FREQ    每多少步存一次 checkpoint，默认 5000；每个约 10GB，启动前按数据盘剩余空间检查
#   NUM_WORKERS  数据加载进程数，默认 8
#   JOB_NAME     输出目录名（outputs/train/ 下），默认 pi05_<时间>
# 其余参数原样传给 lerobot-train，例如：bash start_training.sh --policy.optimizer_lr=5e-5
source "$(dirname "${BASH_SOURCE[0]}")/common.sh" || exit 1
set -uo pipefail
so101_activate_env || exit 1
cd "$ROOT" || exit 1

STAMP=$(date +%Y%m%d_%H%M%S)
SMOKE=${SMOKE:-0}
RESUME=${RESUME:-}
DATA_ROOT=${DATA_ROOT:-}
BASE_MODEL=${BASE_MODEL:-$ROOT/models/pi05_base}
STEPS=${STEPS:-10000}
BATCH_SIZE=${BATCH_SIZE:-32}
SAVE_FREQ=${SAVE_FREQ:-5000}
NUM_WORKERS=${NUM_WORKERS:-8}
LOG_FREQ=${LOG_FREQ:-100}
JOB_NAME=${JOB_NAME:-pi05_$STAMP}
CKPT_GB=10   # 一个 checkpoint 的大致体积：bf16 模型约 8GB + 优化器状态
if [ "$SMOKE" = "1" ]; then
  STEPS=20; SAVE_FREQ=20; LOG_FREQ=5; JOB_NAME=smoke_$STAMP
fi

# 底座模型、分词器、数据集都在本地，禁止联网：缺文件时立刻报错，而不是卡在下载超时上
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$ROOT/logs" "$ROOT/outputs/train"
LOG=$ROOT/logs/train_$STAMP.log

# ---- 1. 定位数据集 ----
if [ -z "$RESUME" ] && [ -z "$DATA_ROOT" ]; then
  shopt -s nullglob
  CANDIDATES=("$ROOT"/data/*/meta/info.json)
  shopt -u nullglob
  case ${#CANDIDATES[@]} in
    0) echo "!! data/ 下没有数据集（找不到 */meta/info.json），先把数采机上的 data/<数据集> 传过来"; exit 1 ;;
    1) DATA_ROOT=${CANDIDATES[0]%/meta/info.json} ;;
    *) echo "!! data/ 下有多个数据集，请用 DATA_ROOT 指定其中一个："
       printf '   %s\n' "${CANDIDATES[@]%/meta/info.json}"
       exit 1 ;;
  esac
fi

# ---- 2. 前置检查：文件、依赖、显卡、任务文本、磁盘 ----
if [ -n "$RESUME" ]; then
  RESUME_CFG=$RESUME/checkpoints/last/pretrained_model/train_config.json
  [ -f "$RESUME_CFG" ] || { echo "!! 找不到 $RESUME_CFG，RESUME 要指向 outputs/train/<名字>"; exit 1; }
else
  [ -f "$BASE_MODEL/model.safetensors" ] || { echo "!! 底座模型不在：$BASE_MODEL/model.safetensors"; exit 1; }
fi
[ -f "$ROOT/models/paligemma_tokenizer/tokenizer.model" ] || { echo "!! 分词器不在：models/paligemma_tokenizer/"; exit 1; }

python - "${DATA_ROOT:-}" "${TASK:-}" <<'PY' || exit 1
import sys
from pathlib import Path

data_root, task = sys.argv[1], sys.argv[2]
ok = True
try:
    import accelerate  # noqa: F401
except ImportError:
    print('!! 缺 accelerate（lerobot 的训练依赖），在项目目录执行：pip install -e "./runtime/lerobot[training]"')
    ok = False
import torch

if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"显卡: {p.name}  {p.total_memory / 2**30:.0f}GB")
else:
    print("!! torch 看不到 CUDA 显卡")
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
    echo "   在 AutoDL 控制台给数据盘扩容，或调大 SAVE_FREQ 少存几个。"
    exit 1
  fi
fi

# ---- 3. 组装命令 ----
# RENAME_MAP 是 k=v 写法，这里转成 lerobot 的 {k: v} 写法（与 start_inference.sh 相同）
RENAME_MAP_DICT=""
if [ -n "${RENAME_MAP:-}" ]; then
  RENAME_MAP_DICT="{$(printf '%s' "$RENAME_MAP" | sed -e 's/[[:space:]]*=[[:space:]]*/: /g')}"
fi

# 视频用 pyav 解码（自带 FFmpeg）：torchcodec 在数据加载进程里会和已加载的 libjpeg 冲突，读不了视频
if [ -n "$RESUME" ]; then
  OUTPUT_DIR=$(cd "$RESUME" && pwd)
  CMD=(python "$ROOT/train_policy.py" --resume=true --config_path="$RESUME_CFG")
else
  OUTPUT_DIR=$ROOT/outputs/train/$JOB_NAME
  CMD=(python "$ROOT/train_policy.py"
    --policy.path="$BASE_MODEL"
    --policy.device=cuda
    --policy.push_to_hub=false
    --policy.dtype=bfloat16
    --policy.train_expert_only=true
    --policy.gradient_checkpointing=true
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
  [ -n "$RENAME_MAP_DICT" ] && CMD+=(--rename_map="$RENAME_MAP_DICT")
fi
CMD+=("$@")

echo "=================================================="
echo "SO101 π0.5 训练   $(date '+%F %T')$([ "$SMOKE" = "1" ] && echo '   【冒烟测试】')"
if [ -n "$RESUME" ]; then
  echo "续训: $RESUME"
else
  echo "数据: $DATA_ROOT"
  echo "底座: $BASE_MODEL"
  echo "参数: ${STEPS} 步 | batch ${BATCH_SIZE} | 每 ${SAVE_FREQ} 步存一次${RENAME_MAP:+ | rename_map=$RENAME_MAP}"
fi
echo "输出: $OUTPUT_DIR"
echo "日志: $LOG"
echo "=================================================="
if [ -z "${TMUX:-}" ] && [ -z "${STY:-}" ]; then
  echo "提示：当前不在 screen/tmux 里，SSH 一断训练就停。正式训练前先 screen -S train 再跑。"
fi
echo "将执行："
printf '  %q' "${CMD[@]}"; echo; echo

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
