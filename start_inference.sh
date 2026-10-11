#!/bin/bash
# SO101 自主推理启动器（单顶部相机）
#
# 与 start_collection.sh 同一套约定：设备、相机、任务都读 station.env，
# 同样的推流端口与日志位置。区别只是把 lerobot_record 换成 lerobot_rollout。
#
# 用法：
#   bash start_inference.sh                                       # 只做前置检查，不接机械臂
#   MODEL_PATH=models/xxx/pretrained_model DURATION_S=20 bash start_inference.sh   # 真跑
#
# 环境变量（station.env 里的同名项也可在这里临时覆盖）：
#   MODEL_PATH       模型导出目录（含 config.json + model.safetensors）；留空=自动找 models/*/pretrained_model
#   TASK             任务文本，必须与训练/数采时一致（默认取 station.env）
#   DURATION_S       本次时长（秒），默认 20；0=无限（需 ALLOW_INFINITE=1）
#   STEPS            直接用步数指定时长（STEPS=800 @30Hz ≈ 26.7s），优先于 DURATION_S
#   FPS              控制频率（默认取 station.env，与数采一致）
#   DEVICE           cuda / cpu，默认 cuda
#   INFERENCE_TYPE   sync / rtc，默认 sync（大 VLA 用 rtc）
#   RTC_HORIZON      INFERENCE_TYPE=rtc 时的 execution_horizon，默认 10
#   RENAME_MAP       观测改名（k=v,k2=v2 写法），默认取 station.env（top -> base_0_rgb）
#   MAX_RELATIVE_TARGET  单拍最大关节增量(度)，留空=不限制；首跑建议给 10 兜底
#   CHUNK_SIZE         策略每次生成多少步（--policy.chunk_size）
#   N_ACTION_STEPS     每段执行多少步（--policy.n_action_steps，同步路径用）
#   QUEUE_THRESHOLD    RTC 队列降到多少就重算（--inference.queue_threshold，即“采用多少步”）
#   LEAVE_AT_FINAL_POSE=1  结束后停在当前姿态（默认平滑回启动姿态）
#   STREAM=0         关闭内置推流
#   YES=1            跳过交互确认
#   PRECHECK_ONLY=1  只做前置检查并打印将要执行的命令
#   LOAD_POLICY=1    前置检查里额外真加载一次策略
source "$(dirname "${BASH_SOURCE[0]}")/common.sh" || exit 1
set -uo pipefail

STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$ROOT/logs"
LOG=$ROOT/logs/rollout_$STAMP.log

# ---- 参数（均可用环境变量覆盖）----
TASK=${TASK:-}
DURATION_S=${DURATION_S:-20}
FPS=${FPS:-30}
DEVICE=${DEVICE:-cuda}
INFERENCE_TYPE=${INFERENCE_TYPE:-sync}
RTC_HORIZON=${RTC_HORIZON:-10}
RENAME_MAP=${RENAME_MAP:-}
MODEL_PATH=${MODEL_PATH:-}
STREAM=${STREAM:-1}
YES=${YES:-0}
PRECHECK_ONLY=${PRECHECK_ONLY:-0}
LOAD_POLICY=${LOAD_POLICY:-0}
ALLOW_INFINITE=${ALLOW_INFINITE:-0}
LEAVE_AT_FINAL_POSE=${LEAVE_AT_FINAL_POSE:-0}
MAX_RELATIVE_TARGET=${MAX_RELATIVE_TARGET:-}
CHUNK_SIZE=${CHUNK_SIZE:-}
N_ACTION_STEPS=${N_ACTION_STEPS:-}
QUEUE_THRESHOLD=${QUEUE_THRESHOLD:-}

# RENAME_MAP 用 preflight 的 k=v 写法，这里转成 rollout 的 {k: v} 写法
RENAME_MAP_DICT=""
if [ -n "$RENAME_MAP" ]; then
  RENAME_MAP_DICT="{$(printf '%s' "$RENAME_MAP" | sed -e 's/[[:space:]]*=[[:space:]]*/: /g')}"
fi

# 也可以直接给步数：STEPS=800 @30Hz -> DURATION_S=26.667
if [ -n "${STEPS:-}" ]; then
  DURATION_S=$(python3 -c "print(f'{$STEPS/$FPS:.3f}')")
  echo "STEPS=$STEPS @ ${FPS}Hz  ->  DURATION_S=${DURATION_S}s"
fi

export SO101_STREAM=$STREAM
so101_activate_env || exit 1

if [ -z "$TASK" ]; then
  echo "!! TASK 为空：在 station.env 里填任务文本（必须与数采时逐字一致），或 TASK=\"...\" bash start_inference.sh"
  exit 1
fi

echo "=================================================="
echo "SO101 推理启动器   $(date '+%F %T')"
echo "模型: ${MODEL_PATH:-<未指定，将在 models/ 下自动查找>}"
echo "任务: $TASK"
echo "从臂: $FOLLOWER_PORT (id=$FOLLOWER_ID)"
echo "相机: top=$TOP_CAM ${TOP_CAM_WIDTH}x${TOP_CAM_HEIGHT}@${TOP_CAM_FPS} $TOP_CAM_FOURCC（与数采同一套 schema）"
echo "参数: ${DURATION_S}s @ ${FPS}Hz | device=$DEVICE | inference=$INFERENCE_TYPE${RENAME_MAP:+ | rename_map=$RENAME_MAP}"
echo "日志: $LOG"
echo "=================================================="

# ---- 1. 定位模型（纯查目录，不碰硬件）----
if [ -z "$MODEL_PATH" ]; then
  shopt -s nullglob
  CANDIDATES=("$ROOT"/models/*/pretrained_model)
  shopt -u nullglob
  case ${#CANDIDATES[@]} in
    0) : ;;
    1) MODEL_PATH=${CANDIDATES[0]}; echo "自动选用模型: $MODEL_PATH" ;;
    *) echo "!! models/ 下有多个模型，请用 MODEL_PATH 指定其中一个："
       printf '   %s\n' "${CANDIDATES[@]}"
       exit 1 ;;
  esac
fi

# ---- 2. 前置检查：不通过就不接硬件 ----
PRE_ARGS=(--cameras "$CAMERA_NAMES" --device "$DEVICE" --fps "$FPS" --duration "$DURATION_S" --inference "$INFERENCE_TYPE" --task "$TASK")
[ -n "$MODEL_PATH" ] && PRE_ARGS+=(--model-dir "$MODEL_PATH" --manifest "$ROOT/logs/rollout_$STAMP.json")
[ -n "$RENAME_MAP" ] && PRE_ARGS+=(--rename-map "$RENAME_MAP")
[ "$LOAD_POLICY" = "1" ] && PRE_ARGS+=(--load)

echo
python "$ROOT/inference_preflight.py" "${PRE_ARGS[@]}"
PRE_RC=$?
echo
if [ "$PRE_RC" -ne 0 ]; then
  echo "!! 前置检查未通过（退出码 $PRE_RC），不接硬件。"
  exit "$PRE_RC"
fi

# ---- 3. 还没有模型：到此为止 ----
if [ -z "$MODEL_PATH" ]; then
  echo "还没有可用模型：$ROOT/models/ 下没有 <名字>/pretrained_model 目录。"
  echo "把训练导出的模型目录放进去（例如 models/my_task_030000/pretrained_model/）再跑本脚本即可。"
  echo "本次只完成环境/设备自检，未连接机械臂。"
  exit 0
fi

# ---- 4. 时长守卫 ----
if [ "$DURATION_S" = "0" ] && [ "$ALLOW_INFINITE" != "1" ]; then
  echo "!! DURATION_S=0 在引擎里表示无限跑（24/7）。真要这样请显式加 ALLOW_INFINITE=1"
  exit 1
fi

# ---- 5. 组装命令 ----
if [ "$LEAVE_AT_FINAL_POSE" = "1" ]; then
  RETURN_TO_INITIAL=false
else
  RETURN_TO_INITIAL=true
fi

CMD=(python "$ROOT/rollout_with_stream.py"
  --strategy.type=base
  --inference.type="$INFERENCE_TYPE"
  --policy.path="$MODEL_PATH"
  --policy.device="$DEVICE"
  --device="$DEVICE"
  --robot.type=so101_follower
  --robot.port="$FOLLOWER_PORT"
  --robot.id="$FOLLOWER_ID"
  --robot.cameras="$CAMERAS"
  --task="$TASK"
  --fps="$FPS"
  --duration="$DURATION_S"
  --display_data=false
  --play_sounds=false
  --return_to_initial_position="$RETURN_TO_INITIAL")
[ "$INFERENCE_TYPE" = "rtc" ] && CMD+=(--inference.rtc.execution_horizon="$RTC_HORIZON")
[ -n "$RENAME_MAP" ] && CMD+=(--rename_map="$RENAME_MAP_DICT")
[ -n "$MAX_RELATIVE_TARGET" ] && CMD+=(--robot.max_relative_target="$MAX_RELATIVE_TARGET")
[ -n "$CHUNK_SIZE" ] && CMD+=(--policy.chunk_size="$CHUNK_SIZE")
[ -n "$N_ACTION_STEPS" ] && CMD+=(--policy.n_action_steps="$N_ACTION_STEPS")
if [ "$INFERENCE_TYPE" = "rtc" ] && [ -n "$QUEUE_THRESHOLD" ]; then
  CMD+=(--inference.queue_threshold="$QUEUE_THRESHOLD")
fi

echo "将执行："
printf '  %q' "${CMD[@]}"; echo; echo

if [ "$PRECHECK_ONLY" = "1" ]; then
  echo "PRECHECK_ONLY=1：只做检查与打印，未连接机械臂。"
  exit 0
fi

# ---- 6. 真跑前的硬门槛 ----
for dev in "$FOLLOWER_PORT" "$TOP_CAM"; do
  if [ ! -e "$dev" ]; then
    echo "!! 设备不在位: $dev —— 检查供电与 USB 接线后再跑。"
    exit 1
  fi
done

cat <<'NOTE'
=== 看护须知 ===
1. 工作区清空，人站在能立刻断电的位置；
2. 第一次跑先给 DURATION_S=5~10，全程盯住机械臂；
3. Ctrl+C 随时停；结束后默认平滑回到启动时的姿态。
=================
NOTE

if [ "$YES" != "1" ]; then
  printf "确认开始自主推理？输入 yes 继续: "
  read -r ANSWER
  if [ "$ANSWER" != "yes" ]; then
    echo "已取消，未连接机械臂。"
    exit 0
  fi
fi

# ---- 7. 独立监控流让位，起推理由程（内置推流接同一端口）----
if pkill -f "[m]jpeg_server"; then
  echo "独立监控流已让位（推理期间 $STREAM_PORT 端口由推理进程内置推流接管）"
else
  echo "独立监控流本来就没在跑"
fi
sleep 1

{
  echo "# SO101 rollout $(date '+%F %T')"
  echo "# model: $MODEL_PATH"
  echo "# task: $TASK"
  echo "# cmd: $(printf '%q ' "${CMD[@]}")"
} > "$LOG"

set +e
"${CMD[@]}" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
set -e

echo "lerobot_rollout 退出码: $RC"
echo "日志: $LOG"
exit "$RC"❯