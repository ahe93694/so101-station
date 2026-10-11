#!/bin/bash
# SO101 数采启动器（单顶部相机，支持续采）
#
# 设备、任务等默认值在 station.env；下面的环境变量可临时覆盖：
#   每集开始、保留/重录都由按键决定（→/回车 下一集，← 重录，Esc 结束），见 record_with_stream.py
#   TASK="任务" EPISODES=10 [EPISODE_S=20] [DATASET_ID=local/xxx] [DATA_ROOT=路径] [RESUME=1] \
#     bash start_collection.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh" || exit 1
so101_activate_env || exit 1

TASK=${TASK:?必须设置 TASK（station.env 里填，或 TASK="..." bash start_collection.sh）}
EPISODES=${EPISODES:-10}
EPISODE_S=${EPISODE_S:-20}   # 每集上限；做完可按 → 提前结束
RESET_S=${RESET_S:-30}   # 已不起作用：复位改为按键结束，保留参数只为兼容 lerobot
STAMP=$(date +%Y%m%d_%H%M%S)
DATA_ROOT=${DATA_ROOT:-$ROOT/data/$(echo "$DATASET_ID" | tr '/' '_')_${STAMP}}
RESUME_ARG=""
if [ "${RESUME:-0}" = "1" ]; then
  RESUME_ARG="--resume=true"
  [ -d "$DATA_ROOT" ] || { echo "续采错误: $DATA_ROOT 不存在"; exit 1; }
fi

echo "=================================================="
echo "Dataset root: $DATA_ROOT"
echo "任务:     $TASK"
echo "Follower: $FOLLOWER_PORT (id=$FOLLOWER_ID)"
echo "Leader:   $LEADER_PORT (id=$LEADER_ID)"
echo "Camera:   top=$TOP_CAM ${TOP_CAM_WIDTH}x${TOP_CAM_HEIGHT}@${TOP_CAM_FPS} $TOP_CAM_FOURCC"
echo "本次计划: ${EPISODES}集 ${RESUME_ARG:+(续采)}"
echo "=================================================="

if [ -n "$SO101_OVERRIDES" ] && [ -t 0 ]; then
  read -r -p "上面有值没用 station.env 的，确认就用这些值采集？[y/N] " _ok
  [ "$_ok" = y ] || [ "$_ok" = Y ] || { echo "已取消"; exit 1; }
fi

for dev in "$FOLLOWER_PORT" "$LEADER_PORT" "$TOP_CAM"; do
  [ -e "$dev" ] || { echo "!! 设备不在位: $dev —— 检查供电、USB 接线，或改 station.env"; exit 1; }
done

# 相机是独占设备：独立监控流必须让位，采集期间改由采集进程内置推流顶替（同一端口，观看不中断）
pkill -f "[m]jpeg_server" && echo "独立监控流已让位给采集进程（端口 $STREAM_PORT 由内置推流接管）" || true
sleep 1

python "$ROOT/record_with_stream.py" \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --robot.cameras="$CAMERAS" \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id="$LEADER_ID" \
  --dataset.repo_id="$DATASET_ID" \
  --dataset.root="$DATA_ROOT" \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes="$EPISODES" \
  --dataset.episode_time_s="$EPISODE_S" \
  --dataset.reset_time_s="$RESET_S" \
  --dataset.fps="$FPS" \
  --dataset.video=true \
  --dataset.streaming_encoding=false \
  --dataset.encoder_threads=2 \
  --dataset.push_to_hub=false \
  --dataset.private=false \
  --display_data=false \
  --play_sounds=false \
  $RESUME_ARG
RC=$?
echo "lerobot_record 退出码: $RC"

# 把本次实际生效的配置存进数据集，换机器训练时能查到当初的设备与任务
if [ -d "$DATA_ROOT/meta" ]; then
  ARCHIVE="$DATA_ROOT/meta/station_${STAMP}.env"
  {
    cat "$STATION_ENV"
    printf '\n# ==== 本次实际生效的值（命令行 / 终端环境变量可能覆盖了上面）====\n'
    printf 'TASK="%s"\nDATASET_ID=%s\nEPISODES=%s\nEPISODE_S=%s\nFPS=%s\n' "$TASK" "$DATASET_ID" "$EPISODES" "$EPISODE_S" "$FPS"
  } > "$ARCHIVE" && echo "配置已存档: $ARCHIVE"
  # lerobot 写出的视频是 600 权限，放开读权限，方便拷到训练机、给别人检查
  chmod -R a+rX "$DATA_ROOT"
fi
exit $RC
