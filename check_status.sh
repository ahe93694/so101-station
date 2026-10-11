#!/bin/bash
# SO101 现状盘点：只读，不碰机械臂、不改任何文件。
source "$(dirname "${BASH_SOURCE[0]}")/common.sh" || exit 1
y(){ [ -e "$1" ] && echo "[有] $1" || echo "[无] $1"; }
echo "用户=$(whoami)  主机=$(hostname)  会话=${XDG_SESSION_TYPE:-?}  日期=$(date '+%F %T')"
echo "项目=$ROOT  配置=$STATION_ENV"

echo "== 1 环境 =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "[无] nvidia-smi"
groups | grep -qw dialout && echo "[有] dialout 组" || echo "[无] dialout 组（串口要 sudo usermod -aG dialout \$USER 后重新登录）"
so101_activate_env \
  && python -c "import torch,lerobot;print('torch',torch.__version__,'cuda',torch.cuda.is_available());print('lerobot',lerobot.__file__)" \
  && (python -c "import lerobot.policies.pi05.modeling_pi05" 2>/dev/null && echo "[有] pi05 依赖" || echo "[无] pi05 依赖") \
  || echo "[无] python 环境 ${CONDA_ENV:-（当前 python）}"

echo "== 2 硬件：station.env 里写的设备是否插着 =="
echo "从臂 $(y "$FOLLOWER_PORT")"
echo "主臂 $(y "$LEADER_PORT")"
echo "顶部相机 $(y "$TOP_CAM")"
echo "-- 实际插着的串口："; ls -l /dev/serial/by-path/ 2>/dev/null | grep -o 'pci-.*' || echo "（无）"
echo "-- 实际插着的相机："; ls -l /dev/v4l/by-path/ 2>/dev/null | grep -o 'pci-.*' || echo "（无）"

echo "== 3 标定 / 数据 / 模型 =="
y "$ROOT/calibration/robots/so_follower/$FOLLOWER_ID.json"
y "$ROOT/calibration/teleoperators/so_leader/$LEADER_ID.json"
for d in "$ROOT"/data/*/; do
  [ -d "$d" ] || continue
  python3 -c "import json,sys;d=json.load(open(sys.argv[1]+'meta/info.json'));print(d['total_episodes'],'集',d['total_frames'],'帧',sys.argv[1])" "$d" 2>/dev/null || echo "?? $d"
done
ls -d "$ROOT"/models/*/pretrained_model 2>/dev/null || echo "（models/ 下没有 pretrained_model）"
echo "TASK=${TASK:-（未填）}"
echo "== 完 =="
