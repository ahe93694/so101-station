# SO101 启动脚本公共部分：被 start_*.sh / check_status.sh source，不单独运行。
#
# 做三件事：
#   1. ROOT = 本文件所在目录（项目放在哪都能跑）
#   2. 读 station.env（同名环境变量优先）并 export
#   3. 激活 conda 环境 CONDA_ENV（留空则用当前 python）

SO101_OVERRIDES=
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STATION_ENV=${STATION_ENV:-$ROOT/station.env}

if [ ! -f "$STATION_ENV" ]; then
  echo "!! 找不到配置文件 $STATION_ENV" >&2
  return 1 2>/dev/null || exit 1
fi

while IFS= read -r _line || [ -n "$_line" ]; do
  _line=${_line%$'\r'}
  case "$_line" in ''|'#'*) continue ;; esac
  _key=${_line%%=*}
  _val=${_line#*=}
  _key=$(printf '%s' "$_key" | tr -d '[:space:]')
  case "$_key" in ''|*[!A-Za-z0-9_]*) continue ;; esac
  _val=${_val#"${_val%%[![:space:]]*}"}
  _val=${_val%"${_val##*[![:space:]]}"}
  case "$_val" in \"*\") _val=${_val:1:${#_val}-2} ;; \'*\') _val=${_val:1:${#_val}-2} ;; esac
  if [ -z "${!_key+x}" ]; then
    export "$_key=$_val"
  elif [ "${!_key}" != "$_val" ]; then
    SO101_OVERRIDES+="  $_key=${!_key}"$'\n'"      （station.env 里是 $_key=$_val）"$'\n'
  fi
done < "$STATION_ENV"
unset _line _key _val

# 终端里残留的同名变量（例如之前 source 过 common.sh）会悄悄盖掉 station.env，必须让人看见
if [ -n "${SO101_OVERRIDES:-}" ]; then
  printf '!! 注意：下面这些值来自终端里的环境变量，不是 station.env：\n%s' "$SO101_OVERRIDES" >&2
  printf '   如果不是你有意临时覆盖，请开一个新终端再运行（或 unset 这些变量）\n' >&2
fi

export SO101_ROOT=$ROOT
export HF_LEROBOT_CALIBRATION=$ROOT/calibration
export HF_LEROBOT_HOME=$ROOT/.lerobot_cache
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1   # 输出实时刷新（管道 tee 时不再攒块）

# 顶部相机的 lerobot 写法（数采与推理共用，保证 schema 一致）
TOP_CAM_SPEC="top: {type: opencv, index_or_path: $TOP_CAM, width: $TOP_CAM_WIDTH, height: $TOP_CAM_HEIGHT, fps: $TOP_CAM_FPS${TOP_CAM_FOURCC:+, fourcc: $TOP_CAM_FOURCC}, warmup_s: 3}"
CAMERAS="{$TOP_CAM_SPEC}"
CAMERA_NAMES=top

so101_activate_env() {
  [ -z "${CONDA_ENV:-}" ] && return 0
  [ "${CONDA_DEFAULT_ENV:-}" = "$CONDA_ENV" ] && return 0
  local base=""
  if command -v conda >/dev/null 2>&1; then
    base=$(conda info --base 2>/dev/null)
  fi
  if [ -z "$base" ]; then
    for d in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge" /opt/conda /opt/miniconda3; do
      [ -f "$d/etc/profile.d/conda.sh" ] && { base=$d; break; }
    done
  fi
  if [ -z "$base" ]; then
    echo "!! 找不到 conda。装好 conda，或在 station.env 里把 CONDA_ENV 留空、先手动激活环境再运行" >&2
    return 1
  fi
  # shellcheck disable=SC1091
  source "$base/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" || { echo "!! conda 环境 $CONDA_ENV 不存在（见 environment.yml）" >&2; return 1; }
}
