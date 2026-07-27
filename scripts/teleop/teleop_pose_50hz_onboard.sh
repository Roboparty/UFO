#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[teleop_pose_50hz_onboard] $*"
}

fail() {
  echo "[teleop_pose_50hz_onboard] ERROR: $*" >&2
  exit 1
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_UFO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

UFO_ROOT="${UFO_ROOT:-${DEFAULT_UFO_ROOT}}"
TELEOP_PY="${TELEOP_PY:-}"
TELEOP_ALLOW_SYSTEM_PY="${TELEOP_ALLOW_SYSTEM_PY:-0}"
START_XROBOT_SERVICE="${START_XROBOT_SERVICE:-0}"
WEB_VISUALIZE="${WEB_VISUALIZE:-0}"
WEB_PORT="${WEB_PORT:-8080}"
WEB_MUJOCO_XML="${WEB_MUJOCO_XML:-${UFO_ROOT}/data/robots/g1/scene_29dof_freebase.xml}"
CTRL_PUB_BIND_ADDR="${CTRL_PUB_BIND_ADDR:-}"
ENABLE_PICO_POLICY_CONTROL="${ENABLE_PICO_POLICY_CONTROL:-0}"
TELEOP_POLICY_CONFIG="${TELEOP_POLICY_CONFIG:-${POLICY_CONFIG:-${UFO_ROOT}/config/policy/g1_policy.yaml}}"
export TELEOP_POLICY_CONFIG

[[ -d "${UFO_ROOT}" ]] || fail "UFO_ROOT does not exist: ${UFO_ROOT}"
[[ -f "${UFO_ROOT}/scripts/teleop/xrobot_teleop_to_pose_zmq_server.py" ]] || \
  fail "missing teleop server under UFO_ROOT: ${UFO_ROOT}"
if [[ -n "${CTRL_PUB_BIND_ADDR}" && "${ENABLE_PICO_POLICY_CONTROL}" != "1" ]]; then
  fail "CTRL_PUB_BIND_ADDR was set but ENABLE_PICO_POLICY_CONTROL is not 1. Legacy/debug 28704 policy control requires both variables."
fi

ld_parts=()
for candidate in \
  "${UFO_ROOT}/external/XRoboToolkit-PC-Service-Pybind/lib/aarch64" \
  "${UFO_ROOT}/external/XRoboToolkit-PC-Service-Pybind/lib"; do
  [[ -d "${candidate}" ]] && ld_parts+=("${candidate}")
done

if (( ${#ld_parts[@]} > 0 )); then
  export LD_LIBRARY_PATH="$(IFS=:; echo "${ld_parts[*]}"):${LD_LIBRARY_PATH:-}"
fi
export PYTHONPATH="${UFO_ROOT}/scripts/teleop:${PYTHONPATH:-}"

python_has_teleop_deps() {
  local python_bin="$1"

  if [[ ! -x "${python_bin}" ]] && ! command -v "${python_bin}" >/dev/null 2>&1; then
    return 1
  fi

  "${python_bin}" "${UFO_ROOT}/scripts/teleop/check_teleop_env.py" \
    --skip-service \
    --skip-ports \
    >/dev/null 2>&1
}

if [[ -z "${TELEOP_PY}" ]]; then
  for candidate in \
    "${UFO_ROOT}/.venv/bin/python" \
    "${UFO_ROOT}/venv/bin/python" \
    "/home/unitree/ufo_teleop_venv/bin/python" \
    "/home/unitree/ufo_deploy_venv/bin/python" \
    "/home/unitree/miniconda3/envs/ufo-teleop/bin/python" \
    "/home/unitree/miniconda3/envs/ufo-deploy/bin/python"; do
    if [[ ! -x "${candidate}" ]]; then
      continue
    fi
    if python_has_teleop_deps "${candidate}"; then
      TELEOP_PY="${candidate}"
      break
    fi
    log "skipping Python without onboard teleop deps: ${candidate}"
  done
fi

if [[ -z "${TELEOP_PY}" ]]; then
  if [[ "${TELEOP_ALLOW_SYSTEM_PY}" == "1" ]]; then
    if command -v python3 >/dev/null 2>&1 && python_has_teleop_deps python3; then
      TELEOP_PY="python3"
    elif command -v python >/dev/null 2>&1 && python_has_teleop_deps python; then
      TELEOP_PY="python"
    fi
  fi
fi

if [[ -z "${TELEOP_PY}" ]]; then
  fail "no UFO teleop Python environment found. Create /home/unitree/ufo_teleop_venv, ${UFO_ROOT}/.venv, or set TELEOP_PY=/path/to/python. Refusing system python by default; set TELEOP_ALLOW_SYSTEM_PY=1 only for debugging."
fi

if [[ ! -x "${TELEOP_PY}" ]] && ! command -v "${TELEOP_PY}" >/dev/null 2>&1; then
  fail "python not found or not executable: ${TELEOP_PY}"
fi

if ! python_has_teleop_deps "${TELEOP_PY}"; then
  "${TELEOP_PY}" "${UFO_ROOT}/scripts/teleop/check_teleop_env.py" \
    --skip-service \
    --skip-ports \
    || true
  fail "selected Python failed onboard teleop dependency probe: ${TELEOP_PY}"
fi

service_pattern='RoboticsServiceProcess|roboticsservice|XRoboToolkit'
if ! pgrep -f "${service_pattern}" >/dev/null 2>&1; then
  if [[ "${START_XROBOT_SERVICE}" == "1" && -x /opt/apps/roboticsservice/runService.sh ]]; then
    log "XRoboToolkit service not detected; starting /opt/apps/roboticsservice/runService.sh"
    bash /opt/apps/roboticsservice/runService.sh >/tmp/ufo-roboticsservice-start.log 2>&1 || \
      fail "failed to start XRoboToolkit service; see /tmp/ufo-roboticsservice-start.log"
    sleep 2
  else
    fail "XRoboToolkit service is not running. Start it first with 'bash /opt/apps/roboticsservice/runService.sh', or explicitly run START_XROBOT_SERVICE=1 $0 after verifying the installed service version."
  fi
fi

check_ports=(28701 28702 28703)
if [[ -n "${CTRL_PUB_BIND_ADDR}" ]]; then
  check_ports+=(28704)
fi

yaml_visualize="$("${TELEOP_PY}" - "${UFO_ROOT}/config/teleop/g1.yaml" <<'PY'
from pathlib import Path
import sys
import yaml

path = Path(sys.argv[1])
try:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    print("1" if bool((cfg or {}).get("server", {}).get("visualize", False)) else "0")
except Exception:
    print("0")
PY
)"
if [[ "${WEB_VISUALIZE}" == "1" || "${yaml_visualize}" == "1" ]]; then
  check_ports+=("${WEB_PORT}")
  [[ -f "${WEB_MUJOCO_XML}" ]] || fail "missing web MuJoCo XML: ${WEB_MUJOCO_XML}"
fi

log "repo: ${UFO_ROOT}"
log "python: ${TELEOP_PY}"
log "checking teleop environment"
"${TELEOP_PY}" "${UFO_ROOT}/scripts/teleop/check_teleop_env.py" --ports "${check_ports[@]}" || \
  fail "teleop environment preflight failed"

cd "${UFO_ROOT}/scripts/teleop"

cmd=(
  "${TELEOP_PY}" xrobot_teleop_to_pose_zmq_server.py
  --robot unitree_g1
  --xr-poll-hz "${XR_POLL_HZ:-50}"
  --req_bind_addr tcp://*:28701
  --rep_bind_addr tcp://*:28702
  --ctrl_bind_addr tcp://*:28703
  --policy-config "${TELEOP_POLICY_CONFIG}"
)

[[ -n "${ACTUAL_HUMAN_HEIGHT:-}" ]] && cmd+=(--actual_human_height "${ACTUAL_HUMAN_HEIGHT}")
[[ -n "${CTRL_FPS:-}" ]] && cmd+=(--ctrl_fps "${CTRL_FPS}")
[[ -n "${LOOKBACK_MS:-}" ]] && cmd+=(--lookback_ms "${LOOKBACK_MS}")
[[ -n "${RETARGET_BUFFER_WINDOW_S:-}" ]] && cmd+=(--retarget_buffer_window_s "${RETARGET_BUFFER_WINDOW_S}")
[[ -n "${LOG_INTERVAL_S:-}" ]] && cmd+=(--log_interval_s "${LOG_INTERVAL_S}")
[[ -n "${VIS_FPS:-}" ]] && cmd+=(--vis_fps "${VIS_FPS}")

if [[ -n "${CTRL_PUB_BIND_ADDR}" ]]; then
  cmd+=(--ctrl_pub_bind_addr "${CTRL_PUB_BIND_ADDR}")
fi

if [[ "${WEB_VISUALIZE}" == "1" ]]; then
  cmd+=(--web-visualize --web-port "${WEB_PORT}" --web-mujoco-xml "${WEB_MUJOCO_XML}")
fi

log "starting PICO -> canonical retarget -> ZMQ pose bridge"
exec "${cmd[@]}" "$@"
