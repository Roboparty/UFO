#!/usr/bin/env bash
set -euo pipefail

UFO_ROOT="${UFO_ROOT:-/home/unitree/UFO-Deploy}"
SIM2REAL_ROOT="${SIM2REAL_ROOT:-/home/unitree/sim2real}"
TELEOP_PY="${TELEOP_PY:-}"
WEB_VISUALIZE="${WEB_VISUALIZE:-1}"
WEB_PORT="${WEB_PORT:-8080}"
WEB_MUJOCO_XML="${WEB_MUJOCO_XML:-${UFO_ROOT}/data/robots/g1/scene_29dof_freebase.xml}"

if [[ -z "${TELEOP_PY}" ]]; then
  if [[ -x "${SIM2REAL_ROOT}/venv/teleop/.venv/bin/python" ]]; then
    TELEOP_PY="${SIM2REAL_ROOT}/venv/teleop/.venv/bin/python"
  else
    TELEOP_PY="python"
  fi
fi

if ! command -v "${TELEOP_PY}" >/dev/null 2>&1 && [[ ! -x "${TELEOP_PY}" ]]; then
  echo "[teleop_pose_50hz_onboard] python not found: ${TELEOP_PY}" >&2
  exit 1
fi

if ! pgrep -f RoboticsServiceProcess >/dev/null 2>&1; then
  if [[ -x /opt/apps/roboticsservice/runService.sh ]]; then
    bash /opt/apps/roboticsservice/runService.sh >/tmp/ufo-roboticsservice-start.log 2>&1
    sleep 2
  fi
fi

cd "${UFO_ROOT}/scripts/teleop"

if [[ -d "${SIM2REAL_ROOT}/external/XRoboToolkit-PC-Service-Pybind/lib/aarch64" ]]; then
  export LD_LIBRARY_PATH="${SIM2REAL_ROOT}/external/XRoboToolkit-PC-Service-Pybind/lib/aarch64:${LD_LIBRARY_PATH:-}"
fi
export PYTHONPATH="${UFO_ROOT}/scripts/teleop:${PYTHONPATH:-}"

cmd=(
  "${TELEOP_PY}" xrobot_teleop_to_pose_zmq_server.py
  --robot unitree_g1
  --actual_human_height "${ACTUAL_HUMAN_HEIGHT:-1.6}"
  --ctrl_fps 50
  --xr-poll-hz "${XR_POLL_HZ:-50}"
  --lookback_ms "${LOOKBACK_MS:-25}"
  --retarget_buffer_window_s 0.5
  --log_interval_s "${LOG_INTERVAL_S:-1}"
  --req_bind_addr tcp://*:28701
  --rep_bind_addr tcp://*:28702
  --ctrl_bind_addr tcp://*:28703
  --ctrl_pub_bind_addr tcp://*:28704
  --min_link_height 0.0
  --min_link_height_align_strategy startup_fixed
  --min_link_height_bootstrap_frames 10
  --vis_fps "${VIS_FPS:-10}"
)

if [[ "${WEB_VISUALIZE}" == "1" ]]; then
  cmd+=(--web-visualize --web-port "${WEB_PORT}" --web-mujoco-xml "${WEB_MUJOCO_XML}")
fi

exec "${cmd[@]}" "$@"
