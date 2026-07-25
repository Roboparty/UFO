#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UFO_ROOT="${UFO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
TELEOP_POLICY_CONFIG="${TELEOP_POLICY_CONFIG:-${POLICY_CONFIG:-${UFO_ROOT}/config/policy/g1_policy.yaml}}"
export TELEOP_POLICY_CONFIG

cd "${SCRIPT_DIR}"

cmd=(
    python xrobot_teleop_to_pose_zmq_server.py
    --robot unitree_g1
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

if [[ "${WEB_VISUALIZE:-0}" == "1" ]]; then
    cmd+=(--web-visualize --web-port "${WEB_PORT:-8080}")
fi

"${cmd[@]}" "$@"
