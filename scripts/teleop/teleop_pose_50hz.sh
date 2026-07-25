#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

cmd=(
    python xrobot_teleop_to_pose_zmq_server.py
    --robot unitree_g1
    --req_bind_addr tcp://*:28701
    --rep_bind_addr tcp://*:28702
    --ctrl_bind_addr tcp://*:28703
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
