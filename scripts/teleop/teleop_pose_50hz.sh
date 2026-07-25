#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

actual_human_height="${ACTUAL_HUMAN_HEIGHT:-1.75}"
lookback_ms="${LOOKBACK_MS:-50}"
log_interval_s="${LOG_INTERVAL_S:-1}"

cmd=(
    python xrobot_teleop_to_pose_zmq_server.py
    --robot unitree_g1
    --actual_human_height "${actual_human_height}"
    --ctrl_fps 50
    --lookback_ms "${lookback_ms}"
    --retarget_buffer_window_s 0.5
    --log_interval_s "${log_interval_s}"
    --req_bind_addr tcp://*:28701
    --rep_bind_addr tcp://*:28702
    --ctrl_bind_addr tcp://*:28703
    --vis_fps 5
)

if [[ "${WEB_VISUALIZE:-0}" == "1" ]]; then
    cmd+=(--web-visualize --web-port "${WEB_PORT:-8080}")
fi

"${cmd[@]}" "$@"
