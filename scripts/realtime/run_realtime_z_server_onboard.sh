#!/usr/bin/env bash
set -euo pipefail

UFO_ROOT="${UFO_ROOT:-/home/unitree/UFO-Deploy}"
Z_PY="${Z_PY:-python}"
MODEL_DIR="${MODEL_DIR:-${UFO_ROOT}/model/g1_policy}"
BACKWARD_ONNX="${BACKWARD_ONNX:-${MODEL_DIR}/exported/backward_encoder.onnx}"
MUJOCO_XML="${MUJOCO_XML:-${UFO_ROOT}/data/robots/g1/scene_29dof_freebase.xml}"
DEVICE="${DEVICE:-cpu}"
ENABLE_PICO_Z_CONTROL="${ENABLE_PICO_Z_CONTROL:-1}"

if ! command -v "${Z_PY}" >/dev/null 2>&1 && [[ ! -x "${Z_PY}" ]]; then
  echo "[run_realtime_z_server_onboard] python not found: ${Z_PY}" >&2
  exit 1
fi
if [[ ! -f "${BACKWARD_ONNX}" ]]; then
  echo "[run_realtime_z_server_onboard] missing backward ONNX: ${BACKWARD_ONNX}" >&2
  exit 1
fi
if [[ ! -f "${MUJOCO_XML}" ]]; then
  echo "[run_realtime_z_server_onboard] missing MuJoCo XML: ${MUJOCO_XML}" >&2
  exit 1
fi

cd "${UFO_ROOT}"

cmd=(
  "${Z_PY}" scripts/realtime/realtime_z_server.py
  --teleop_req tcp://127.0.0.1:28701
  --teleop_rep tcp://127.0.0.1:28702
  --teleop_ctrl tcp://127.0.0.1:28703
  --z_bind tcp://*:28711
  --hz 50
  --backward_onnx "${BACKWARD_ONNX}"
  --mujoco_xml "${MUJOCO_XML}"
  --device "${DEVICE}"
  --root_height_obs
  --wall-clock-dt
  --fix-quat-continuity
  --angvel-delta-frame world
  --enable-pose-buffer
  --pose-buffer-lookback-ms "${POSE_BUFFER_LOOKBACK_MS:-40}"
  --pose-buffer-window-ms "${POSE_BUFFER_WINDOW_MS:-500}"
  --max-retarget-age-ms "${MAX_RETARGET_AGE_MS:-200}"
  --max-z-delta "${MAX_Z_DELTA:-0.75}"
)

if [[ "${ENABLE_PICO_Z_CONTROL}" == "1" ]]; then
  cmd+=(--enable-pico-control)
fi

exec "${cmd[@]}" "$@"
