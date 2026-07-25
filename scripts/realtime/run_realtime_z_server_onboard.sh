#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[run_realtime_z_server_onboard] $*"
}

fail() {
  echo "[run_realtime_z_server_onboard] ERROR: $*" >&2
  exit 1
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_UFO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

UFO_ROOT="${UFO_ROOT:-${DEFAULT_UFO_ROOT}}"
Z_PY="${Z_PY:-}"
MODEL_DIR="${MODEL_DIR:-${UFO_ROOT}/model/g1_policy}"
BACKWARD_ONNX="${BACKWARD_ONNX:-${MODEL_DIR}/exported/backward_encoder.onnx}"
MUJOCO_XML="${MUJOCO_XML:-${UFO_ROOT}/data/robots/g1/scene_29dof_freebase.xml}"
DEVICE="${DEVICE:-cpu}"
ENABLE_PICO_Z_CONTROL="${ENABLE_PICO_Z_CONTROL:-1}"

[[ -d "${UFO_ROOT}" ]] || fail "UFO_ROOT does not exist: ${UFO_ROOT}"
[[ -f "${UFO_ROOT}/scripts/realtime/realtime_z_server.py" ]] || \
  fail "missing realtime z server under UFO_ROOT: ${UFO_ROOT}"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${UFO_ROOT}:${PYTHONPATH}"
else
  export PYTHONPATH="${UFO_ROOT}"
fi

python_has_realtime_deps() {
  local python_bin="$1"

  if [[ ! -x "${python_bin}" ]] && ! command -v "${python_bin}" >/dev/null 2>&1; then
    return 1
  fi

  "${python_bin}" - <<'PY' >/dev/null 2>&1
import mujoco
import numpy
import onnxruntime
import zmq
PY
}

if [[ -z "${Z_PY}" ]]; then
  for candidate in \
    "/home/unitree/ufo_deploy_venv/bin/python" \
    "${UFO_ROOT}/.venv/bin/python" \
    "${UFO_ROOT}/venv/bin/python" \
    "/home/unitree/miniconda3/envs/ufo-deploy/bin/python"; do
    if [[ ! -x "${candidate}" ]]; then
      continue
    fi
    if python_has_realtime_deps "${candidate}"; then
      Z_PY="${candidate}"
      break
    fi
    log "skipping Python without realtime z deps: ${candidate}"
  done
fi

if [[ -z "${Z_PY}" ]]; then
  fail "no valid deployment Python found. Create /home/unitree/ufo_deploy_venv, ${UFO_ROOT}/.venv, or set Z_PY=/path/to/python."
fi
if ! command -v "${Z_PY}" >/dev/null 2>&1 && [[ ! -x "${Z_PY}" ]]; then
  fail "python not found or not executable: ${Z_PY}"
fi
if ! python_has_realtime_deps "${Z_PY}"; then
  "${Z_PY}" - <<'PY' || true
import mujoco
import numpy
import onnxruntime
import zmq
print("realtime z deps ok")
PY
  fail "selected Python failed realtime z dependency probe: ${Z_PY}"
fi
for arg in "$@"; do
  if [[ "${arg}" == "-h" || "${arg}" == "--help" ]]; then
    cd "${UFO_ROOT}"
    exec "${Z_PY}" scripts/realtime/realtime_z_server.py "$@"
  fi
done
if [[ ! -f "${BACKWARD_ONNX}" ]]; then
  fail "missing backward ONNX: ${BACKWARD_ONNX}"
fi
if [[ ! -f "${MUJOCO_XML}" ]]; then
  fail "missing MuJoCo XML: ${MUJOCO_XML}"
fi

cd "${UFO_ROOT}"

log "repo: ${UFO_ROOT}"
log "python: ${Z_PY}"

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
  --pose-buffer-lookback-ms "${POSE_BUFFER_LOOKBACK_MS:-0}"
  --pose-buffer-window-ms "${POSE_BUFFER_WINDOW_MS:-500}"
  --max-retarget-age-ms "${MAX_RETARGET_AGE_MS:-200}"
  --max-z-delta "${MAX_Z_DELTA:-0.75}"
)

if [[ "${ENABLE_PICO_Z_CONTROL}" == "1" ]]; then
  cmd+=(--enable-pico-control)
fi

exec "${cmd[@]}" "$@"
