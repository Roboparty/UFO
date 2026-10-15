#!/usr/bin/env bash
set -euo pipefail

if [[ "${UFO_REAL_ROBOT_OK:-}" != "1" ]]; then
  echo "Refusing to start real robot control without UFO_REAL_ROBOT_OK=1." >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UFO_ROOT="${UFO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
VENV_PATH="${VENV_PATH:-}"
CYCLONEDDS_HOME="${CYCLONEDDS_HOME:-/home/unitree/cyclonedds_ws/install/cyclonedds}"
UNITREE_SDK_LIB="${UNITREE_SDK_LIB:-/home/unitree/unitree_sdk2_bfm/build/lib}"
UNITREE_SDK_THIRDPARTY_LIB="${UNITREE_SDK_THIRDPARTY_LIB:-/home/unitree/unitree_sdk2_bfm/thirdparty/lib/aarch64}"
OPENSSL11_LIB="${OPENSSL11_LIB:-${UFO_ROOT}/external/openssl-1.1-aarch64}"
CUDA_LIB="${CUDA_LIB:-/usr/local/cuda-11.4/lib64}"
G1_INTERFACE="${G1_INTERFACE:-}"
ALLOW_DEFAULT_ROUTE_INTERFACE="${ALLOW_DEFAULT_ROUTE_INTERFACE:-0}"

MODEL_DIR="${MODEL_DIR:-${UFO_ROOT}/model/g1_policy}"
TASK="${TASK:-tracking}"
TASK_CONFIG="${TASK_CONFIG:-}"
ROBOT_CONFIG="${ROBOT_CONFIG:-${UFO_ROOT}/config/robot/g1_real.yaml}"
MODEL_PATH="${MODEL_PATH:-${MODEL_DIR}/exported/FBcprAuxModel.onnx}"
POLICY_CONFIG="${POLICY_CONFIG:-}"
PICO_CONTROL_ADDR="${PICO_CONTROL_ADDR:-tcp://127.0.0.1:28704}"
ENABLE_PICO_POLICY_CONTROL="${ENABLE_PICO_POLICY_CONTROL:-0}"
CTRL_PUB_BIND_ADDR="${CTRL_PUB_BIND_ADDR:-}"

cd "${UFO_ROOT}"

if [[ "${ENABLE_PICO_POLICY_CONTROL}" == "1" && -z "${CTRL_PUB_BIND_ADDR}" ]]; then
  echo "[run_policy] ENABLE_PICO_POLICY_CONTROL=1 requires CTRL_PUB_BIND_ADDR=tcp://127.0.0.1:28704 in this shell as an explicit legacy/debug acknowledgement." >&2
  exit 1
fi

if [[ -n "${VENV_PATH}" ]]; then
  if [[ ! -f "${VENV_PATH}" ]]; then
    echo "[run_policy] venv not found: ${VENV_PATH}" >&2
    exit 1
  fi
  source "${VENV_PATH}"
fi

ld_parts=()
[[ -d "${CYCLONEDDS_HOME}/lib" ]] && ld_parts+=("${CYCLONEDDS_HOME}/lib")
[[ -d "${UNITREE_SDK_LIB}" ]] && ld_parts+=("${UNITREE_SDK_LIB}")
[[ -d "${UNITREE_SDK_THIRDPARTY_LIB}" ]] && ld_parts+=("${UNITREE_SDK_THIRDPARTY_LIB}")
[[ -d "${CUDA_LIB}" ]] && ld_parts+=("${CUDA_LIB}")
if (( ${#ld_parts[@]} > 0 )); then
  export LD_LIBRARY_PATH="$(IFS=:; echo "${ld_parts[*]}"):${LD_LIBRARY_PATH:-}"
fi
export CYCLONEDDS_HOME
export PYTHONPATH="${UNITREE_SDK_LIB}:${PYTHONPATH:-}"

try_g1_import() {
  python - <<'PY'
import g1_interface
print("g1_interface import ok", getattr(g1_interface, "G1_NUM_MOTOR", None))
PY
}

if ! g1_import_output="$(try_g1_import 2>&1)"; then
  echo "${g1_import_output}" >&2
  if [[ "${g1_import_output}" == *"libssl.so.1.1"* || "${g1_import_output}" == *"libcrypto.so.1.1"* ]]; then
    if [[ ! -d "${OPENSSL11_LIB}" ]]; then
      echo "[run_policy] missing OPENSSL11_LIB=${OPENSSL11_LIB}; set OPENSSL11_LIB to a directory containing libssl.so.1.1 and libcrypto.so.1.1" >&2
      exit 1
    fi
    [[ -f "${OPENSSL11_LIB}/libssl.so.1.1" ]] || { echo "[run_policy] ${OPENSSL11_LIB}/libssl.so.1.1 missing" >&2; exit 1; }
    [[ -f "${OPENSSL11_LIB}/libcrypto.so.1.1" ]] || { echo "[run_policy] ${OPENSSL11_LIB}/libcrypto.so.1.1 missing" >&2; exit 1; }
    export LD_LIBRARY_PATH="${OPENSSL11_LIB}:${LD_LIBRARY_PATH:-}"
    echo "[run_policy] retrying g1_interface import with OPENSSL11_LIB=${OPENSSL11_LIB}"
    if ! g1_import_output="$(try_g1_import 2>&1)"; then
      echo "${g1_import_output}" >&2
      echo "[run_policy] g1_interface still failed after OPENSSL11_LIB. ldd:" >&2
      ldd "${UNITREE_SDK_LIB}/g1_interface.cpython-310-aarch64-linux-gnu.so" >&2 || true
      exit 1
    fi
  else
    echo "[run_policy] g1_interface import failed. ldd:" >&2
    ldd "${UNITREE_SDK_LIB}/g1_interface.cpython-310-aarch64-linux-gnu.so" >&2 || true
    exit 1
  fi
fi
echo "[run_policy] ${g1_import_output}"

echo "[run_policy] repo: ${UFO_ROOT}"
echo "[run_policy] python: $(command -v python)"
pico_policy_control_state="disabled"
if [[ "${ENABLE_PICO_POLICY_CONTROL}" == "1" ]]; then
  pico_policy_control_state="enabled legacy/debug compatibility mode"
fi
echo "[run_policy] pico_policy_control: ${pico_policy_control_state}"
if [[ "${ENABLE_PICO_POLICY_CONTROL}" == "1" ]]; then
  echo "[run_policy] WARNING: legacy/debug compatibility mode; PICO policy buttons can control policy state via ${PICO_CONTROL_ADDR}"
  echo "[run_policy] expected teleop CTRL_PUB_BIND_ADDR: ${CTRL_PUB_BIND_ADDR}"
fi

cmd=(
  python deploy.py --mode sim2real
  --robot-config "${ROBOT_CONFIG}"
  --model "${MODEL_PATH}"
  --task "${TASK}"
)

[[ -n "${POLICY_CONFIG}" ]] && cmd+=(--policy-config "${POLICY_CONFIG}")
[[ -n "${TASK_CONFIG}" ]] && cmd+=(--task-config "${TASK_CONFIG}")
[[ -n "${G1_INTERFACE}" ]] && cmd+=(--interface "${G1_INTERFACE}")
[[ "${ALLOW_DEFAULT_ROUTE_INTERFACE}" == "1" ]] && cmd+=(--allow-default-route-interface)

if [[ "${ENABLE_PICO_POLICY_CONTROL}" == "1" ]]; then
  cmd+=(--pico-control --pico-control-addr "${PICO_CONTROL_ADDR}")
fi

exec "${cmd[@]}" "$@"
