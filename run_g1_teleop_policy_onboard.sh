#!/usr/bin/env bash
set -euo pipefail

if [[ "${UFO_REAL_ROBOT_OK:-}" != "1" ]]; then
  echo "Refusing to start real robot control without UFO_REAL_ROBOT_OK=1." >&2
  exit 2
fi

UFO_ROOT="${UFO_ROOT:-/home/unitree/UFO-Deploy}"
VENV_PATH="${VENV_PATH:-}"
CYCLONEDDS_HOME="${CYCLONEDDS_HOME:-/home/unitree/cyclonedds_ws/install/cyclonedds}"
UNITREE_SDK_LIB="${UNITREE_SDK_LIB:-/home/unitree/unitree_sdk2_bfm/build/lib}"
UNITREE_SDK_THIRDPARTY_LIB="${UNITREE_SDK_THIRDPARTY_LIB:-/home/unitree/unitree_sdk2_bfm/thirdparty/lib/aarch64}"
CUDA_LIB="${CUDA_LIB:-/usr/local/cuda-11.4/lib64}"

MODEL_DIR="${MODEL_DIR:-${UFO_ROOT}/model/g1_policy}"
POLICY_CONFIG="${POLICY_CONFIG:-${UFO_ROOT}/config/policy/g1_policy.yaml}"
TASK_CONFIG="${TASK_CONFIG:-${UFO_ROOT}/config/exp/tracking/teleop.yaml}"
ROBOT_CONFIG="${ROBOT_CONFIG:-${UFO_ROOT}/config/robot/g1_real.yaml}"
MODEL_PATH="${MODEL_PATH:-${MODEL_DIR}/exported/FBcprAuxModel.onnx}"
PICO_CONTROL_ADDR="${PICO_CONTROL_ADDR:-tcp://127.0.0.1:28704}"
ENABLE_PICO_POLICY_CONTROL="${ENABLE_PICO_POLICY_CONTROL:-0}"
CTRL_PUB_BIND_ADDR="${CTRL_PUB_BIND_ADDR:-}"

cd "${UFO_ROOT}"

if [[ "${ENABLE_PICO_POLICY_CONTROL}" == "1" && -z "${CTRL_PUB_BIND_ADDR}" ]]; then
  echo "[run_g1_teleop_policy_onboard] ENABLE_PICO_POLICY_CONTROL=1 requires CTRL_PUB_BIND_ADDR=tcp://*:28704 in this shell as an explicit legacy/debug acknowledgement." >&2
  exit 1
fi

if [[ -n "${VENV_PATH}" ]]; then
  if [[ ! -f "${VENV_PATH}" ]]; then
    echo "[run_g1_teleop_policy_onboard] venv not found: ${VENV_PATH}" >&2
    exit 1
  fi
  source "${VENV_PATH}"
fi

for path in "${MODEL_PATH}" "${POLICY_CONFIG}" "${TASK_CONFIG}" "${ROBOT_CONFIG}"; do
  if [[ ! -f "${path}" ]]; then
    echo "[run_g1_teleop_policy_onboard] missing file: ${path}" >&2
    exit 1
  fi
done

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

echo "[run_g1_teleop_policy_onboard] repo: ${UFO_ROOT}"
echo "[run_g1_teleop_policy_onboard] python: $(command -v python)"
echo "[run_g1_teleop_policy_onboard] model: ${MODEL_PATH}"
echo "[run_g1_teleop_policy_onboard] policy: ${POLICY_CONFIG}"
echo "[run_g1_teleop_policy_onboard] task: ${TASK_CONFIG}"
pico_policy_control_state="disabled"
if [[ "${ENABLE_PICO_POLICY_CONTROL}" == "1" ]]; then
  pico_policy_control_state="enabled legacy/debug compatibility mode"
fi
echo "[run_g1_teleop_policy_onboard] pico_policy_control: ${pico_policy_control_state}"
if [[ "${ENABLE_PICO_POLICY_CONTROL}" == "1" ]]; then
  echo "[run_g1_teleop_policy_onboard] WARNING: legacy/debug compatibility mode; PICO policy buttons can control policy state via ${PICO_CONTROL_ADDR}"
  echo "[run_g1_teleop_policy_onboard] expected teleop CTRL_PUB_BIND_ADDR: ${CTRL_PUB_BIND_ADDR}"
fi

cmd=(
  python rl_policy/ufo_policy.py
  --robot_config "${ROBOT_CONFIG}"
  --policy_config "${POLICY_CONFIG}"
  --model_path "${MODEL_PATH}"
  --task "${TASK_CONFIG}"
)

if [[ "${ENABLE_PICO_POLICY_CONTROL}" == "1" ]]; then
  cmd+=(--pico-control --pico-control-addr "${PICO_CONTROL_ADDR}")
fi

exec "${cmd[@]}" "$@"
