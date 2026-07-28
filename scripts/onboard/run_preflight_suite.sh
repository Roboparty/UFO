#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/onboard/run_preflight_suite.sh [options]

No-actuation onboard deployment checks. This script does not start robot
control, teleop bridge, or realtime z server.

The validated G1 onboard deployment uses a CPython 3.10 venv. Conda is the
validated workstation default, not the onboard preflight default.

Options:
  --profile PROFILE    Dependency profile: ordinary, teleop, diagnostic, all (default: ordinary).
  --require-body        Fail if XRoboToolkit body data is unavailable.
  --check-z-stream      Also validate an already-running realtime z stream.
  --check-teleop-qpos   Also validate an already-running teleop qpos bridge.
  --duration SECONDS    XR/ZMQ polling duration for data checks (default: 3).
  -h, --help            Show this help.

Environment:
  UFO_ROOT              Repository root (default: auto-detected).
  ONBOARD_PY           Python executable (default: /home/unitree/ufo_deploy_venv/bin/python).
  ONBOARD_ALLOW_NONDEFAULT_PY
                       Set to 1 only for debugging non-default onboard Python environments.
  G1_INTERFACE          Explicit low-level DDS NIC used by real launch checks.
  UNITREE_SDK_PYTHON    Optional unitree_sdk2_python checkout for diagnostic profile.
EOF
}

log() {
  echo "[run_preflight_suite] $*"
}

fail() {
  echo "[run_preflight_suite] ERROR: $*" >&2
  exit 1
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UFO_ROOT="${UFO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
ONBOARD_PY="${ONBOARD_PY:-/home/unitree/ufo_deploy_venv/bin/python}"
ONBOARD_ALLOW_NONDEFAULT_PY="${ONBOARD_ALLOW_NONDEFAULT_PY:-0}"
PROFILE="ordinary"
REQUIRE_BODY=0
CHECK_Z_STREAM=0
CHECK_TELEOP_QPOS=0
DURATION=3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      [[ $# -ge 2 ]] || fail "--profile requires a value"
      PROFILE="$2"
      case "${PROFILE}" in
        ordinary|teleop|diagnostic|all) ;;
        *) fail "--profile must be one of: ordinary, teleop, diagnostic, all" ;;
      esac
      shift 2
      ;;
    --require-body)
      REQUIRE_BODY=1
      shift
      ;;
    --check-z-stream)
      CHECK_Z_STREAM=1
      shift
      ;;
    --check-teleop-qpos)
      CHECK_TELEOP_QPOS=1
      shift
      ;;
    --duration)
      [[ $# -ge 2 ]] || fail "--duration requires a value"
      DURATION="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "unknown option: $1"
      ;;
  esac
done

[[ -d "${UFO_ROOT}" ]] || fail "UFO_ROOT does not exist: ${UFO_ROOT}"
[[ -x "${ONBOARD_PY}" ]] || fail "ONBOARD_PY is not executable: ${ONBOARD_PY}"

cd "${UFO_ROOT}"
log "NO-ACTUATION: this suite does not start robot control, teleop bridge, or realtime z server"
log "repo: ${UFO_ROOT}"
log "python: ${ONBOARD_PY}"
log "profile: ${PROFILE}"
log "onboard_python_policy: CPython 3.10 venv"
log "git_head: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
log "disk:"
df -h . || true

run_control_checks=0
run_teleop_checks=0
run_diagnostic_checks=0

case "${PROFILE}" in
  ordinary)
    run_control_checks=1
    ;;
  teleop)
    run_control_checks=1
    run_teleop_checks=1
    CHECK_TELEOP_QPOS=1
    ;;
  diagnostic)
    run_diagnostic_checks=1
    ;;
  all)
    run_control_checks=1
    run_teleop_checks=1
    run_diagnostic_checks=1
    CHECK_TELEOP_QPOS=1
    ;;
esac

env_check_args=(--profile "${PROFILE}" --target g1-onboard)
if [[ "${ONBOARD_ALLOW_NONDEFAULT_PY}" == "1" ]]; then
  log "WARNING: ONBOARD_ALLOW_NONDEFAULT_PY=1; using a non-default onboard Python environment"
  env_check_args+=(--allow-nondefault-python-env)
fi

"${ONBOARD_PY}" scripts/onboard/check_g1_onboard_env.py "${env_check_args[@]}"

if [[ "${run_control_checks}" == "1" ]]; then
  "${ONBOARD_PY}" scripts/onboard/check_deploy_artifacts.py
  "${ONBOARD_PY}" scripts/onboard/check_policy_preflight.py --task config/exp/tracking/tracking.yaml
fi

if [[ "${run_teleop_checks}" == "1" ]]; then
  "${ONBOARD_PY}" scripts/onboard/check_policy_preflight.py --task config/exp/tracking/teleop.yaml

  xrobot_args=(--duration "${DURATION}")
  if [[ "${REQUIRE_BODY}" == "1" ]]; then
    xrobot_args+=(--require-body)
  fi
  "${ONBOARD_PY}" scripts/onboard/check_xrobot_sdk.py "${xrobot_args[@]}"
fi

if [[ "${run_control_checks}" == "1" ]]; then
  log "checking real policy launcher safety gates"
  if UFO_REAL_ROBOT_OK=0 VENV_PATH=/dev/null ./run_g1_teleop_policy_onboard.sh >/tmp/ufo-launcher-refusal.log 2>&1; then
    fail "launcher did not refuse without UFO_REAL_ROBOT_OK=1"
  else
    rc=$?
    if [[ "${rc}" == "2" ]]; then
      log "launcher refuses without UFO_REAL_ROBOT_OK=1"
    else
      cat /tmp/ufo-launcher-refusal.log || true
      fail "launcher refusal returned ${rc}, expected 2"
    fi
  fi

  if [[ -n "${G1_INTERFACE:-}" ]]; then
    launcher_venv="${VENV_PATH:-${ONBOARD_PY%/bin/python}/bin/activate}"
    [[ -f "${launcher_venv}" ]] || fail "launcher venv activate not found: ${launcher_venv}"
    UFO_REAL_ROBOT_OK=1 \
    VENV_PATH="${launcher_venv}" \
    G1_INTERFACE="${G1_INTERFACE}" \
    ./run_g1_teleop_policy_onboard.sh --help >/tmp/ufo-launcher-help.log 2>&1 || {
      cat /tmp/ufo-launcher-help.log || true
      fail "launcher --help preflight failed"
    }
    log "launcher --help preflight passed"
  else
    log "G1_INTERFACE not set; skipping launcher --help interface/OpenSSL gate"
  fi
fi

if [[ "${run_diagnostic_checks}" == "1" ]]; then
  lowstate_args=(--duration "${DURATION}")
  if [[ -n "${G1_INTERFACE:-}" ]]; then
    lowstate_args+=(--interface "${G1_INTERFACE}")
  fi
  "${ONBOARD_PY}" scripts/onboard/check_g1_state_readonly.py "${lowstate_args[@]}"
fi

if [[ "${CHECK_Z_STREAM}" == "1" ]]; then
  "${ONBOARD_PY}" scripts/onboard/check_z_stream.py \
    --addr tcp://127.0.0.1:28711 \
    --duration "${DURATION}" \
    --min-count 1
fi

if [[ "${CHECK_TELEOP_QPOS}" == "1" ]]; then
  "${ONBOARD_PY}" scripts/onboard/check_teleop_qpos.py \
    --duration "${DURATION}" \
    --min-valid 1
fi

log "summary: preflight suite completed"
