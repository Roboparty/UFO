#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/onboard/run_preflight_suite.sh [options]

No-actuation onboard deployment checks. This script does not start robot
control, teleop bridge, or realtime z server.

Options:
  --require-body        Fail if XRoboToolkit body data is unavailable.
  --check-z-stream      Also validate an already-running realtime z stream.
  --check-teleop-qpos   Also validate an already-running teleop qpos bridge.
  --duration SECONDS    XR/ZMQ polling duration for data checks (default: 3).
  -h, --help            Show this help.

Environment:
  UFO_ROOT              Repository root (default: auto-detected).
  ONBOARD_PY           Python executable (default: /home/unitree/ufo_deploy_venv/bin/python).
  G1_INTERFACE          Explicit low-level DDS NIC used by real launch checks.
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
REQUIRE_BODY=0
CHECK_Z_STREAM=0
CHECK_TELEOP_QPOS=0
DURATION=3

while [[ $# -gt 0 ]]; do
  case "$1" in
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
log "git_head: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
log "disk:"
df -h . || true

"${ONBOARD_PY}" scripts/onboard/check_g1_onboard_env.py
"${ONBOARD_PY}" scripts/onboard/check_deploy_artifacts.py
"${ONBOARD_PY}" scripts/onboard/check_policy_preflight.py --task config/exp/tracking/tracking.yaml
"${ONBOARD_PY}" scripts/onboard/check_policy_preflight.py --task config/exp/tracking/teleop.yaml

xrobot_args=(--duration "${DURATION}")
if [[ "${REQUIRE_BODY}" == "1" ]]; then
  xrobot_args+=(--require-body)
fi
"${ONBOARD_PY}" scripts/onboard/check_xrobot_sdk.py "${xrobot_args[@]}"

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
