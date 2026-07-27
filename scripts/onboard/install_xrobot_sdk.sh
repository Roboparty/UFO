#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/onboard/install_xrobot_sdk.sh --sdk-root PATH --venv PATH [options]

Install the ARM64 CPython 3.10 XRoboToolkit binding into a stable UFO external
directory and make the selected venv import it through an idempotent .pth file.

Options:
  --sdk-root PATH   XRoboToolkit-PC-Service-Pybind_X86_and_ARM64 root.
  --venv PATH       Python venv path or venv python executable.
  --ufo-root PATH   UFO repo root (default: auto-detected).
  --copy            Copy .so files instead of creating symlinks.
  -h, --help        Show this help.

This script rejects x86_64 binaries and exits if ldd reports missing libraries.
It does not commit SDK binaries or machine symlinks to Git.
EOF
}

fail() {
  echo "[install_xrobot_sdk] ERROR: $*" >&2
  exit 1
}

log() {
  echo "[install_xrobot_sdk] $*"
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UFO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SDK_ROOT=""
VENV=""
COPY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sdk-root)
      [[ $# -ge 2 ]] || fail "--sdk-root requires a value"
      SDK_ROOT="$2"
      shift 2
      ;;
    --venv)
      [[ $# -ge 2 ]] || fail "--venv requires a value"
      VENV="$2"
      shift 2
      ;;
    --ufo-root)
      [[ $# -ge 2 ]] || fail "--ufo-root requires a value"
      UFO_ROOT="$2"
      shift 2
      ;;
    --copy)
      COPY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "unknown argument: $1"
      ;;
  esac
done

[[ -n "${SDK_ROOT}" ]] || fail "--sdk-root is required"
[[ -n "${VENV}" ]] || fail "--venv is required"
[[ -d "${SDK_ROOT}" ]] || fail "SDK root does not exist: ${SDK_ROOT}"
[[ -d "${UFO_ROOT}" ]] || fail "UFO root does not exist: ${UFO_ROOT}"

if [[ -x "${VENV}/bin/python" ]]; then
  PYTHON="${VENV}/bin/python"
elif [[ -x "${VENV}" ]]; then
  PYTHON="${VENV}"
else
  fail "venv python not found from --venv ${VENV}"
fi

PY_EXT="${SDK_ROOT}/xrobotoolkit_sdk.cpython-310-aarch64-linux-gnu.so"
NATIVE_LIB="${SDK_ROOT}/lib/aarch64/libPXREARobotSDK.so"
[[ -f "${PY_EXT}" ]] || fail "missing Python extension: ${PY_EXT}"
[[ -f "${NATIVE_LIB}" ]] || fail "missing native library: ${NATIVE_LIB}"

check_file_arch() {
  local path="$1"
  local output
  output="$(file "${path}")"
  log "${output}"
  [[ "${output}" == *"ELF 64-bit"* ]] || fail "${path} is not an ELF 64-bit binary"
  [[ "${output}" == *"ARM aarch64"* ]] || fail "${path} is not ARM aarch64"
  [[ "${output}" != *"x86-64"* ]] || fail "${path} is x86_64, refusing"
}

check_ldd() {
  local path="$1"
  local output
  output="$(ldd "${path}")" || fail "ldd failed for ${path}"
  echo "${output}"
  if echo "${output}" | grep -q "not found"; then
    fail "ldd reports missing libraries for ${path}"
  fi
}

check_file_arch "${PY_EXT}"
check_file_arch "${NATIVE_LIB}"
check_ldd "${PY_EXT}"
check_ldd "${NATIVE_LIB}"

DEST="${UFO_ROOT}/external/XRoboToolkit-PC-Service-Pybind"
mkdir -p "${DEST}/lib/aarch64"

install_one() {
  local src="$1"
  local dst="$2"
  if [[ "${COPY}" == "1" ]]; then
    cp -f "${src}" "${dst}"
    log "copied ${src} -> ${dst}"
  else
    ln -sfn "${src}" "${dst}"
    log "symlinked ${dst} -> ${src}"
  fi
}

install_one "${PY_EXT}" "${DEST}/xrobotoolkit_sdk.cpython-310-aarch64-linux-gnu.so"
install_one "${NATIVE_LIB}" "${DEST}/lib/aarch64/libPXREARobotSDK.so"

PTH_PATH="$("${PYTHON}" - <<'PY'
import site
from pathlib import Path
paths = site.getsitepackages()
if not paths:
    raise SystemExit("site.getsitepackages() returned no paths")
print(Path(paths[0]) / "ufo_xrobotoolkit_sdk.pth")
PY
)"

"${PYTHON}" - "${PTH_PATH}" "${DEST}" <<'PY'
from pathlib import Path
import sys

pth = Path(sys.argv[1])
line = str(Path(sys.argv[2]).resolve(strict=False))
pth.parent.mkdir(parents=True, exist_ok=True)
existing = []
if pth.exists():
    existing = pth.read_text(encoding="utf-8").splitlines()
if line not in existing:
    existing.append(line)
pth.write_text("\n".join(existing).rstrip() + "\n", encoding="utf-8")
print(pth)
PY

LD_LIBRARY_PATH="${DEST}/lib/aarch64:${LD_LIBRARY_PATH:-}" "${PYTHON}" - <<'PY'
import xrobotoolkit_sdk as xrt
print("module:", getattr(xrt, "__file__", None))
print("has init:", hasattr(xrt, "init"))
print("polling api:", all(hasattr(xrt, name) for name in [
    "is_body_data_available",
    "get_body_joints_pose",
    "get_body_timestamp_ns",
]))
print("callback api:", all(hasattr(xrt, name) for name in [
    "register_frame_callback",
    "clear_frame_callback",
    "has_frame_callback",
]))
PY

log "XRoboToolkit SDK install complete"
