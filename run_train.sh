#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${UFO_CACHE_DIR:-}" ]]; then
  work_dir=""
  args=("$@")
  for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in
      --work-dir)
        if ((i + 1 < ${#args[@]})); then
          work_dir="${args[$((i + 1))]}"
        fi
        ;;
      --work-dir=*)
        work_dir="${args[$i]#--work-dir=}"
        ;;
    esac
  done

  if [[ -n "$work_dir" ]]; then
    work_parent="$(dirname "$work_dir")"
    if [[ "$(basename "$work_parent")" == "runs" ]]; then
      export UFO_CACHE_DIR="$(dirname "$work_parent")/cache"
    else
      export UFO_CACHE_DIR="$work_parent/cache"
    fi
  else
    export UFO_CACHE_DIR="$script_dir/cache"
  fi
fi

mkdir -p \
  "$UFO_CACHE_DIR/uv" \
  "$UFO_CACHE_DIR/pycache" \
  "$UFO_CACHE_DIR/tmp" \
  "$UFO_CACHE_DIR/torchinductor" \
  "$UFO_CACHE_DIR/triton" \
  "$UFO_CACHE_DIR/cuda" \
  "$UFO_CACHE_DIR/warp"

export BFMZERO_MJLAB_CACHE_DIR="${BFMZERO_MJLAB_CACHE_DIR:-$UFO_CACHE_DIR}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$UFO_CACHE_DIR/uv}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$UFO_CACHE_DIR/pycache}"
export TMPDIR="${TMPDIR:-$UFO_CACHE_DIR/tmp}"
export TEMP="${TEMP:-$TMPDIR}"
export TMP="${TMP:-$TMPDIR}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$UFO_CACHE_DIR/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$UFO_CACHE_DIR/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$UFO_CACHE_DIR/cuda}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$UFO_CACHE_DIR/warp}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

exec uv run python -m humanoidverse.train "$@"
