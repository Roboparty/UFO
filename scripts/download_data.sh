#!/usr/bin/env bash
set -euo pipefail

DATASET_REPO="${UFO_DATASET_REPO:-xuewang/UFO-MotionData}"
DATASET_REVISION="${UFO_DATASET_REVISION:-main}"
DATASET_FILES=(
  "g1/lafan_29dof_10s-clipped.pkl"
  "g1/lafan_29dof.pkl"
)
DESTS=(
  "humanoidverse/data/lafan_29dof_10s-clipped.pkl"
  "humanoidverse/data/lafan_29dof.pkl"
)
EXPECTED_SHA256S=(
  "7f5aa36957808ee2e972472b18add8510533742710ba312d8b8c6e6014f1c010"
  "f3a0c2810363f5c50bf4146fa2db33c1ff5b90d00cb7c0bc2aa4622696375e11"
)

usage() {
  cat <<USAGE
Usage: bash scripts/download_data.sh [g1_lafan]

Downloads the default G1 LaFAN training motion data from Hugging Face:
  https://huggingface.co/datasets/${DATASET_REPO}
  ${DATASET_FILES[0]}
  ${DATASET_FILES[1]}

Environment overrides:
  UFO_DATASET_REPO=${DATASET_REPO}
  UFO_DATASET_REVISION=${DATASET_REVISION}
USAGE
}

case "${1:-g1_lafan}" in
  g1_lafan|lafan|g1)
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown dataset: $1" >&2
    usage >&2
    exit 2
    ;;
esac

needed_indices=()
for i in "${!DATASET_FILES[@]}"; do
  dest="${DESTS[$i]}"
  expected="${EXPECTED_SHA256S[$i]}"

  if [[ -f "${dest}" ]]; then
    actual="$(sha256sum "${dest}" | awk '{print $1}')"
    if [[ "${actual}" == "${expected}" ]]; then
      echo "Data already present: ${dest}"
      continue
    fi
    echo "Existing ${dest} has unexpected sha256: ${actual}" >&2
    echo "Re-downloading." >&2
  fi
  needed_indices+=("${i}")
done

if [[ "${#needed_indices[@]}" -eq 0 ]]; then
  exit 0
fi

tmpdir="$(mktemp -d)"
cleanup() {
  rm -rf "${tmpdir}"
}
trap cleanup EXIT

download_one() {
  local dataset_file="$1"

  if command -v hf >/dev/null 2>&1; then
    hf download "${DATASET_REPO}" \
      "${dataset_file}" \
      --repo-type dataset \
      --revision "${DATASET_REVISION}" \
      --local-dir "${tmpdir}"
    return
  fi

  if python3 - <<'PYTEST' >/dev/null 2>&1
import huggingface_hub
PYTEST
  then
    python3 - "${DATASET_REPO}" "${dataset_file}" "${DATASET_REVISION}" "${tmpdir}" <<'PYDL'
import sys
from huggingface_hub import hf_hub_download

repo_id, filename, revision, local_dir = sys.argv[1:]
path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    repo_type="dataset",
    revision=revision,
    local_dir=local_dir,
)
print(path)
PYDL
    return
  fi

  cat >&2 <<'EOF'
Missing Hugging Face downloader.
Install one of the following and rerun:
  python -m pip install -U huggingface_hub
or
  uv tool install huggingface_hub
EOF
  exit 1
}

for i in "${needed_indices[@]}"; do
  download_one "${DATASET_FILES[$i]}"
done

for i in "${needed_indices[@]}"; do
  dataset_file="${DATASET_FILES[$i]}"
  dest="${DESTS[$i]}"
  expected="${EXPECTED_SHA256S[$i]}"
  src="${tmpdir}/${dataset_file}"

  if [[ ! -f "${src}" ]]; then
    echo "Download failed: ${src} not found" >&2
    exit 1
  fi

  actual="$(sha256sum "${src}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "sha256 mismatch for downloaded data: ${dataset_file}" >&2
    echo "expected: ${expected}" >&2
    echo "actual:   ${actual}" >&2
    exit 1
  fi

  mkdir -p "$(dirname "${dest}")"
  cp "${src}" "${dest}"
  echo "Downloaded ${dest}"
  ls -lh "${dest}"
done
