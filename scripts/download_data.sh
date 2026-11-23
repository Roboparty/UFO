#!/usr/bin/env bash
set -euo pipefail

DATASET_REPO="${UFO_DATASET_REPO:-xuewang/UFO-MotionData}"
DATASET_REVISION="${UFO_DATASET_REVISION:-main}"
DATASET_FILES=(
  "g1/lafan_29dof_10s-clipped.pkl"
  "g1/lafan_29dof.pkl"
)
DESTS=(
  "ufo/data/lafan_29dof_10s-clipped.pkl"
  "ufo/data/lafan_29dof.pkl"
)
EXPECTED_SHA256S=(
  "7f5aa36957808ee2e972472b18add8510533742710ba312d8b8c6e6014f1c010"
  "f3a0c2810363f5c50bf4146fa2db33c1ff5b90d00cb7c0bc2aa4622696375e11"
)

usage() {
  cat <<USAGE
Usage: bash scripts/download_data.sh [g1_lafan|rp1_lafan|rpo_lafan]

Downloads full motions and 10-second training clips from Hugging Face:
  https://huggingface.co/datasets/${DATASET_REPO}
  g1_lafan:  G1 29-DoF, 40 motions
  rp1_lafan: RP1.3 24-DoF, 55 motions
  rpo_lafan: RPO 23-DoF, 56 motions

Environment overrides:
  UFO_DATASET_REPO=${DATASET_REPO}
  UFO_DATASET_REVISION=${DATASET_REVISION}
USAGE
}

case "${1:-g1_lafan}" in
  g1_lafan|lafan|g1)
    ;;
  rp1_lafan|rp1)
    DATASET_FILES=(
      "rp1/rp1_24dof_lafan_v7.pkl"
      "rp1/rp1_24dof_lafan_v7_10s-clipped.pkl"
    )
    DESTS=(
      "ufo/data/rp1_24dof_lafan_v7.pkl"
      "ufo/data/rp1_24dof_lafan_v7_10s-clipped.pkl"
    )
    EXPECTED_SHA256S=(
      "5cf6c98c298ad148b9182dc8dda5701a73224653325beee7e5ccb38f7eb0b75c"
      "c6587383d5888782f0dc8ff55b2ce2fd28e22008cd69fd8534e826a38aa4bc1b"
    )
    ;;
  rpo_lafan|rpo)
    DATASET_FILES=(
      "rpo/rpo_23dof_lafan_grounded.pkl"
      "rpo/rpo_23dof_lafan_grounded_10s-clipped.pkl"
    )
    DESTS=(
      "ufo/data/rpo_23dof_lafan_grounded.pkl"
      "ufo/data/rpo_23dof_lafan_grounded_10s-clipped.pkl"
    )
    EXPECTED_SHA256S=(
      "debf689bf1c555f0a1ba5d14c5b49589c96408cb8d690ca18b617bbe7d83466d"
      "a3e4c3a0a80c4079c6aa4929eb95de6440ebdbd9f065ae7bdf68abfff5ce87e7"
    )
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

if [[ -n "${HF_BIN:-}" ]]; then
  :
elif [[ -x ".venv/bin/hf" ]]; then
  HF_BIN=".venv/bin/hf"
elif command -v hf >/dev/null 2>&1; then
  HF_BIN="$(command -v hf)"
else
  HF_BIN=""
fi

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python3"
fi

download_one() {
  local dataset_file="$1"

  if [[ -n "${HF_BIN}" ]]; then
    "${HF_BIN}" download "${DATASET_REPO}" \
      "${dataset_file}" \
      --repo-type dataset \
      --revision "${DATASET_REVISION}" \
      --local-dir "${tmpdir}"
    return
  fi

  if "${PYTHON_BIN}" - <<'PYTEST' >/dev/null 2>&1
import huggingface_hub
PYTEST
  then
    "${PYTHON_BIN}" - "${DATASET_REPO}" "${dataset_file}" "${DATASET_REVISION}" "${tmpdir}" <<'PYDL'
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
