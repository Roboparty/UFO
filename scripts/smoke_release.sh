#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

echo "[smoke_release] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

uv run python tests/test_reference_observations.py
uv run python tests/test_motion_data_adapter.py
uv run python tests/test_import_tools.py
uv run python tests/test_robot_config_training.py
uv run python tests/test_robot_config_goal_reward_inference.py
uv run python tests/test_robot_config_onnx_export.py
uv run python -m compileall -q humanoidverse tests
git diff --check

rm -rf /tmp/ufo_smoke_release_train
./run_train.sh \
  --agent fb \
  --data-manifest configs/data/example_mix.yaml \
  --gpu-ids single \
  --smoke \
  --work-dir /tmp/ufo_smoke_release_train

rm -f /tmp/g1_robot_state_release_smoke.yaml
uv run python -m humanoidverse.tools.data_build \
  --robot configs/robots/g1_29dof.yaml \
  --source humanoidverse/data/examples/g1_robot_state_sample.csv \
  --format robot_state_csv \
  --name g1_robot_state_sample \
  --fps 50 \
  --clip-seconds 10 \
  --out /tmp/g1_robot_state_release_smoke.yaml \
  --rebuild-cache \
  --force

git diff --check
git status --short
