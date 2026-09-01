#!/usr/bin/env bash
set -euo pipefail

repo_dir=/home/xue/UFO-g1depth
run_name=PBFM_g1_fb_depth_heading_zeroctr_QH0002_D001_lafan80_highknee20_8gpu_20260902_002531
run_dir=/data/xue/UFO/runs/$run_name

cd "$repo_dir"

export UFO_CACHE_DIR=/data/xue/UFO/cache
export UFO_DATA_DIR=/data/xue/UFO/cache
export BFMZERO_MJLAB_CACHE_DIR=/data/xue/UFO/cache
export TMPDIR=/data/xue/UFO/cache/tmp
export TEMP=/data/xue/UFO/cache/tmp
export TMP=/data/xue/UFO/cache/tmp
export PYTHONPYCACHEPREFIX=/data/xue/UFO/cache/pycache
export TORCHINDUCTOR_CACHE_DIR=/data/xue/UFO/cache/torchinductor
export TRITON_CACHE_DIR=/data/xue/UFO/cache/triton
export CUDA_CACHE_PATH=/data/xue/UFO/cache/cuda
export WARP_CACHE_PATH=/data/xue/UFO/cache/warp
export TORCHRUNX_LOG_DIR="$run_dir/torchrunx"

mkdir -p "$run_dir" "$TMPDIR" "$PYTHONPYCACHEPREFIX" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$WARP_CACHE_PATH" "$TORCHRUNX_LOG_DIR"

exec .venv/bin/python scripts/watch_training_autoresume.py \
  --work-dir "$run_dir" \
  --target-global-steps 192020480 \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --poll-seconds 3 \
  --stable-polls 2 \
  --allow-fresh \
  -- \
  .venv/bin/python -m humanoidverse.train \
  --agent fb_depth \
  --terrain-mode rp1_simple \
  --gpu-ids all \
  --work-dir "$run_dir" \
  --robot-config configs/robots/g1_29dof.yaml \
  --num-envs 1024 \
  --num-env-steps 192020480 \
  --checkpoint-every-steps 9600000 \
  --tracking-eval-every-steps 3200000 \
  --same-z-eval-every-steps 0 \
  --data-path humanoidverse/data/lafan_29dof_10s-clipped.pkl /data/xue/bfmzero/data/g1_stair_highknee_forward_50fps.pkl \
  --data-mix-weights 0.8 0.2 \
  --buffer-size 5120000 \
  --prior-plane-envs 128 \
  --expert-buffer-cache \
  --gradient-sync ddp \
  --ddp-bucket-cap-mb 25 \
  --num-agent-updates 16 \
  --heading-context \
  --heading-reg-coeff 0.002 \
  --seed 4831 \
  --use-wandb \
  --wandb-project PBFM \
  --wandb-run-name "$run_name"
