#!/usr/bin/env bash
set -euo pipefail

repo=/home/xue/UFO-g1depth
run=/data/xue/UFO/runs/PBFM_g1_fb_depth_canonical_plane_D001_lafan80_100style20_horizonfix_8gpu_20260830_214530

cd "$repo"
export UFO_CACHE_DIR=/data/xue/UFO/cache
export BFMZERO_MJLAB_CACHE_DIR=/data/xue/UFO/cache
export TMPDIR=/data/xue/UFO/cache/tmp
export TEMP=/data/xue/UFO/cache/tmp
export TMP=/data/xue/UFO/cache/tmp
export PYTHONPYCACHEPREFIX=/data/xue/UFO/cache/pycache
mkdir -p "$TMPDIR" "$PYTHONPYCACHEPREFIX"

exec .venv/bin/python scripts/watch_training_autoresume.py \
  --work-dir "$run" \
  --target-global-steps 192020480 \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --poll-seconds 3 \
  --stable-polls 2 \
  -- \
  .venv/bin/python -m humanoidverse.train \
  --agent fb_depth \
  --terrain-mode rp1_simple \
  --gpu-ids all \
  --work-dir "$run" \
  --robot-config configs/robots/g1_29dof.yaml \
  --num-envs 1024 \
  --num-env-steps 192020480 \
  --checkpoint-every-steps 9600000 \
  --tracking-eval-every-steps 3200000 \
  --same-z-eval-every-steps 0 \
  --data-path humanoidverse/data/lafan_29dof_10s-clipped.pkl /data/xue/bfmzero/data/100style_near10s.pkl \
  --data-mix-weights 0.8 0.2 \
  --buffer-size 5120000 \
  --prior-plane-envs 128 \
  --expert-buffer-cache \
  --gradient-sync ddp \
  --ddp-bucket-cap-mb 25 \
  --num-agent-updates 16 \
  --seed 4728 \
  --use-wandb \
  --wandb-project PBFM \
  --wandb-run-name PBFM_g1_fb_depth_canonical_plane_D001_lafan80_100style20_horizonfix_8gpu_20260830_214530
