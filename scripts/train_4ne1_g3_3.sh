#!/usr/bin/env bash
# Full TeCH/FB training of UFO on the NEURA 4NE-1 G3.3 (single GPU, W&B logging, preview videos).
#
# Usage (from the repo root):
#   /usr/bin/tmux new-session -d -s ufo_4ne1_train -c "$PWD" \
#     'bash scripts/train_4ne1_g3_3.sh 2>&1 | tee -a runs/ufo_fb_4ne1_g3_3.log'
#
# Environment overrides: AGENT (tech by default, or fb), GPU (CUDA device id, default 0),
# WORK_DIR, NUM_ENVS, BUFFER_SIZE, NUM_ENV_STEPS, CHECKPOINT_EVERY, WANDB_RUN_NAME,
# EXTRA_ARGS (appended verbatim).
# Re-running with the same WORK_DIR resumes from the last checkpoint (agent + replay buffer).
#
# Sizing notes (see docs/superpowers/specs/2026-09-11-neura-4ne1-g3-3-port-design.md):
# - the 70-body 4NE-1 observation makes each replay transition ~1.6k floats, so the
#   trajectory buffer is halved to 2.56M transitions (~16 GB on the GPU);
# - checkpoints every 9.6M env steps keep the replay-buffer dump cadence reasonable;
# - preview videos (walk1_subject1=65, run1_subject2=59, dance1_subject1=5) are rendered by a
#   background subprocess after every checkpoint and uploaded to W&B under video/.
set -euo pipefail

cd "$(dirname "$0")/.."

# Training prints are block-buffered when piped (tmux | tee); make them appear immediately.
export PYTHONUNBUFFERED=1

GPU="${GPU:-0}"
AGENT="${AGENT:-tech}"
case "${AGENT}" in
  tech) UPDATE_Z_EVERY_STEP=10 ;;
  fb) UPDATE_Z_EVERY_STEP=100 ;;
  *)
    echo "Unsupported AGENT=${AGENT}; expected tech or fb" >&2
    exit 2
    ;;
esac
WORK_DIR="${WORK_DIR:-runs/ufo_${AGENT}_4ne1_g3_3_groundoffset}"
NUM_ENVS="${NUM_ENVS:-1024}"
BUFFER_SIZE="${BUFFER_SIZE:-2560000}"
NUM_ENV_STEPS="${NUM_ENV_STEPS:-192000000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-9600000}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-ufo_${AGENT}_4ne1_g3_3_groundoffset}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "[train_4ne1_g3_3] AGENT=${AGENT} GPU=${GPU} WORK_DIR=${WORK_DIR} NUM_ENVS=${NUM_ENVS} BUFFER_SIZE=${BUFFER_SIZE} NUM_ENV_STEPS=${NUM_ENV_STEPS}"

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="${GPU}" \
./run_train.sh \
  --agent "${AGENT}" \
  --robot-config configs/robots/neura_4ne1_g3_3.yaml \
  --data-manifest configs/data/lafan_neura_4ne1_g3_3.yaml \
  --gpu-ids single \
  --num-envs "${NUM_ENVS}" \
  --num-env-steps "${NUM_ENV_STEPS}" \
  --buffer-size "${BUFFER_SIZE}" \
  --checkpoint-every-steps "${CHECKPOINT_EVERY}" \
  --update-z-every-step "${UPDATE_Z_EVERY_STEP}" \
  --work-dir "${WORK_DIR}" \
  --use-wandb \
  --wandb-run-name "${WANDB_RUN_NAME}" \
  --video-every-checkpoints 1 \
  --video-motion-ids 65 59 5 \
  --video-max-steps 400 \
  --video-render-size 320 \
  ${EXTRA_ARGS}
