#!/usr/bin/env bash
set -euo pipefail

repo=/home/xue/UFO
run=/data/xue/UFO/runs/PBFM_g1_fb_terrain_split10_8gpu_20260823_030740
model="$run/milestone_evaluations/192M"
out="${PBFM_OUTPUT_DIR:-$run/goal_sequence_inference_step192020480_42s_6terrain}"
goal_data="${PBFM_GOAL_DATA:-humanoidverse/data/lafan_29dof.pkl}"
goal_json="${PBFM_GOAL_JSON:-humanoidverse/data/robots/g1/goal_frames_lafan29dof.json}"

IFS=, read -r -a terrains <<< "${PBFM_TERRAINS:-flat,slope,stairs_up,stairs_down,rough,platforms}"
IFS=, read -r -a gpus <<< "${PBFM_GPUS:-0,1,2,3,4,5}"
goal_index="${PBFM_GOAL_INDEX:-0}"
goal_frame="${PBFM_GOAL_FRAME:-2193}"
episode_steps="${PBFM_EPISODE_STEPS:-2100}"
goal_switch_interval="${PBFM_GOAL_SWITCH_INTERVAL:-100}"

if [[ ${#terrains[@]} -ne ${#gpus[@]} ]]; then
  echo "PBFM_TERRAINS and PBFM_GPUS must contain the same number of entries" >&2
  exit 2
fi

mkdir -p "$out/logs"
cd "$repo"

pids=()
for index in "${!terrains[@]}"; do
  terrain=${terrains[$index]}
  gpu=${gpus[$index]}
  stem="goal_sequence_i${goal_switch_interval}_${terrain}"
  CUDA_VISIBLE_DEVICES="$gpu" \
    MUJOCO_EGL_DEVICE_ID=0 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    .venv/bin/python -m humanoidverse.terrain_transfer_inference \
      --model-folder "$model" \
      --prompt-type goal \
      --terrains "$terrain" \
      --data-path "$goal_data" \
      --goal-json "$goal_json" \
      --goal-index "$goal_index" \
      --goal-frame "$goal_frame" \
      --goal-sequence \
      --goal-switch-interval "$goal_switch_interval" \
      --device cuda:0 \
      --seed 4728 \
      --episode-length "$episode_steps" \
      --fall-clearance 0.45 \
      --save-mp4 \
      --render-size 480 \
      --fps 50 \
      --output "$out/${stem}.json" \
      >"$out/logs/${stem}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
