#!/usr/bin/env bash
set -euo pipefail

repo=/home/xue/UFO
run=/data/xue/UFO/runs/PBFM_g1_fb_terrain_split10_8gpu_20260823_030740
model="$run/milestone_evaluations/192M"
out="${PBFM_OUTPUT_DIR:-$run/tracking_inference_step192020480_60s_6terrain}"
mkdir -p "$out/logs"

IFS=, read -r -a terrains <<< "${PBFM_TERRAINS:-flat,slope,stairs_up,stairs_down,rough,platforms}"
IFS=, read -r -a gpus <<< "${PBFM_GPUS:-0,1,2,5,6,7}"
IFS=';' read -r -a motion_id_sequences <<< "${PBFM_MOTION_ID_SEQUENCES:-159,160,161,162,163,164,165;0,1,2,3,4,5,6;146,147,148,149,150,151,152}"
IFS=, read -r -a motion_names <<< "${PBFM_MOTION_NAMES:-walk1_subject5_fullmotion,fallAndGetUp1_subject4_fullmotion,dance1_subject2_fullmotion}"
episode_steps="${PBFM_EPISODE_STEPS:-3000}"
duration_label="${PBFM_DURATION_LABEL:-60s}"

if [[ ${#terrains[@]} -ne ${#gpus[@]} ]]; then
  echo "PBFM_TERRAINS and PBFM_GPUS must contain the same number of entries" >&2
  exit 2
fi
if [[ ${#motion_id_sequences[@]} -ne ${#motion_names[@]} ]]; then
  echo "PBFM_MOTION_ID_SEQUENCES and PBFM_MOTION_NAMES must contain the same number of entries" >&2
  exit 2
fi

cd "$repo"
pids=()
for index in "${!terrains[@]}"; do
  terrain=${terrains[$index]}
  gpu=${gpus[$index]}
  (
    for motion_index in "${!motion_id_sequences[@]}"; do
      motion_ids=${motion_id_sequences[$motion_index]}
      motion_id=${motion_ids%%,*}
      motion_name=${motion_names[$motion_index]}
      stem="${motion_name}_${duration_label}_${terrain}"
      CUDA_VISIBLE_DEVICES="$gpu" \
        PBFM_FULL_MOTION_IDS="$motion_ids" \
        PBFM_FULL_MOTION_NAME="$motion_name" \
        MUJOCO_EGL_DEVICE_ID=0 \
        MUJOCO_GL=egl \
        PYOPENGL_PLATFORM=egl \
        .venv/bin/python -c '
import os
import torch
import humanoidverse.terrain_transfer_inference as module

original = module._compute_goal_or_tracking_z

def extended_tracking(args, model, base_cfg):
    motion_ids = [int(value) for value in os.environ["PBFM_FULL_MOTION_IDS"].split(",")]
    original_motion_id = args.motion_id
    z_parts = []
    initial_target_states = None
    try:
        for motion_id in motion_ids:
            args.motion_id = motion_id
            z_part, _identifier, target_states = original(args, model, base_cfg)
            z_parts.append(z_part)
            if initial_target_states is None:
                initial_target_states = target_states
    finally:
        args.motion_id = original_motion_id
    z = torch.cat(z_parts, dim=0)
    if z.shape[0] < args.episode_length:
        raise RuntimeError(
            f"full motion has only {z.shape[0]} tracking steps; requested {args.episode_length}"
        )
    identifier = "fullmotion:" + os.environ["PBFM_FULL_MOTION_NAME"]
    return z[: args.episode_length], identifier, initial_target_states

module._compute_goal_or_tracking_z = extended_tracking
module.main()
' \
        --model-folder "$model" \
        --prompt-type tracking \
        --terrains "$terrain" \
        --motion-id "$motion_id" \
        --device cuda:0 \
        --seed 4728 \
        --episode-length "$episode_steps" \
        --fall-clearance 0.45 \
        --save-mp4 \
        --render-size 480 \
        --fps 50 \
        --output "$out/${stem}.json" \
        >"$out/logs/${stem}.log" 2>&1
    done
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
