#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
media_dir="$repo_root/website/public/media"
mkdir -p "$media_dir"

copy_video() {
  local source_path="$1"
  local target_name="$2"
  if [[ -f "$source_path" ]]; then
    cp "$source_path" "$media_dir/$target_name"
    printf 'synced %s\n' "$target_name"
  else
    printf 'missing %s\n' "$source_path" >&2
  fi
}

cartwheel_run="/data/xue/bfmzero-mjlab/runs/lafan_cartwheel95_05_fixedmix_8gpu_auxsafe_lr1_v3_z10/tracking_inference_mjlab"
formal_run="/data/xue/bfmzero-mjlab/runs/formal_8gpu_mimiclite_dc_wandb"

copy_video "$cartwheel_run/tracking_mjlab_18.mp4" "cartwheel_18.mp4"
copy_video "$cartwheel_run/tracking_mjlab_22.mp4" "cartwheel_22.mp4"
copy_video "$cartwheel_run/tracking_mjlab_29.mp4" "cartwheel_29.mp4"
copy_video "$formal_run/tracking_inference_mjlab/tracking_mjlab_6.mp4" "lafan_6.mp4"
copy_video "$formal_run/reward_inference_mjlab/videos/sitonground.mp4" "sitonground.mp4"
copy_video "$formal_run/reward_inference_mjlab/videos/raisearms-l-l.mp4" "raisearms_l_l.mp4"
copy_video "$formal_run/reward_inference_mjlab/videos/crouch-0.25.mp4" "crouch_025.mp4"
copy_video "$formal_run/goal_inference_mjlab/videos/goal_mjlab.mp4" "goal_mjlab.mp4"
