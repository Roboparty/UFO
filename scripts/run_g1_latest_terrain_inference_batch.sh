#!/usr/bin/env bash
set -euo pipefail

repo=/home/xue/UFO
model=/data/xue/UFO/runs/PBFM_fb_terrain_impact_5terrain_8gpu_20260819_211059
data="$repo/humanoidverse/data/lafan_29dof.pkl"
out="$model/inference_step176168960_20260820"
mkdir -p "$out/logs" "$out/reward" "$out/tracking"

export UFO_CACHE_DIR=/data/xue/UFO/cache
export MUJOCO_GL=egl

run_tracking() {
  local gpu=$1 motion_id=$2 motion_name=$3 terrain=$4
  CUDA_VISIBLE_DEVICES="$gpu" "$repo/.venv/bin/python" -m humanoidverse.terrain_transfer_inference \
    --model-folder "$model" \
    --prompt-type tracking \
    --data-path "$data" \
    --device cuda:0 \
    --motion-id "$motion_id" \
    --terrains "$terrain" \
    --episode-length 5000 \
    --save-mp4 \
    --fps 50 \
    --output "$out/tracking/${motion_name}_100s_${terrain}.json"
}

case "${1:-}" in
  reward)
    CUDA_VISIBLE_DEVICES=0 "$repo/.venv/bin/python" -m humanoidverse.terrain_transfer_inference \
      --model-folder "$model" \
      --prompt-type reward \
      --data-path "$data" \
      --device cuda:0 \
      --reward-task move-ego-0-0.7 \
      --terrains flat,slope,stairs,rough,platforms \
      --patch-size 60 \
      --episode-length 1500 \
      --save-mp4 \
      --fps 50 \
      --output "$out/reward/forward_0.7ms_30s_5terrain.json"
    ;;
  walk_a_main)
    run_tracking 1 8 walk3_subject2 flat
    run_tracking 1 8 walk3_subject2 rough
    ;;
  walk_a_grade)
    run_tracking 2 8 walk3_subject2 slope
    run_tracking 2 8 walk3_subject2 stairs
    ;;
  getup_main)
    run_tracking 3 0 fallAndGetUp1_subject4 flat
    run_tracking 3 0 fallAndGetUp1_subject4 rough
    ;;
  getup_grade)
    run_tracking 4 0 fallAndGetUp1_subject4 slope
    run_tracking 4 0 fallAndGetUp1_subject4 stairs
    ;;
  walk_b_main)
    run_tracking 5 25 walk3_subject5 flat
    run_tracking 5 25 walk3_subject5 rough
    ;;
  walk_b_grade)
    run_tracking 6 25 walk3_subject5 slope
    run_tracking 6 25 walk3_subject5 stairs
    ;;
  run_main)
    run_tracking 5 3 run1_subject5 flat
    run_tracking 5 3 run1_subject5 rough
    run_tracking 5 3 run1_subject5 platforms
    ;;
  run_grade)
    run_tracking 6 3 run1_subject5 slope
    run_tracking 6 3 run1_subject5 stairs
    ;;
  fight_main)
    run_tracking 1 26 fightAndSports1_subject4 flat
    run_tracking 3 26 fightAndSports1_subject4 rough
    ;;
  fight_grade)
    run_tracking 2 26 fightAndSports1_subject4 slope
    run_tracking 4 26 fightAndSports1_subject4 stairs
    ;;
  fight_platforms)
    run_tracking 7 26 fightAndSports1_subject4 platforms
    ;;
  platforms)
    run_tracking 7 8 walk3_subject2 platforms
    run_tracking 7 0 fallAndGetUp1_subject4 platforms
    run_tracking 7 25 walk3_subject5 platforms
    ;;
  *)
    echo "usage: $0 {reward|walk_a_main|walk_a_grade|getup_main|getup_grade|walk_b_main|walk_b_grade|run_main|run_grade|fight_main|fight_grade|fight_platforms|platforms}" >&2
    exit 2
    ;;
esac
