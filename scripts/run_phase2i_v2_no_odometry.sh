#!/usr/bin/env bash
set -euo pipefail

# Reproducible Phase-2I v2 entry point.  Every command is odometry-free and
# refuses to overwrite an existing output.
repo_root="${PHASE2I_REPO_ROOT:-/home/xue/UFO}"
eval_root="${PHASE2I_EVAL_ROOT:-/data/xue/UFO/evaluations/phase2i_v2_no_odometry_20260827}"
teacher="${PHASE2I_TEACHER:-/data/xue/UFO/runs/PBFM_g1_fb_terrain_split10_8gpu_20260823_030740/milestone_evaluations/192M}"
gpu="${PHASE2I_GPU:-0}"
python_bin="${repo_root}/.venv/bin/python"

cd "${repo_root}"

require_new_path() {
  local path="$1"
  if [[ -e "${path}" ]]; then
    echo "Refusing to overwrite existing output: ${path}" >&2
    exit 2
  fi
}

collect_common=(
  -m humanoidverse.terrain_perception_collection
  --model-folder "${teacher}"
  --device cuda:0
  --width 480
  --height 270
  --target-width 64
  --target-height 36
  --min-range 0.10
  --max-range 2.0
  --projection-mode local_no_odometry
  --self-occlusion
  --depth-crop full
)

collect_dataset() {
  local split="$1"
  local output_dir num_envs num_steps seed
  case "${split}" in
    train)
      output_dir="${eval_root}/dataset_train_mixed_14to18cm"
      num_envs=1024
      num_steps=3400
      seed=7700
      ;;
    validation)
      output_dir="${eval_root}/dataset_validation_mixed_14to18cm"
      num_envs=512
      num_steps=1700
      seed=8700
      ;;
    *)
      echo "dataset split must be train or validation" >&2
      exit 2
      ;;
  esac
  require_new_path "${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" "${collect_common[@]}" \
    --output-dir "${output_dir}" \
    --num-envs "${num_envs}" \
    --num-steps "${num_steps}" \
    --chunk-steps 128 \
    --terrain mixed \
    --terrain-difficulty-range 0.5 1.0 \
    --seed "${seed}" \
    --depth-dr-preset phase2i_v2 \
    --timing-dr-preset phase2i_v2 \
    --calibration-dr-preset phase2i_v2
}

collect_baseline_dataset() {
  local split="$1"
  local output_dir num_envs num_steps seed
  case "${split}" in
    train)
      output_dir="${eval_root}/dataset_train_full_baseline_v1_14to18cm"
      num_envs=1024
      num_steps=3400
      seed=7700
      ;;
    validation)
      output_dir="${eval_root}/dataset_validation_full_baseline_v1_14to18cm"
      num_envs=512
      num_steps=1700
      seed=8700
      ;;
    *)
      echo "baseline dataset split must be train or validation" >&2
      exit 2
      ;;
  esac
  require_new_path "${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" "${collect_common[@]}" \
    --output-dir "${output_dir}" \
    --num-envs "${num_envs}" \
    --num-steps "${num_steps}" \
    --chunk-steps 128 \
    --terrain mixed \
    --terrain-difficulty-range 0.5 1.0 \
    --seed "${seed}" \
    --depth-dr-preset phase2i_v1 \
    --timing-dr-preset phase2i_v1 \
    --calibration-dr-preset none
}

collect_holdout() {
  local corruption="$1"
  local terrain="$2"
  local seed="$3"
  local output_dir="${eval_root}/holdout_${corruption}_${terrain}_16cm"
  local -a dr_args
  case "${corruption}" in
    clean)
      dr_args=(--depth-dr-preset deployment_clean --timing-dr-preset deployment_clean)
      ;;
    dr)
      dr_args=(
        --depth-dr-preset phase2i_v2
        --timing-dr-preset phase2i_v2
        --calibration-dr-preset phase2i_v2
      )
      ;;
    *)
      echo "holdout corruption must be clean or dr" >&2
      exit 2
      ;;
  esac
  require_new_path "${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" "${collect_common[@]}" \
    --output-dir "${output_dir}" \
    --num-envs 256 \
    --num-steps 500 \
    --chunk-steps 128 \
    --terrain "${terrain}" \
    --terrain-difficulty 0.75 \
    --seed "${seed}" \
    "${dr_args[@]}"
}

train_model() {
  local loss_mode="$1"
  local output_dir="${eval_root}/convgru16_dr_${loss_mode}_loss"
  require_new_path "${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m humanoidverse.train_terrain_perception \
    --dataset-dir "${eval_root}/dataset_train_mixed_14to18cm" \
    --validation-dataset-dir "${eval_root}/dataset_validation_mixed_14to18cm" \
    --output-dir "${output_dir}" \
    --sequence-steps 31 \
    --history-seconds 0.6 \
    --epochs 20 \
    --batch-size 256 \
    --learning-rate 3e-4 \
    --hidden-channels 16 \
    --proprio-channels 8 \
    --history-mode no_odometry \
    --loss-mode "${loss_mode}" \
    --num-workers 4 \
    --seed 7900 \
    --device cuda:0
}

train_clean_ablation() {
  local ablation="$1"
  local output_dir
  case "${ablation}" in
    full_baseline)
      output_dir="${eval_root}/convgru16_full_fov_baseline_v1_loss"
      ;;
    roi_full)
      # ROI coverage selected full-FOV, so this is intentionally the same
      # data/config/compute as full_baseline but remains a separately trained
      # run for the four-way ablation ledger.
      output_dir="${eval_root}/convgru16_roi_full_v1_loss"
      ;;
    *)
      echo "clean ablation must be full_baseline or roi_full" >&2
      exit 2
      ;;
  esac
  require_new_path "${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m humanoidverse.train_terrain_perception \
    --dataset-dir "${eval_root}/dataset_train_full_baseline_v1_14to18cm" \
    --validation-dataset-dir "${eval_root}/dataset_validation_full_baseline_v1_14to18cm" \
    --output-dir "${output_dir}" \
    --sequence-steps 31 \
    --history-seconds 0.6 \
    --epochs 20 \
    --batch-size 256 \
    --learning-rate 3e-4 \
    --hidden-channels 16 \
    --proprio-channels 8 \
    --history-mode no_odometry \
    --loss-mode baseline \
    --num-workers 4 \
    --samples-per-epoch 1932842 \
    --seed 7900 \
    --device cuda:0
}

evaluate_model() {
  local checkpoint="$1"
  local dataset="$2"
  local tag="$3"
  local output="${eval_root}/reports/${tag}.json"
  require_new_path "${output}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m humanoidverse.evaluate_terrain_perception \
    --checkpoint "${checkpoint}" \
    --dataset-dir "${dataset}" \
    --output "${output}" \
    --batch-size 1280 \
    --num-workers 4 \
    --device cuda:0
}

distill_actor() {
  local perception_checkpoint="$1"
  local tag="$2"
  local output_dir="${eval_root}/phase2j_${tag}"
  require_new_path "${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m humanoidverse.distill_temporal_terrain_actor \
    --model-folder "${teacher}" \
    --perception-checkpoint "${perception_checkpoint}" \
    --latent "${teacher}/forward_latent.pt" \
    --output-dir "${output_dir}" \
    --device cuda:0 \
    --high-stairs-envs 384 \
    --mixed-envs 256 \
    --training-steps 1000 \
    --learning-rate 3e-5 \
    --anchor-weight 1.0 \
    --max-grad-norm 1.0 \
    --high-stairs-min-difficulty 0.5555555555555556 \
    --checkpoint-every 500 \
    --milestone-steps 500 1000 \
    --log-every 25 \
    --seed 8900 \
    --max-episode-length-s 20.0 \
    --width 480 \
    --height 270 \
    --min-range 0.10 \
    --max-range 2.0
}

usage() {
  cat <<'EOF'
Usage:
  run_phase2i_v2_no_odometry.sh collect-dataset train|validation
  run_phase2i_v2_no_odometry.sh collect-baseline-dataset train|validation
  run_phase2i_v2_no_odometry.sh collect-holdout clean|dr stairs_up|stairs_down|mixed SEED
  run_phase2i_v2_no_odometry.sh train baseline|phase2i_v2
  run_phase2i_v2_no_odometry.sh train-clean-ablation full_baseline|roi_full
  run_phase2i_v2_no_odometry.sh evaluate CHECKPOINT DATASET TAG
  run_phase2i_v2_no_odometry.sh distill PERCEPTION_CHECKPOINT TAG

Set PHASE2I_GPU to choose the physical GPU. Outputs are never overwritten.
EOF
}

case "${1:-}" in
  collect-dataset)
    [[ $# -eq 2 ]] || { usage; exit 2; }
    collect_dataset "$2"
    ;;
  collect-holdout)
    [[ $# -eq 4 ]] || { usage; exit 2; }
    collect_holdout "$2" "$3" "$4"
    ;;
  collect-baseline-dataset)
    [[ $# -eq 2 ]] || { usage; exit 2; }
    collect_baseline_dataset "$2"
    ;;
  train)
    [[ $# -eq 2 ]] || { usage; exit 2; }
    [[ "$2" == baseline || "$2" == phase2i_v2 ]] || { usage; exit 2; }
    train_model "$2"
    ;;
  train-clean-ablation)
    [[ $# -eq 2 ]] || { usage; exit 2; }
    train_clean_ablation "$2"
    ;;
  evaluate)
    [[ $# -eq 4 ]] || { usage; exit 2; }
    evaluate_model "$2" "$3" "$4"
    ;;
  distill)
    [[ $# -eq 3 ]] || { usage; exit 2; }
    distill_actor "$2" "$3"
    ;;
  *)
    usage
    exit 2
    ;;
esac
