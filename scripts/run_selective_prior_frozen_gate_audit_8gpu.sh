#!/usr/bin/env bash
set -euo pipefail

repo_dir=${REPO_DIR:-/home/xue/UFO-g1depth-selective-prior}
run_dir=${RUN_DIR:-/data/xue/UFO/runs/PBFM_g1_fb_depth_selective_online_prior_QH0002_lafan_only_8gpu_20260904_003021}
expert_cache=${EXPERT_CACHE:-/data/xue/UFO/cache/expert_buffers/v3-20260904-002200-bfd0020cddf16845305b/expert_buffer.pt}
output_root=${OUTPUT_ROOT:-/data/xue/UFO/diagnostics/selective_prior_matched_gate_8rank}
python_bin=${PYTHON_BIN:-/home/xue/UFO-g1depth/.venv/bin/python}
windows=${WINDOWS_PER_RANK:-4096}
skip_offline_d=${SKIP_OFFLINE_D:-1}
offline_d_steps=${OFFLINE_D_STEPS:-512}
offline_d_batch_size=${OFFLINE_D_BATCH_SIZE:-1024}
offline_d_grad_penalty=${OFFLINE_D_GRAD_PENALTY:-10.0}

cd "$repo_dir"
mkdir -p "$output_root"

pids=()
for rank in 0 1 2 3 4 5 6 7; do
  output_dir="$output_root/rank_$rank"
  mkdir -p "$output_dir"
  extra_args=()
  if [[ "$skip_offline_d" == "1" ]]; then
    extra_args+=(--skip-offline-d)
  else
    extra_args+=(
      --offline-d-steps "$offline_d_steps"
      --offline-d-batch-size "$offline_d_batch_size"
      --offline-d-grad-penalty "$offline_d_grad_penalty"
    )
  fi
  CUDA_VISIBLE_DEVICES="$rank" "$python_bin" -m humanoidverse.selective_prior_audit \
    --run-dir "$run_dir" \
    --buffer-rank "$rank" \
    --expert-cache "$expert_cache" \
    --output-dir "$output_dir" \
    --device cuda:0 \
    --windows "$windows" \
    --seed "$((4831 + rank))" \
    "${extra_args[@]}" \
    > "$output_dir/audit.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
