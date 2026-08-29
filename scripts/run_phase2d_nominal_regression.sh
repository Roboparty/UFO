#!/usr/bin/env bash
set -u

cd /home/xue/UFO

output_root=/data/xue/UFO/evaluations/phase2d_robustness_split10_192M_20260824/nominal_regression
model_folder=/data/xue/UFO/runs/PBFM_g1_fb_terrain_split10_8gpu_20260823_030740
perception_checkpoint=/data/xue/UFO/evaluations/phase2b_temporal_split10_192M_20260823/convgru_default/best.pt
latent=/data/xue/UFO/runs/PBFM_g1_fb_terrain_split10_8gpu_20260823_030740/milestone_evaluations/192M/forward_latent.pt
actor_checkpoint=/data/xue/UFO/evaluations/phase2c_actor_distill_split10_192M_20260823_211943/milestones/actor_step_002000.pt
noise_seed=271828

conditions=(clean measurement dropout edge latency extrinsic combined)
terrains=(flat slope rough platforms stairs_down stairs_up)

mkdir -p "${output_root}"

run_worker() {
  local gpu=$1
  local task_index=0
  local failed=0
  local condition terrain output_dir summary
  for condition in "${conditions[@]}"; do
    for terrain in "${terrains[@]}"; do
      if ((task_index % 8 != gpu)); then
        task_index=$((task_index + 1))
        continue
      fi
      task_index=$((task_index + 1))
      output_dir="${output_root}/${condition}/${terrain}"
      summary="${output_dir}/summary.json"
      if [[ -s "${summary}" ]]; then
        echo "GPU${gpu} SKIP condition=${condition} terrain=${terrain}"
        continue
      fi

      mkdir -p "${output_dir}"
      echo "GPU${gpu} START condition=${condition} severity=nominal terrain=${terrain}"
      if CUDA_VISIBLE_DEVICES="${gpu}" \
        MUJOCO_GL=egl \
        PYOPENGL_PLATFORM=egl \
        UFO_CACHE_DIR=/data/xue/UFO/cache \
        .venv/bin/python -u -m humanoidverse.terrain_perception_closed_loop \
          --model-folder "${model_folder}" \
          --perception-checkpoint "${perception_checkpoint}" \
          --latent "${latent}" \
          --actor-checkpoint "${actor_checkpoint}" \
          --output-dir "${output_dir}" \
          --terrain "${terrain}" \
          --num-envs 64 \
          --episode-steps 1000 \
          --seed 6840 \
          --device cuda:0 \
          --modes temporal \
          --noise-condition "${condition}" \
          --noise-severity nominal \
          --noise-seed "${noise_seed}" \
          >"${output_dir}/evaluation.log" 2>&1; then
        echo "GPU${gpu} DONE condition=${condition} severity=nominal terrain=${terrain}"
      else
        echo "GPU${gpu} FAIL condition=${condition} severity=nominal terrain=${terrain}"
        failed=1
      fi
    done
  done
  return "${failed}"
}

failed=0
pids=()
for gpu in $(seq 0 7); do
  run_worker "${gpu}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done

date -Ins >"${output_root}/regression_finished_at.txt"
exit "${failed}"
