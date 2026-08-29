#!/usr/bin/env bash
set -u

cd /home/xue/UFO

output_root=/data/xue/UFO/evaluations/phase2d_robustness_split10_192M_20260824/stairs_up
model_folder=/data/xue/UFO/runs/PBFM_g1_fb_terrain_split10_8gpu_20260823_030740
perception_checkpoint=/data/xue/UFO/evaluations/phase2b_temporal_split10_192M_20260823/convgru_default/best.pt
latent=/data/xue/UFO/runs/PBFM_g1_fb_terrain_split10_8gpu_20260823_030740/milestone_evaluations/192M/forward_latent.pt
actor_checkpoint=/data/xue/UFO/evaluations/phase2c_actor_distill_split10_192M_20260823_211943/milestones/actor_step_002000.pt
noise_seed=271828

conditions=(
  clean:nominal
  measurement:mild measurement:nominal measurement:strong
  dropout:mild dropout:nominal dropout:strong
  edge:mild edge:nominal edge:strong
  latency:mild latency:nominal latency:strong
  extrinsic:mild extrinsic:nominal extrinsic:strong
  combined:mild combined:nominal combined:strong
)

mkdir -p "${output_root}"

run_worker() {
  local gpu=$1
  local task_index=0
  local failed=0
  local condition_severity condition severity condition_dir seed output_dir summary

  for condition_severity in "${conditions[@]}"; do
    condition=${condition_severity%%:*}
    severity=${condition_severity##*:}
    condition_dir="${output_root}/${condition}_${severity}"
    if [[ "${condition}" == clean ]]; then
      condition_dir="${output_root}/clean"
    fi

    for seed in $(seq 6840 6849); do
      if ((task_index % 8 != gpu)); then
        task_index=$((task_index + 1))
        continue
      fi
      task_index=$((task_index + 1))
      output_dir="${condition_dir}/seed_${seed}"
      summary="${output_dir}/summary.json"
      if [[ -s "${summary}" ]]; then
        echo "GPU${gpu} SKIP condition=${condition} severity=${severity} seed=${seed}"
        continue
      fi

      mkdir -p "${output_dir}"
      echo "GPU${gpu} START condition=${condition} severity=${severity} seed=${seed}"
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
          --terrain stairs_up \
          --num-envs 256 \
          --episode-steps 1000 \
          --seed "${seed}" \
          --device cuda:0 \
          --modes temporal \
          --noise-condition "${condition}" \
          --noise-severity "${severity}" \
          --noise-seed "${noise_seed}" \
          >"${output_dir}/evaluation.log" 2>&1; then
        echo "GPU${gpu} DONE condition=${condition} severity=${severity} seed=${seed}"
      else
        echo "GPU${gpu} FAIL condition=${condition} severity=${severity} seed=${seed}"
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

date -Ins >"${output_root}/sweep_finished_at.txt"
exit "${failed}"
