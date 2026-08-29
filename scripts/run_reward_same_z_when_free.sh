#!/usr/bin/env bash
set -euo pipefail

model=/data/xue/UFO/runs/PBFM_fb_terrain_impact_5terrain_8gpu_20260819_211059
root="$model/inference_step176168960_20260820"
output="$root/reward_same_z"
latent="$output/forward_0.7ms_z.pt"
log_dir="$output/logs"
python=/home/xue/UFO/.venv/bin/python
data=/home/xue/UFO/humanoidverse/data/lafan_29dof.pkl
allowed_gpus=(0 1 2 5 6)

mkdir -p "$output" "$log_dir"

gpu_is_idle() {
    local gpu=$1
    local values memory utilization
    values=$(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    IFS=, read -r memory utilization <<< "$values"
    memory=${memory//[[:space:]]/}
    utilization=${utilization//[[:space:]]/}
    (( memory < 10000 && utilization < 10 ))
}

if [[ "${PBFM_ALLOW_SHARED_GPUS:-0}" != 1 ]]; then
    while true; do
        all_idle=1
        for gpu in "${allowed_gpus[@]}"; do
            if ! gpu_is_idle "$gpu"; then
                all_idle=0
                break
            fi
        done
        if (( all_idle )); then
            sleep 30
            stable=1
            for gpu in "${allowed_gpus[@]}"; do
                if ! gpu_is_idle "$gpu"; then
                    stable=0
                    break
                fi
            done
            (( stable )) && break
        fi
        sleep 30
    done
fi

common=(
    -m humanoidverse.terrain_transfer_inference
    --model-folder "$model"
    --prompt-type reward
    --data-path "$data"
    --device cuda:0
    --reward-task move-ego-0-0.7
    --patch-size 60
    --episode-length 1500
    --save-mp4
    --fps 50
)

CUDA_VISIBLE_DEVICES=0 UFO_CACHE_DIR=/data/xue/UFO/cache MUJOCO_GL=egl \
    "$python" "${common[@]}" \
    --terrains flat \
    --save-latent "$latent" \
    --output "$output/flat.json" \
    >"$log_dir/flat.log" 2>&1

terrains=(slope stairs rough platforms)
gpus=(1 2 5 6)
pids=()
for index in "${!terrains[@]}"; do
    terrain=${terrains[$index]}
    gpu=${gpus[$index]}
    CUDA_VISIBLE_DEVICES="$gpu" UFO_CACHE_DIR=/data/xue/UFO/cache MUJOCO_GL=egl \
        "$python" "${common[@]}" \
        --terrains "$terrain" \
        --load-latent "$latent" \
        --output "$output/$terrain.json" \
        >"$log_dir/$terrain.log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
(( status == 0 )) || exit "$status"

checksum=$(jq -r '.[0].z_checksum' "$output/flat.json")
for terrain in flat slope stairs rough platforms; do
    actual=$(jq -r '.[0].z_checksum' "$output/$terrain.json")
    [[ "$actual" == "$checksum" ]]
    frames=$(ffprobe -v error -count_frames -select_streams v:0 \
        -show_entries stream=nb_read_frames -of default=nw=1:nk=1 \
        "$output/${terrain}_${terrain}.mp4")
    [[ "$frames" == 1500 ]]
done

deliver="$root/deliverables/reward_forward_0.7ms_30s_same_z"
mkdir -p "$deliver"
for terrain in flat slope stairs rough platforms; do
    ln -f "$output/${terrain}_${terrain}.mp4" "$deliver/$terrain.mp4"
done
printf '%s\n' "$checksum" > "$deliver/z_checksum.txt"
printf 'completed %s checksum=%s\n' "$(date --iso-8601=seconds)" "$checksum" > "$output/COMPLETE"
