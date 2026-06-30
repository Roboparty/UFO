

# BFM-Zero MJLab Quick Start

MJLab/MuJoCo-Warp training does not require IsaacLab. Commands below assume the repo is at `/home/xue/bfmzero-mjlab`.

## Install

Install dependencies with `uv`:

```bash
cd /home/xue/bfmzero-mjlab
uv sync
```

## Train

8-GPU formal training; `--num-envs` is per GPU and `--num-env-steps` is the global step budget:

```bash
cd /home/xue/bfmzero-mjlab

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./run_mjlab.sh \
  --gpu-ids all \
  --num-envs 1024 \
  --num-env-steps 384000000 \
  --work-dir /data/xue/bfmzero-mjlab/runs/formal_8gpu_mjlab_clean_eval \
  --use-wandb
```

Remove `--use-wandb` if you do not want W&B logging.

## Tracking Inference

Run after a checkpoint exists; saves expert-vs-policy video and `z`:

```bash
cd /home/xue/bfmzero-mjlab

CUDA_VISIBLE_DEVICES=0 \
uv run python -m humanoidverse.tracking_inference_mjlab \
  --model-folder /data/xue/bfmzero-mjlab/runs/formal_8gpu_mjlab_clean_eval \
  --data-path /data/xue/bfmzero/data/lafan_29dof.pkl \
  --device cuda:0 \
  --headless \
  --disable-dr \
  --disable-obs-noise \
  --save-mp4 \
  --motion-list 20 \
  --render-size 480
```

Outputs go to `<model-folder>/tracking_inference_mjlab/`.

## Goal Inference

Computes goal-conditioned `z`; optionally exports ONNX:

```bash
cd /home/xue/bfmzero-mjlab

CUDA_VISIBLE_DEVICES=0 \
uv run python -m humanoidverse.goal_inference_mjlab \
  --model-folder /data/xue/bfmzero-mjlab/runs/formal_8gpu_mjlab_clean_eval \
  --data-path /data/xue/bfmzero/data/lafan_29dof.pkl \
  --device cuda:0 \
  --headless \
  --disable-dr \
  --disable-obs-noise \
  --save-mp4 \
  --export-onnx
```

Outputs go to `<model-folder>/goal_inference_mjlab/`.

## Reward Inference

Computes reward-task `z` from one rank-local replay buffer shard:

```bash
cd /home/xue/bfmzero-mjlab

CUDA_VISIBLE_DEVICES=0 \
uv run python -m humanoidverse.reward_inference_mjlab \
  --model-folder /data/xue/bfmzero-mjlab/runs/formal_8gpu_mjlab_clean_eval \
  --data-path /data/xue/bfmzero/data/lafan_29dof.pkl \
  --device cuda:0 \
  --headless \
  --disable-dr \
  --disable-obs-noise \
  --buffer-rank 0 \
  --num-samples 150000 \
  --n-inferences 1 \
  --skip-rollouts \
  --export-onnx
```

Outputs go to `<model-folder>/reward_inference_mjlab/`.
