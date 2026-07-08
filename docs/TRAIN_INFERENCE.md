# UFO Setup, Training, and Inference

This document mirrors the repository quick start with a little more context for training and inference runs.

## Install

```bash
uv sync
```

For W&B logging, authenticate before launching multi-process training:

```bash
uv run wandb login
# or
export WANDB_API_KEY=your_wandb_api_key
```

## Defaults

The default training configuration is defined in `humanoidverse/train.py`:

- `--num-envs`: `1024` environments per GPU.
- `--num-env-steps`: `192000000` global environment steps.
- `--data-path`: `humanoidverse/data/lafan_29dof_10s-clipped.pkl`.
- `--work-dir`: `runs/ufo`.
- `--checkpoint-every-steps`: `3200000` global environment steps.
- `--buffer-size`: `5120000` transitions per GPU.

All of these can be overridden from the command line.

## FB Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./run_train.sh \
  --agent fb \
  --gpu-ids all \
  --use-wandb \
  --wandb-run-name ufo_fb_8gpu
```

## TLDR Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./run_train.sh \
  --agent tldr \
  --gpu-ids all \
  --use-wandb \
  --wandb-run-name ufo_tldr_8gpu
```

## Tracking Inference

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python -m humanoidverse.tracking_inference \
  --model-folder runs/ufo \
  --device cuda:0 \
  --headless \
  --disable-dr \
  --disable-obs-noise \
  --save-mp4 \
  --motion-list 20
```

## Goal Inference

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python -m humanoidverse.goal_inference \
  --model-folder runs/ufo \
  --device cuda:0 \
  --headless \
  --disable-dr \
  --disable-obs-noise \
  --save-mp4 \
  --export-onnx
```

## Reward Inference

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python -m humanoidverse.reward_inference \
  --model-folder runs/ufo \
  --device cuda:0 \
  --headless \
  --disable-dr \
  --disable-obs-noise \
  --buffer-rank 0 \
  --num-samples 150000 \
  --n-inferences 1 \
  --save-mp4 \
  --export-onnx
```
