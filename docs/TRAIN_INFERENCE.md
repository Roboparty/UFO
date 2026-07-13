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
- `--update-z-every-step`: defaults to `100` for FB and `10` for TeCH.

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

## TeCH Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./run_train.sh \
  --agent tech \
  --gpu-ids all \
  --use-wandb \
  --wandb-run-name ufo_tech_8gpu
```

TeCH was previously exposed as the TLDR preset in early UFO versions. `--agent tldr` is kept as a deprecated compatibility alias for `--agent tech`.

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

When `--export-onnx` is enabled, `tracking_inference` exports a
robot-config-aware policy ONNX next to the checkpoint. The policy input split is
derived from the checkpoint model's `obs_space` and actor `input_filter`, and a
metadata JSON records the robot name, robot config path, XML path, controlled
joints, actor input dimensions, z dimension, actor observation dimension, and
output action dimension.

The exported ONNX is tied to the checkpoint's robot, action, and observation
dimensions. One checkpoint cannot be reused across different robots. The deploy
branch remains G1-only unless a robot-specific deploy configuration is created.

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

`goal_inference` accepts `--robot-config`, `--data-manifest`, `--dataset`, and
`--rebuild-motion-cache` with the same manifest behavior as tracking inference.
If no robot config is provided, it defaults to `configs/robots/g1_29dof.yaml`.
For G1, omitting `--goal-json` keeps the existing
`goal_frames_lafan29dof.json` fallback. For non-G1 robots, pass a goal JSON
generated for the selected robot; the G1 goal JSON is not shared across robot
morphologies.

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

`reward_inference` also accepts `--robot-config`, `--data-manifest`,
`--dataset`, and `--rebuild-motion-cache`. G1 keeps the full default reward task
set. For non-G1 robots, the first robot-config-aware path is limited to
robot-config-aware rollout/relabel setup and root/locomotion tasks such as
`move-ego-*` and `rotate-z-*`; arm, crouch, sit-on-ground, and other
G1-semantics tasks require robot-specific reward semantics and are rejected
early. The exported ONNX and reward/goal outputs remain tied to the checkpoint's
robot, action, and observation dimensions. The deploy branch remains G1-only
unless a robot-specific deploy config is created.
