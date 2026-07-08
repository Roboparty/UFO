# UFO

UFO is an unsupervised reinforcement learning framework for humanoid control, with training and inference tools for Unitree G1-style humanoid policies.

The codebase provides two training presets in one project:

- `fb`: the default forward-backward representation learning preset.
- `tldr`: the TLDR preset for temporal latent distance reward training.

## Install

Run commands from the repository root:

```bash
uv sync
```

If you use W&B logging, log in before launching multi-GPU training. Multi-process training should not depend on an interactive login prompt:

```bash
uv run wandb login
# or
export WANDB_API_KEY=your_wandb_api_key
```

## Training Defaults

Core defaults live in `humanoidverse/train.py` and can still be overridden from the command line:

- `--num-envs`: `1024` environments per GPU.
- `--num-env-steps`: `192000000` global environment steps.
- `--data-path`: `humanoidverse/data/lafan_29dof_10s-clipped.pkl`.
- `--work-dir`: `runs/ufo`.
- `--checkpoint-every-steps`: `3200000` global environment steps.
- `--buffer-size`: `5120000` transitions per GPU.

## Train FB

FB is the default agent, so the minimal 8-GPU command is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./run_train.sh \
  --agent fb \
  --gpu-ids all \
  --use-wandb \
  --wandb-run-name ufo_fb_8gpu
```

Override defaults only when needed:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
./run_train.sh \
  --agent fb \
  --gpu-ids all \
  --data-path path/to/motions.pkl \
  --work-dir runs/ufo_fb_custom \
  --num-envs 1024 \
  --num-env-steps 192000000
```

## Train TLDR

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

Outputs are written to `<model-folder>/tracking_inference/`.

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

Outputs are written to `<model-folder>/goal_inference/`.

## Reward Inference

Reward inference reads a rank-local replay buffer shard from the training run:

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

Outputs are written to `<model-folder>/reward_inference/`.
