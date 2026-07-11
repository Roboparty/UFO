# UFO: An **U**nsupervised Reinforcement Learning **F**ramework for Humanoid **CO**ntrol

<p align="center">
  <img src="./assets/rplab_logo.png" alt="ROBO PARTY LAB Logo" width="420" />
</p>

<p align="center">
  <img src="./assets/teleop.gif" alt="Teleoperation Demo" width="760" />
</p>

UFO is an unsupervised reinforcement learning framework for humanoid control. The
main branch provides MJLab training, robot-aware motion-data import, and policy
inference, with a curated and best-tested path for Unitree G1.

## Highlights

- FB and TLDR unsupervised RL presets.
- G1 humanoid training in MJLab.
- RobotState CSV and NPZ motion-data import.
- Manifest-based, source-weighted multi-source data mixing.
- Experimental `--robot-config` training initialization for bringing up new robots.
- Robot-config-aware tracking inference and video export.

## Current Support Matrix

| Capability | Status |
| --- | --- |
| G1 training | Supported and best tested |
| Motion data: RobotState CSV / NPZ / `ufo_pkl` | Supported |
| Multi-source data manifest | Supported |
| Tracking inference | Robot-config aware |
| Goal inference | G1-only |
| Reward inference | G1-only |
| Deployment and teleoperation | Use the [`deploy` branch](https://github.com/Xuewang01/UFO/tree/deploy) / UFO-Deploy runtime |
| Automatic motion retargeting | Not supported |
| Cross-robot shared-policy training | Not supported |

> [!NOTE]
> The `main` branch focuses on training, data import, and inference. Real-robot
> deployment and teleoperation runtime live in the `deploy` branch (UFO-Deploy).

## Install

Install [`uv`](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
```

Alternatively:

```bash
python -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"
```

Clone UFO and install its environment:

```bash
git clone https://github.com/Xuewang01/UFO.git
cd UFO
uv sync
```

## Motion Data

Large motion datasets are hosted separately so that `git clone` only downloads
the code. Download the default processed G1 LaFAN training data from the
[`xuewang/UFO-MotionData`](https://huggingface.co/datasets/xuewang/UFO-MotionData)
dataset:

```bash
bash scripts/download_data.sh g1_lafan
ls -lh humanoidverse/data/lafan_29dof_10s-clipped.pkl
```

The download script verifies the SHA256 checksum and places the approximately
205 MB file at the default training path.

For W&B logging, authenticate before starting a multi-process run:

```bash
uv run wandb login
# Or: export WANDB_API_KEY=your_wandb_api_key
```

## Quick Start

### G1 Smoke Test

Run this first to verify the environment, motion data, and short training loop:

```bash
./run_train.sh \
  --agent fb \
  --data-manifest configs/data/example_mix.yaml \
  --gpu-ids single \
  --smoke \
  --work-dir /tmp/ufo_smoke_g1
```

### Full FB Training on 8 GPUs

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./run_train.sh \
  --agent fb \
  --gpu-ids all \
  --num-envs 1024 \
  --num-env-steps 192000000 \
  --work-dir runs/ufo_fb_g1 \
  --data-path humanoidverse/data/lafan_29dof_10s-clipped.pkl \
  --update-z-every-step 100 \
  --buffer-size 5120000 \
  --use-wandb \
  --wandb-run-name ufo_fb_g1
```

### TLDR Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./run_train.sh \
  --agent tldr \
  --gpu-ids all \
  --num-envs 1024 \
  --num-env-steps 192000000 \
  --work-dir runs/ufo_tldr_g1 \
  --data-path humanoidverse/data/lafan_29dof_10s-clipped.pkl \
  --update-z-every-step 10 \
  --buffer-size 5120000 \
  --use-wandb \
  --wandb-run-name ufo_tldr_g1
```

Core defaults live in `humanoidverse/train.py`. In particular, `--num-envs`
and `--buffer-size` are per GPU, while `--num-env-steps` is the global sample
budget.

## Bring Up a New Robot

This experimental path assumes you already have a MuJoCo XML and RobotState
motion data for the same robot. UFO does not automatically retarget motion from
another skeleton.

1. Generate a robot YAML draft from the XML:

   ```bash
   uv run python -m humanoidverse.tools.robot_inspect \
     --xml /path/to/robot.xml \
     --name my_robot \
     --out configs/robots/my_robot.yaml
   ```

2. Curate the generated YAML. Verify the base body, control-joint order, feet,
   hands, key bodies, initial state, controller, and actuator fields. The draft
   is not a finished robot configuration.

3. Inspect the adapted RobotState data:

   ```bash
   uv run python -m humanoidverse.tools.data_inspect \
     --robot configs/robots/my_robot.yaml \
     --source "/path/to/motions/*.csv" \
     --format robot_state_csv \
     --fps 50
   ```

4. Build the full inference pickle, clipped training pickle, and data manifest:

   ```bash
   uv run python -m humanoidverse.tools.data_build \
     --robot configs/robots/my_robot.yaml \
     --source "/path/to/motions/*.csv" \
     --format robot_state_csv \
     --name my_motion \
     --fps 50 \
     --clip-seconds 10 \
     --out configs/data/my_motion_auto_build.yaml \
     --rebuild-cache
   ```

5. Start with a smoke training run:

   ```bash
   ./run_train.sh \
     --agent fb \
     --robot-config configs/robots/my_robot.yaml \
     --data-manifest configs/data/my_motion_auto_build.yaml \
     --gpu-ids single \
     --smoke \
     --work-dir /tmp/ufo_smoke_my_robot
   ```

See [Import Wizard](docs/import_wizard.md) for data schemas and import commands,
and [Robot-Config Training](docs/robot_config_training.md) for required training
fields, current constraints, and bring-up guidance. New robots may still require
environment and controller tuning before high-quality training.

## Tracking Inference

Use full motion sequences for inference, rather than clipped training data:

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python -m humanoidverse.tracking_inference \
  --model-folder runs/ufo_fb_g1 \
  --data-path /path/to/full_motions.pkl \
  --device cuda:0 \
  --headless \
  --save-mp4 \
  --motion-list 0
```

Outputs are written to `<model-folder>/tracking_inference/`. Goal inference and
reward inference remain G1-only in this release; they are not part of the
experimental new-robot bring-up path.

## Documentation

- [Import Wizard](docs/import_wizard.md): RobotState schemas, inspection, and data building.
- [Robot-Config Training](docs/robot_config_training.md): experimental robot-aware training initialization.
- [Training and Inference](docs/TRAIN_INFERENCE.md): additional commands and runtime notes.
