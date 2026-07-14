<h1 align="center">
  UFO: A General <b>U</b>nsupervised Reinforcement Learning <b>F</b>ramework for Humanoid C<b>O</b>ntrol
</h1>

<p align="center">
  <a href="https://roboparty.github.io/UFO/"><img alt="Website" src="https://img.shields.io/badge/Website-roboparty.github.io%2FUFO-2563eb?style=for-the-badge" /></a>
  <a href="https://youtu.be/uJPcLdn9sNA"><img alt="Video" src="https://img.shields.io/badge/Video-Demo-7c3aed?style=for-the-badge" /></a>
  <a href="https://roboparty.github.io/UFO/assets/UFO.pdf"><img alt="PDF" src="https://img.shields.io/badge/PDF-Available-0f766e?style=for-the-badge" /></a>
</p>

<p align="center">
  <img src="./assets/UFO.png" alt="UFO" height="72" style="vertical-align: middle;" />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="./assets/rplab_logo.png" alt="ROBO PARTY LAB Logo" height="72" style="vertical-align: middle;" />
</p>


UFO is an unsupervised reinforcement learning framework for humanoid control. The
main branch provides MJLab training, robot-aware motion-data import, and policy
inference, with a curated and best-tested path for Unitree G1.

## Highlights

- FB and TeCH unsupervised RL presets.
- G1 humanoid training in MJLab.
- RobotState CSV and NPZ motion-data import.
- Manifest-based, source-weighted multi-source data mixing.
- Experimental `--robot-config` training initialization for bringing up new robots.
- Robot-config-aware tracking inference and video export.
- Robot-config-aware goal inference and limited reward inference bring-up paths.

## Current Support Matrix

| Capability | Status |
| --- | --- |
| G1 training | Supported and best tested |
| Motion data: RobotState CSV / NPZ / `ufo_pkl` | Supported |
| Multi-source data manifest | Supported |
| Tracking inference | Robot-config aware |
| Goal inference | Robot-config aware; non-G1 requires robot-specific goal JSON |
| Reward inference | G1 full default tasks; non-G1 limited to root/locomotion tasks |
| Deployment and teleoperation | Use the [`deploy` branch](https://github.com/Roboparty/UFO/tree/deploy) / UFO-Deploy runtime |
| Automatic motion retargeting | Not supported |
| Cross-robot shared-policy training | Not supported |

> [!NOTE]
> The `main` branch focuses on training, data import, and inference. Real-robot
> deployment and teleoperation runtime live in the `deploy` branch (UFO-Deploy).

## Install

Clone UFO first:

```bash
git clone https://github.com/Roboparty/UFO.git
cd UFO
```

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

Install the environment:

```bash
uv sync
```

## Motion Data

Large motion datasets are hosted separately so that `git clone` only downloads
the code. Download the default processed G1 LaFAN training data from the
[`xuewang/UFO-MotionData`](https://huggingface.co/datasets/xuewang/UFO-MotionData)
dataset:

```bash
bash scripts/download_data.sh g1_lafan
```



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

### TeCH Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./run_train.sh \
  --agent tech \
  --gpu-ids all \
  --num-envs 1024 \
  --num-env-steps 192000000 \
  --work-dir runs/ufo_tech_g1 \
  --data-path humanoidverse/data/lafan_29dof_10s-clipped.pkl \
  --update-z-every-step 10 \
  --buffer-size 5120000 \
  --use-wandb \
  --wandb-run-name ufo_tech_g1
```

TeCH was previously exposed as the TLDR preset in early UFO versions. `--agent tldr` is kept as a deprecated compatibility alias for `--agent tech`.

Core defaults live in `humanoidverse/train.py`. In particular, `--num-envs`
and `--buffer-size` are per GPU, while `--num-env-steps` is the global sample
budget.

## Bring Up a New Robot

This experimental path assumes you already have a MuJoCo XML and RobotState
motion data for the same robot. UFO does not automatically retarget motion from
another skeleton.

`robot_inspect --hydra-out` can generate XML-derived draft RobotTrainingSpec
and Hydra robot config files. If a matching URDF is available, pass `--urdf` to
enrich draft hardware limits, dynamics, semantic hints, and symmetry metadata;
MuJoCo XML still defines qpos/qvel/action layout and actuator order. See
[Robot-Config Training](docs/robot_config_training.md).

1. Generate robot YAML and Hydra config drafts from the XML, optionally assisted
   by a matching URDF:

   ```bash
   uv run python -m humanoidverse.tools.robot_inspect \
     --xml /path/to/robot.xml \
     --urdf /path/to/robot.urdf \
     --name my_robot \
     --out configs/robots/my_robot.yaml \
     --hydra-out humanoidverse/config/robot/my_robot/my_robot_auto.yaml
   ```

   Omit `--urdf` if you only have MJCF. URDF is auxiliary; it does not replace
   the MuJoCo XML as the training/inference source of truth.

2. Curate the generated YAML. Verify the base body, control-joint order, feet,
   hands, key bodies, initial state, controller, and actuator fields. The draft
   is not a finished robot configuration.

3. Build the full inference pickle, clipped training pickle, and data manifest:

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

   Headerless RobotState CSV files are accepted. Without a header, columns are
   interpreted as `root_pos` xyz, `root_quat` xyzw, then DOF positions in the
   robot XML/control-joint order; an optional leading `time` column is also
   accepted.

   Optional: run `humanoidverse.tools.data_inspect` first if you want to
   validate the CSV schema without building cache files.

4. Start with a smoke training run:

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

Outputs are written to `<model-folder>/tracking_inference/`. Tracking, goal,
and reward inference accept `--robot-config` and manifest-based inference data.
Goal inference defaults to the curated G1 goal JSON for G1 checkpoints; non-G1
goal inference requires a robot-specific `--goal-json`, because goal frames and
joint targets are not shared across robot morphologies. Reward inference keeps
the full default task set for G1. For non-G1 robots, the first robot-config-aware
path is limited to rollout/relabel setup and root/locomotion tasks such as
`move-ego-*` and `rotate-z-*` unless robot-specific reward semantics are added.

With `--export-onnx`, tracking inference exports a policy ONNX that is aware of
the selected robot config by deriving actor input dimensions from the loaded
checkpoint's `obs_space` and actor `input_filter`. A companion metadata JSON is
written next to the ONNX with the robot config, XML path, controlled joints,
actor input dimensions, z dimension, actor observation dimension, and output
action dimension. The exported ONNX is tied to that checkpoint's robot, action,
and observation dimensions; one checkpoint cannot be reused across different
robots. The deploy branch remains G1-only unless a robot-specific deploy config
is created.

## Documentation

- [Import Wizard](docs/import_wizard.md): RobotState schemas, inspection, and data building.
- [Robot-Config Training](docs/robot_config_training.md): experimental robot-aware training initialization.
- [Training and Inference](docs/TRAIN_INFERENCE.md): additional commands and runtime notes.
