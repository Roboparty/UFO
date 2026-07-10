# UFO: An **U**nsupervised Reinforcement Learning **F**ramework for Humanoid **CO**ntrol

<p align="center">
  <img src="./assets/rplab_logo.png" alt="ROBO PARTY LAB Logo" width="420" />
</p>


UFO is an unsupervised reinforcement learning framework for humanoid control, with training, motion-data import, and inference tools for MJLab-based humanoid policies.


## Install

Run commands from the repository root:

```bash
git clone git@github.com:Xuewang01/UFO.git
cd UFO
python show_help.py
git clone https://github.com/Xuewang01/UFO.git
cd BFMZero-mjlab
uv sync
```

If you use W&B logging, log in before launching multi-GPU training. Multi-process
training should not depend on an interactive login prompt:

```bash
uv run wandb login
# or
export WANDB_API_KEY=your_wandb_api_key
```

## Training Defaults

Core defaults live in `humanoidverse/train.py` and can still be overridden from
the command line:

- `--num-envs`: `1024` environments per GPU.
- `--num-env-steps`: `192000000` global environment steps.
- `--data-path`: `humanoidverse/data/lafan_29dof_10s-clipped.pkl`.
- `--work-dir`: `runs/ufo`.
- `--checkpoint-every-steps`: `3200000` global environment steps.
- `--buffer-size`: `5120000` transitions per GPU.
- `update_z_every_step`: FB uses `100` by default; TLDR uses `10`.
- `num_agent_updates`: FB uses `16`; TLDR uses `128`.

Changing `--num-envs` changes how often optimizer updates happen relative to
collected samples unless you also override `--num-agent-updates`.

## Data Import and Current Scope

UFO supports RobotState motion data import through manifests and the import
wizard tools. The public motion formats are:

- `ufo_pkl`: already processed UFO motion dictionaries.
- `robot_state_csv`: robot-state CSV files interpreted with a robot YAML config.
- `robot_state_npz`: robot-state NPZ files interpreted with a robot YAML config.

The recommended raw schema is:

- `root_pos`: `[T, 3]`, world root position.
- `root_quat`: `[T, 4]`, root quaternion in `x, y, z, w` order.
- `dof_pos`: `[T, N]`, robot control-joint positions.
- `fps` or `time`: sampling rate information.

CSV and NPZ readers can build full UFO motion pickles plus near-10-second
training clips. Use the near10s clips for training and the full data for
tracking inference.

This release is centered on the MJLab training environment and the curated G1
path is the best-tested path. The experimental `--robot-config` path makes
training initialization robot-aware, but it does not provide retargeting, does
not guarantee arbitrary robots train without tuning, and does not implement
cross-robot shared-policy training.

## Quick Start: Existing G1 Data

The shortest smoke test uses the default G1 config and bundled example motion
manifest:

```bash
./run_train.sh \
  --agent fb \
  --data-manifest configs/data/example_mix.yaml \
  --gpu-ids single \
  --smoke \
  --work-dir /tmp/ufo_smoke_g1
```

A typical 8-GPU G1 training run is:

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

## Quick Start: New Robot and New Motion Data

This path is for users who already have a MuJoCo XML and motion data adapted to
that robot. UFO does not retarget motions in this step. Your motion data must
already use the same robot, joint semantics, coordinate convention, and control
joint set as the robot YAML.

### 1. Inspect the MuJoCo XML

Generate a draft robot YAML:

```bash
uv run python -m humanoidverse.tools.robot_inspect \
  --xml /path/to/robot.xml \
  --name my_robot \
  --out configs/robots/my_robot.yaml
```

Read the generated YAML before training. At minimum, check or fill:

- `xml_path`: path to the MuJoCo XML.
- `base_body`: root/base body used by the floating base.
- `control_joint_names`: actuated joints, in policy/action order.
- `feet`, `hands`, `key_bodies`: semantic bodies used by observations,
  rewards, and diagnostics.
- `training.hydra_robot`: current base Hydra robot config used to compose the
  MJLab training environment.
- `training.semantics.contact_bodies`: at least two foot/contact bodies for the
  current biped reward implementation.
- `training.semantics.torso_name`: body used by domain randomization and torso
  diagnostics.
- `training.init_state.pos`, `training.init_state.rot`,
  `training.init_state.default_joint_angles`.
- `training.control`: action scale, clipping, stiffness, damping, effort limits,
  and velocity limits.
- `training.actuator`: use `source: yaml` for non-G1 robots and provide actuator
  values for every control joint.

`robot_inspect` is a draft generator, not a final curated config. The automatic
guesses for feet, hands, and key bodies are heuristics. For formal experiments,
curate these fields manually.

### 2. Prepare RobotState CSV or NPZ

For CSV, the default column names are:

```text
time,
root_pos_x,root_pos_y,root_pos_z,
root_quat_x,root_quat_y,root_quat_z,root_quat_w,
<one column per control joint name>
```

By default, CSV joint columns are matched by `control_joint_names`. For example,
if the robot YAML contains:

```yaml
control_joint_names:
  - left_hip_pitch_joint
  - left_knee_joint
```

then the CSV should contain columns named:

```text
left_hip_pitch_joint,left_knee_joint
```

CSV alternatives:

- Set `columns.dof_pos: xml_order` in the manifest and provide columns
  `dof_0`, `dof_1`, ..., in `control_joint_names` order.
- Set `columns.dof_pos` to an explicit list of column names in the manifest.
- If you do not pass `--fps`, include a strictly increasing `time` column.

For NPZ, provide arrays:

```text
root_pos   float32 [T, 3]
root_quat  float32 [T, 4]   # x, y, z, w
dof_pos    float32 [T, N]
fps        scalar, or time float [T]
joint_names optional string [N]
motion_key optional string
```

If `joint_names` is present, UFO reorders `dof_pos` into the robot YAML
`control_joint_names` order. If it is absent, `dof_pos` is assumed to already be
in `control_joint_names` order.

### 3. Inspect the Motion Data

CSV example:

```bash
uv run python -m humanoidverse.tools.data_inspect \
  --robot configs/robots/my_robot.yaml \
  --source "/path/to/motions/*.csv" \
  --format robot_state_csv \
  --fps 50
```

NPZ example:

```bash
uv run python -m humanoidverse.tools.data_inspect \
  --robot configs/robots/my_robot.yaml \
  --source "/path/to/motions/*.npz" \
  --format robot_state_npz \
  --fps 50
```

This catches common issues before training:

- missing root or joint columns;
- wrong `dof_pos` width;
- non-finite values;
- duplicate or missing joint names;
- non-increasing `time`;
- robot/data joint-order mismatch.

### 4. Build a Data Manifest and Motion Cache

Build full UFO pkl plus near10s train pkl from raw RobotState CSV:

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

The generated manifest records the robot config and data source. Auto-build
datasets produce:

- a full pkl for inference;
- a near10s train pkl for training.

For an already processed UFO pkl, a minimal manifest looks like:

```yaml
robot_config: configs/robots/g1_29dof.yaml
datasets:
  - name: lafan
    format: ufo_pkl
    train_path: humanoidverse/data/lafan_29dof_10s-clipped.pkl
    inference_path: /path/to/lafan_full.pkl
    weight: 1.0
```

For weighted multi-source training:

```yaml
robot_config: configs/robots/g1_29dof.yaml
datasets:
  - name: lafan
    format: ufo_pkl
    train_path: /data/motions/lafan_near10s.pkl
    inference_path: /data/motions/lafan_full.pkl
    weight: 0.95
  - name: cartwheel
    format: ufo_pkl
    train_path: /data/motions/cartwheel_near10s.pkl
    inference_path: /data/motions/cartwheel_full.pkl
    weight: 0.05
```

Weights are source-level sampling probabilities and are normalized internally.

### 5. Run a Smoke Training Job

Always start with a single-GPU smoke before launching a full training run:

```bash
./run_train.sh \
  --agent fb \
  --robot-config configs/robots/my_robot.yaml \
  --data-manifest configs/data/my_motion_auto_build.yaml \
  --gpu-ids single \
  --smoke \
  --work-dir /tmp/ufo_smoke_my_robot
```

The smoke should at least build the MJLab environment, load MotionLib, allocate
buffers, reset the environment, and enter the short training loop. If this fails,
fix the robot YAML or data manifest before running multi-GPU training.

### 6. Launch Training

Once the smoke passes:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./run_train.sh \
  --agent fb \
  --robot-config configs/robots/my_robot.yaml \
  --data-manifest configs/data/my_motion_auto_build.yaml \
  --gpu-ids all \
  --num-envs 1024 \
  --num-env-steps 192000000 \
  --work-dir runs/my_robot_fb \
  --update-z-every-step 100 \
  --buffer-size 5120000 \
  --use-wandb \
  --wandb-run-name my_robot_fb
```

For TLDR:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./run_train.sh \
  --agent tldr \
  --robot-config configs/robots/my_robot.yaml \
  --data-manifest configs/data/my_motion_auto_build.yaml \
  --gpu-ids all \
  --work-dir runs/my_robot_tldr \
  --use-wandb \
  --wandb-run-name my_robot_tldr
```

### 7. Tracking Inference with the Full Motion Data

Use full motions for tracking inference, not near10s training clips.

Manifest dataset example:

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python -m humanoidverse.tracking_inference \
  --model-folder runs/my_robot_fb \
  --data-manifest configs/data/my_motion_auto_build.yaml \
  --dataset my_motion \
  --device cuda:0 \
  --headless \
  --save-mp4 \
  --motion-list 0 1 2 3 4
```

Direct pkl example:

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python -m humanoidverse.tracking_inference \
  --model-folder runs/my_robot_fb \
  --data-path /path/to/full_motions.pkl \
  --device cuda:0 \
  --headless \
  --save-mp4 \
  --motion-list 0
```

Outputs are written to `<model-folder>/tracking_inference/`.

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

Goal inference is currently G1-only and expects a G1 goal JSON file. If your
checkout does not include the default `goal_frames_lafan29dof.json`, pass it
explicitly with `--goal-json`.

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

Reward inference reads a rank-local replay buffer shard from the training run.
It is currently G1-oriented and is not part of the experimental new-robot
bring-up path.

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

## More Documentation

- `docs/import_wizard.md`: import wizard command details.
- `docs/robot_config_training.md`: experimental `--robot-config` training path.
- `docs/TRAIN_INFERENCE.md`: additional training and inference notes.
