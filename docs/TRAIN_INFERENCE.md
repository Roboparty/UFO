# UFO Setup, Training, and Inference

Detailed training and inference options. For installation and the main commands,
see the [README](../README.md).

## Install

```bash
uv sync --locked
```

For W&B logging, authenticate before launching multi-process training:

```bash
uv run wandb login
# or
export WANDB_API_KEY=your_wandb_api_key
```

## Defaults

| FB default | G1 | RP1 | RPO |
|---|---|---|---|
| Robot configuration | `configs/robots/g1_29dof.yaml` | `configs/robots/rp1_3.yaml` | `configs/robots/rpo.yaml` |
| Controlled joints | 29 | 24 | 23 |
| Actor / Forward / critic width | 1024 | 1024 | 1024 |
| Critic structure | Shared trunk | Shared trunk | Shared trunk |
| Update precision | FP32 with TF32 allowed | BF16 mixed precision; FP32 parameters and losses | BF16 mixed precision; FP32 parameters and losses |
| Global training steps | 192,000,000 | 384,000,000 | 768,000,000 |
| Seed | 4728 | 52 | 52 |
| Model save interval, global steps | 3,200,000 | 16,384,000 | 16,384,000 |

The RP1 FB baseline follows the original IsaacLab `model_45989` configuration;
migration between simulators does not guarantee matching results. Each robot
retains its control parameters, auxiliary rewards and domain randomization.
Both flat-ground slip penalties use horizontal foot velocity, with different
contact gates; both angular-velocity observation scales are `1.0`.

The CLI defaults below describe G1. RP1 and RPO override the training budget,
seed, model-save interval and FB precision in their robot YAML files. RPO TeCH
has not been validated.

The entrypoint builds defaults from `ufo/train.py`, `ufo/training/config.py` and the selected algorithm config:

- `--num-envs`: `1024` environments per GPU.
- `--num-env-steps`: `192000000` global environment steps.
- `--data-path`: `ufo/data/lafan_29dof_10s-clipped.pkl`.
- `--work-dir`: `runs/<robot>_<agent>` (for example, `runs/g1_fb`).
- `--checkpoint-every-steps`: model checkpoints every `3200000` global environment steps.
- `--buffer-every-steps`: complete rank-0 replay snapshots every `16000000` global environment steps.
- `--buffer-size`: `5120000` transitions per GPU.
- `--update-z-every-step`: defaults to `100` for FB and `10` for TeCH.

All of these can be overridden from the command line; both save intervals must be positive.

Normal completion saves a final model and rank-0 replay. To request a planned
stop with replay from every rank, run `touch RUN/SAVE_AND_STOP`, replacing `RUN`
with the work directory. All ranks save and stop at the next model/replay save
boundary. Resuming requires the original GPU count/world size. With that
requirement satisfied, training restores replay only when every rank snapshot
matches the model save ID and world size; otherwise all ranks warm up fresh replay.
Environment/RNG state is not saved. See [save and restore behavior](ARCHITECTURE.md#completion-planned-stopping-and-replay-restore).

Checkpoint saves are staged before replacing `checkpoint`; `checkpoint.previous`
retains the preceding complete snapshot. Training and inference fall back to it
if replacement was interrupted. Unchanged replay files share disk storage; a new
replay snapshot may temporarily require space for both versions.

Training-time tracking evaluation disables observation noise while retaining the
training physics and domain randomization. Its temporary episode limit covers
the entire reference motion, and unexpected resets fail evaluation rather than
entering the metrics. This changes evaluation conditions relative to older runs;
re-evaluate checkpoints under the same conditions before comparing EMD.

The legacy `randomize_torque_rfi`, `randomize_rfi_lim`, and
`randomize_ctrl_delay` options are unsupported and rejected when enabled.
RP1's motor delay remains configured through `training.actuator`.

During multi-GPU training, tracking evaluation runs on every rank. Motion IDs
are partitioned as `rank, rank + world_size, ...`; with 8 GPUs each rank evaluates
approximately one eighth of the motions. Each rank reuses its training
environment and processes its assigned motions in batches. Rank 0 gathers the
per-motion results, checks coverage, and computes the global mean/std before
writing CSV/W&B output. It then broadcasts sampling priorities to every rank.
Training resumes once all ranks finish evaluation; it does not overlap with
evaluation. Single-GPU evaluation still covers the full motion set.

`--smoke` and `--disable-eval-prioritization` disable evaluation and priority
updates. To test the distributed evaluation collectives without a simulator:

```bash
uv run --with pytest python -m pytest tests/test_distributed_tracking_eval.py
# Optional: exercise NCCL on 8 available GPUs instead of CPU/Gloo.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 UFO_TEST_DISTRIBUTED_BACKEND=nccl \
  uv run --with pytest python -m pytest tests/test_distributed_tracking_eval.py
```

## FB Training

FB defaults to width 1024 and six hidden layers with a shared critic trunk,
including for custom robots. G1, RP1 and RPO inherit these algorithm defaults. Use
`--network-width 2048` to override actor, Forward and critic width, or
`--no-shared-critic` for independent critics. See [critic topology](ARCHITECTURE.md#fb-critic-topology)
for the training implications. Use a separate `--work-dir` for each experiment.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./run_train.sh \
  --agent fb \
  --gpu-ids all \
  --use-wandb \
  --wandb-run-name ufo_fb_8gpu
```

## TeCH Training

TeCH has its own network preset and does not accept FB width/shared-critic options.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./run_train.sh \
  --agent tech \
  --gpu-ids all \
  --use-wandb \
  --wandb-run-name ufo_tech_8gpu
```


## Tracking Inference

Tracking, goal and reward videos render rollout qpos through the same MuJoCo
renderer and robot XML. Video frames are owned HWC uint8 RGB arrays; simulator
native rendering is not used by these exporters.

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python -m ufo.inference.tracking_inference \
  --model-folder runs/g1_fb \
  --device cuda:0 \
  --disable-dr \
  --disable-obs-noise \
  --save-mp4 \
  --motion-list 20
```

Motion IDs index the dataset actually loaded. The [full-sequence LaFAN index](#lafan-full-sequence-index)
below applies to `lafan_29dof.pkl`, not the default 862-clip training file.

When `--export-onnx` is enabled, `tracking_inference` exports a
robot-config-aware policy ONNX next to the checkpoint. The policy input split is
derived from the checkpoint model's `obs_space` and actor `input_filter`, and a
metadata JSON records the robot name, robot config path, XML path, controlled
joints, actor input dimensions, z dimension, actor observation dimension, and
output action dimension.

Current exports also write `<Model>.deploy.yaml` with the saved training
observation order/scales, history lengths, joint defaults, motor gains and action
scales. Keep the `.onnx`, `.meta.json` and `.deploy.yaml` together: metadata binds
the configuration to the ONNX hash. Tracking additionally writes
`backward_encoder.meta.json` for reference observation preprocessing. Reward z
files store NumPy arrays so deployment does not require PyTorch.

`history_oldest_first` in the metadata preserves the checkpoint's history order:
G1 uses newest-first and RP1 uses oldest-first. Export does not reorder policy
inputs. A deployment runtime must support and validate this order; the current
G1 deploy runtime rejects RP1's order.

The exported ONNX is tied to the checkpoint's robot, action, and observation
dimensions. One checkpoint cannot be reused across different robots. The deploy
branch remains G1-only unless a robot-specific deploy configuration is created.

Both policy and backward-encoder exporters live in `ufo/tools/export_onnx.py`.
The standalone backward-encoder command remains
`python -m ufo.tools.export_onnx`; use its `--help` for checkpoint,
output and verification options.

## Goal Inference

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python -m ufo.inference.goal_inference \
  --model-folder runs/g1_fb \
  --data-path ufo/data/lafan_29dof.pkl \
  --device cuda:0 \
  --disable-dr \
  --disable-obs-noise \
  --save-mp4 \
  --export-onnx
```

`goal_inference` accepts `--robot-config`, `--data-manifest`, `--dataset`, and
`--rebuild-motion-cache` with the same manifest behavior as tracking inference.
If no robot config is provided, inference reads it from the saved run, preserving training-time control and observation settings.
For G1, omitting `--goal-json` uses
`ufo/data/robots/g1/goal_frames_lafan29dof.json`. For non-G1 robots, pass a goal JSON
generated for the selected robot; the G1 goal JSON is not shared across robot
morphologies.

The bundled G1 JSON indexes the **full** `lafan_29dof.pkl`, not the training
clips saved in the run configuration. Always pass the matching `--data-path`.

For RP1, list the full dataset to choose a motion ID and name:

```bash
./run_inference.sh tracking --model-folder runs/rp1_fb \
  --data-path ufo/data/rp1_24dof_lafan_v7.pkl --list-motions
```

In the original RP1 LaFAN v7 data, ID 0 is `dance1_subject1`. Create a small
goal file (adjust the ID, exact name and frame indices if your data differs):

```bash
cat > /tmp/rp1_goals.json <<'JSON'
[{"motion_id": 0, "motion_name": "dance1_subject1", "frames": [0, 10, 20]}]
JSON
./run_inference.sh goal --model-folder runs/rp1_fb \
  --data-path ufo/data/rp1_24dof_lafan_v7.pkl \
  --goal-json /tmp/rp1_goals.json --save-mp4
```

Frames are zero-based indices in that motion. For TeCH, replace the run folder
with `runs/rp1_tech` (or `runs/g1_tech` for G1). RP1 tracking and FB reward use
the same commands as G1 with `--model-folder runs/rp1_fb`.

RPO FB uses the same commands with `runs/rpo_fb` and
`ufo/data/rpo_23dof_lafan_grounded.pkl`; use `--list-motions` before creating an
RPO-specific goal JSON.

## Short training check

`--smoke` runs at most 2048 steps, below the 10240-step warmup, so it checks
environment sampling and checkpoint saving but does not update weights.
To also check optimization and two-GPU synchronization:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./run_train.sh --robot g1 --agent fb --gpu-ids all \
  --num-envs 16 --num-env-steps 24576 --buffer-size 32768 \
  --num-agent-updates 1 --disable-eval-prioritization \
  --work-dir runs/g1_fb_check
```

Replace `g1` with `rp1` or `rpo`, or `fb` with `tech` for the validated G1/RP1
combinations, and use a fresh run folder for each experiment. This validation command skips tracking evaluation/prioritization
and produces short-test weights. Check `checkpoint/train_status.json` for
`optimizer_steps > 0` and `checkpoint/distributed_sync.json` for synchronized
model state; it is not a training-quality or FPS benchmark.

## Reward Inference

Reward inference requires FB. TeCH supports tracking and goal inference only.
Use `./run_inference.sh reward --model-folder RUN --list-tasks` to inspect supported tasks.

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python -m ufo.inference.reward_inference \
  --model-folder runs/g1_fb \
  --device cuda:0 \
  --disable-dr \
  --disable-obs-noise \
  --num-samples 150000 \
  --n-inferences 1 \
  --save-mp4 \
  --export-onnx
```

Reward inference reads the saved rank-0 replay; additional rank buffers are not required.
It reads the current nested `next` transition fields and calls
`model.reward_wr_inference`; flat `next_*` fields and cached `B` vectors are unsupported.

`reward_inference` also accepts `--robot-config`, `--data-manifest`,
`--dataset`, and `--rebuild-motion-cache`. G1 keeps the full default reward task
set. For non-G1 robots, the first robot-config-aware path is limited to
robot-config-aware rollout/relabel setup and root/locomotion tasks such as
`move-ego-*` and `rotate-z-*`; arm, crouch, sit-on-ground, and other
G1-semantics tasks require robot-specific reward semantics and are rejected
early. The exported ONNX and reward/goal outputs remain tied to the checkpoint's
robot, action, and observation dimensions. The deploy branch remains G1-only
unless a robot-specific deploy config is created.


## Fixed training execution

GPU training uses FP32 parameters/data with TF32 permitted, compiled network
forwards, fused Adam, manual 128 MiB gradient buckets and trajectory replay.
RP1 FB additionally enables BF16 mixed-precision update forwards through
`train.use_bf16_amp`; G1 FB leaves that flag disabled. For runtime ownership, see [execution and ownership](ARCHITECTURE.md#runtime-and-state-boundaries)
for details and [validation](ARCHITECTURE.md#validation-and-evidence) for checks
and their limits.

## LaFAN full-sequence index

The following 40 IDs apply only to `ufo/data/lafan_29dof.pkl`, with the original
full-sequence ordering. The default training dataset,
`ufo/data/lafan_29dof_10s-clipped.pkl`, contains 862 clips with a different index.
Custom imports or reordered datasets also have their own motion IDs; do not use
this table to interpret their evaluation rows or `--motion-list` values.

| id | name |
|---:|:-----|
| 0 | `fallAndGetUp1_subject4` |
| 1 | `dance1_subject3` |
| 2 | `dance2_subject4` |
| 3 | `fightAndSports1_subject1` |
| 4 | `jumps1_subject1` |
| 5 | `fight1_subject3` |
| 6 | `run1_subject5` |
| 7 | `dance1_subject2` |
| 8 | `walk1_subject5` |
| 9 | `run2_subject1` |
| 10 | `sprint1_subject4` |
| 11 | `jumps1_subject5` |
| 12 | `sprint1_subject2` |
| 13 | `walk2_subject1` |
| 14 | `walk3_subject4` |
| 15 | `walk3_subject2` |
| 16 | `fallAndGetUp2_subject3` |
| 17 | `dance1_subject1` |
| 18 | `fight1_subject5` |
| 19 | `fallAndGetUp1_subject5` |
| 20 | `fallAndGetUp2_subject2` |
| 21 | `walk3_subject3` |
| 22 | `jumps1_subject2` |
| 23 | `walk2_subject4` |
| 24 | `run1_subject2` |
| 25 | `walk1_subject1` |
| 26 | `run2_subject4` |
| 27 | `walk3_subject1` |
| 28 | `dance2_subject5` |
| 29 | `dance2_subject1` |
| 30 | `walk4_subject1` |
| 31 | `dance2_subject2` |
| 32 | `walk2_subject3` |
| 33 | `fallAndGetUp1_subject1` |
| 34 | `walk1_subject2` |
| 35 | `fallAndGetUp3_subject1` |
| 36 | `dance2_subject3` |
| 37 | `fight1_subject2` |
| 38 | `walk3_subject5` |
| 39 | `fightAndSports1_subject4` |


## Unified entrypoints and goal identity

Use `./run_train.sh --robot g1|rp1|rpo --agent fb|tech` and
`./run_inference.sh tracking|reward|goal --model-folder RUN`.
Robot YAML remains available through `--robot-config` for custom robots.
Tracking accepts `--list-motions`, `--motion-name NAME ...` and `--motion-list ID ...`.
Names must be exact dataset keys or unambiguous filename stems.

Goal files select frames from the loaded dataset. Inference checks motion ID/name
agreement. To also bind a file to a robot and joint order, use this envelope:

```json
{
  "robot_name": "rp1_24dof",
  "control_joint_names": ["replace_with_all_control_joints_in_order"],
  "goals": [{"motion_id": 1, "motion_name": "dance1_subject2", "frames": [100]}]
}
```

Replace joint names and motion selectors with the actual robot/dataset values.
The previous list-of-goals format is still accepted, but carries no robot identity.
Do not use the G1 default goal list with clipped training data or RP1 data.

### Joint-position MAE tools

`python -m ufo.tools.evaluate_mae tracking --model-folder RUN --data-path MOTIONS`
uses the same latent calculation as tracking inference for FB and TeCH. Evaluation
always runs headless with domain randomization and observation noise disabled;
the redundant `--agent`, `--headless`, `--disable-dr` and `--disable-obs-noise` flags
have been removed from this tool. The JSON report records `model_type`.

`python -m ufo.tools.evaluate_mae goal --model-folder RUN --goal-json GOALS`
validates robot metadata when present, motion IDs and motion names. Negative or
out-of-range goal frames are skipped, as in goal inference.

All inference rollouts run offscreen. Use `--save-mp4` for videos; the unused
`--headless` option has been removed from inference and goal MAE commands.
Inference loads omit training target networks while strictly checking the remaining
weights. ONNX export copies only its actor/encoder and observation normalizer.
Reward inference accepts the current trajectory replay format, including rank-zero
snapshots; older flat replay buffers are no longer supported.

The standalone helpers are `ufo.tools.motion_data` (formerly `ufo.tools.data`)
and `ufo.tools.evaluate_mae` (formerly `ufo.tools.evaluate`). Use the new module
names; the training and inference shell entrypoints are unchanged.
