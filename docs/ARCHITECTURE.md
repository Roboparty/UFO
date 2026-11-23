# Code layout

The Python package is `ufo`. Training starts with `python -m ufo.train` or
`./run_train.sh`. FB-CPR-Aux (`--agent fb`) and TeCH (`--agent tech`) share the
mjlab environment and training runtime.

This document covers code ownership, configuration, training and checkpoint restoration.
For commands, see [Training and inference](TRAIN_INFERENCE.md); for new robots and
data, see [Robot adaptation](ROBOT_ADAPTATION.md).

| Directory | Responsibility |
| --- | --- |
| `ufo/agents/` | Networks, losses, optimizers, replay buffers, algorithm configs and execution helpers |
| `ufo/training/` | Launch schema, rollout/update scheduling, checkpoints, distributed operations, diagnostics and evaluation |
| `ufo/envs/` | Task resolution, mjlab scene construction, UFO task state/reset, observations/rewards and vector API |
| `ufo/motion/` | Motion import/cache, robot FK, motion sampling, expert trajectories and reference observations |
| `ufo/inference/` | Tracking, goal/reward inference and offline reward relabeling |
| `ufo/tools/` | Robot configuration loading, XML/URDF inspection and drafts, data preparation, ONNX export and standalone evaluation commands |
| `ufo/math.py` | Shared tensor, quaternion and rotation operations |
| `configs/` | Environment defaults, robot YAML and dataset manifests |

## Main files

```text
ufo/
  train.py                       # parse CLI, select builder, assemble launch config
  math.py                        # tensor, quaternion and rotation operations
  training/
    config.py                    # TrainConfig and shared save defaults
    trainer.py                   # prepare → save/evaluate → collect → update → log
    distributed.py               # seeding, gradient buckets and distributed operations
    logging.py                   # CSV output, finite checks and contact-force metrics
    evaluation.py                # tracking evaluation sharded across ranks
  agents/
    fb_cpr_aux/{config,model,agent}.py
    tech/{config,model,agent}.py
    common/
      base.py                    # shared configuration and model bases
      presets.py                 # network, normalizer and auxiliary reward presets
      networks.py                # network layers, constructors and tensor batch sizing
      inputs.py                  # observation filtering and normalization
      execution.py               # compile/sync, actor isolation, Adam, gradient penalty
      checkpoint.py              # model/agent I/O and observation-space serialization
      buffers/
        trajectory.py            # motion trajectories and contiguous replay sampling
        storage.py               # transition storage, serialization and latent ZBuffer
  envs/
    config.py                    # task/robot resolution, MJLabEnvConfig and physics scene
    actuators.py                 # G1 calibration and YAML motor parameters/delay
    core.py                      # task state, motion sampling, reset and action mapping
    observations.py              # history buffers, coordinate transforms and policy inputs
    rewards.py                   # auxiliary reward kernels and weighted environment reward
    vector_env.py                # vector API and terminal transitions
  motion/
    library.py                   # MotionLib: load, mix, sample and interpolate motions
    kinematics.py                # robot forward kinematics from motion states
    reference.py                 # shared reference observations and expert trajectories
    manifest.py                  # dataset manifests, source mixing and cache orchestration
    formats.py                   # motion readers, conversion, validation and clipping
  inference/
    runtime.py                     # shared config loading, reset targets and rendering
    tracking_inference.py          # motion tracking, latent sequence and video export
    goal_inference.py                      # infer latent goals from robot poses
    reward_inference.py                    # reward tasks, replay relabeling and reward inference
  tools/
    motion_data.py               # inspect data and build manifests/caches
    robot_config.py              # robot XML/YAML loading, validation and training metadata
    robot_inspect.py              # robot inspection and XML/URDF configuration drafts
    evaluate_mae.py              # tracking/goal joint-position MAE
    robot_setup.py               # compose robot draft and data preparation
    rp1_motion_import.py          # convert RP1 BeyondMimic NPZ to UFO
    rpo_motion_import.py          # convert the RPO NPZ dataset to UFO
    export_onnx.py                # policy and backward encoder export
  data/                          # robot assets and local motion data
    robots/g1/
      g1_29dof.xml                # G1 physics and visual model
      goal_frames_lafan29dof.json # default G1 goal-frame selections
      meshes/                    # 35 STL files referenced by the model
    robots/rpo/                  # RPO MJCF, meshes and hardware license
```

`training/config.py` declares the supported `AgentConfig` union. `train.py` calls
`build_fb_agent` or `build_tech_agent` directly; each algorithm's `config.py`
owns its model/train schemas and runtime preset. There is no agent-selection
wrapper between the entrypoint and these builders.

Both models directly inherit `BaseModel`; both agents use `AgentExecutionMixin`
for execution helpers rather than inheriting another algorithm. Architecture,
losses and update order remain in each algorithm's `model.py` and `agent.py`.
Each agent's `optimizers` mapping supplies checkpoint save/load and distributed
state broadcast; no separate optimizer-name list is maintained.
Actor and Forward/critic networks use residual layers; Forward/critic ensembles
use the batched path.
G1/RP1/RPO FB defaults to a shared critic trunk; `--no-shared-critic` selects independent critics. TeCH retains its
separate auxiliary critic. The FB-only shared critic lives in
`agents/fb_cpr_aux/model.py`. Numerical hyperparameters remain configurable.

`agents/common/` contains the modules shared by FB and TeCH.
`agents/common/buffers/` owns trajectory replay and transition storage; the
small latent `ZBuffer` shares `storage.py`. Model loading, saving and
observation-space JSON conversion use `agents/common/checkpoint.py`.
`ExpertTrajectoryBuffer` holds offline motion demonstrations; `ReplayBuffer`
holds online transitions. Renaming the classes does not invalidate existing replay snapshots.
Replay HDF5 keys use the fixed `-` separator. Existing snapshots recording that
default remain readable; other separator values are rejected.
`tree_get_batch_size` lives with networks, and `gradient_penalty_wgan` with
execution helpers. Run-directory naming belongs to `train.py`,
`EveryNStepsChecker` to `training/trainer.py`, and `set_seed_everywhere` to
`training/distributed.py`. CSV output and finite checks share `training/logging.py`.

`TrainConfig.log_every_steps` counts global environment steps, as do checkpoint
and evaluation intervals. Runtime Torch/Warp caches are configured once by
`train.py` before those libraries are imported. `run_train.sh` selects
`UFO_CACHE_DIR` and sets the uv, Python-bytecode and temporary directories needed
before Python starts.

`motion/formats.py` reads, validates and converts source motion data;
`motion/manifest.py` coordinates dataset manifests and caches. `motion/library.py`
consumes the normalized motions. `motion/reference.py` builds expert buffers and
shares observation construction with training evaluation and inference. Goal
inference and goal MAE use its `get_goal_observation` to map dataset motion IDs
to the local slots of the loaded motion subset, including when all motions are
already loaded. `tools/robot_config.py`
keeps `RobotSpec`, XML parsing, robot validation and training metadata together.
Training, inference and motion import use this module directly; configuration
draft generation stays in `tools/robot_inspect.py`.
`inference/runtime.py` owns shared configuration, initialization and rendering helpers.
Tracking, goal and reward entrypoints import it directly without importing each other.
`tools/export_onnx.py` owns policy and backward-encoder ONNX export; the standalone
`python -m ufo.tools.export_onnx` command exports the backward encoder.
All inference modes load and validate robot metadata once, then reuse the resolved
environment configuration and robot specification.

Reference `state` and `privileged_state` use the simulator's `scale_observation`
for joint positions, joint velocities, projected gravity, local base angular
velocity and `max_local_self`. Reference inputs never add observation noise.
The separate reference pose/velocity dictionary stays in physical units for
reset, rendering and metrics; observation scales do not modify motion data.
Tracking CLI, MAE and training evaluation all call `model.tracking_inference`,
which projects each frame's encoding independently, without temporal averaging.
Observation-noise curriculum settings use `add_noise_curriculum` and
`noise_curriculum_*`; these replace the misspelled switch and `soft_dof_pos_curriculum_*` names.

## FB critic topology

Independent critics have separate discriminator-reward Q and auxiliary-reward Q
networks, targets and optimizers. `--shared-critic` shares their embeddings and
residual trunk, with separate reward heads and one optimizer. Both losses update
the shared trunk; the discriminator itself remains a separate network.

The shared path retains independent target-action draws and target forwards for
the two rewards, then soft-updates target weights once per agent update. Actor
updates freeze critic parameters while retaining gradients through the action.
Checkpoint configuration records the topology and numerical dimensions; resuming
recreates them. Changing topology requires a fresh run.

## Runtime and state boundaries

G1 GPU execution uses FP32 parameters/data with TF32 permitted
(`torch.set_float32_matmul_precision("high")`), compiled network forwards,
fused Adam and manual asynchronous 128 MiB gradient buckets. Losses and
discriminator gradient penalties remain eager. CPU validation uses native
Adam and eager forwards. RP1 and RPO FB enable BF16 autocast through their robot baselines;
parameters remain FP32. See the [precision defaults](TRAIN_INFERENCE.md#fixed-training-execution).

Policy, simulator and replay exchange device tensors. Replay owns copied
storage; ordered observations are snapshots taken before the simulator advances.
Replay stores observations, actions, latents, termination flags and auxiliary
rewards. Training batches omit unused current-step termination flags; next-step
termination still controls bootstrapping. Environment `reward` and `step_count`
are not stored. Only FB retains `qpos/qvel` for reward inference. Restoring a
training replay removes the same unused fields without changing trajectory boundaries.
TeCH reuses one target actor distribution across its two critic updates, with
independent action samples at the original points in the update order.
Gradients synchronize during optimizer stages, floating model buffers after
updates, and accumulated metrics when logged. Tracking evaluation shares the
training environment, shards motion IDs across ranks and aggregates on rank 0.
The evaluator requires this supplied environment; trainer owns its lifecycle.
`training/logging.py:torso_contact_force_metrics` calculates logging metrics without
holding environment state; `_log_training` supplies its enable flag and threshold.

mjlab owns physics tensors, sensors, event timers and episode counters.
`MJLabCore` owns motion sampling, action/history state, observation snapshots
and auxiliary rewards. `episode_length_buf` refers to mjlab's counter rather
than a copied counter. `reset_idx` is the sole UFO task-reset entry: public
mjlab reset, motion-state write, action/history clear, then state refresh.
Independent mjlab auto-reset is rejected because it would reset physics before
UFO collects terminal transition rewards. UFO does not edit mjlab's private
pending-reset flags.

`core.py` retains previous actions, normalizes/reorders the new action, steps
mjlab, refreshes task state, calculates rewards and updates curriculum before
resetting completed worlds. `observations.py` defines `ObservationHistory` and the
observation kernels; the core owns each history instance and controls its reset.
Reward kernels remain stateless. Observation construction reads previous
history before inserting current scaled/noisy fields. Indexed joint/body state
is already a snapshot; direct root-state reads retain ownership-protecting
copies. Replay and public observation/history outputs must remain valid after
the next step or reset.

Contact forces use **force acting on the robot body**: upward ground reaction
under a planted foot. With UFO's primary-body sensor matching, mjlab reports
the force the primary body exerts on its contact partner, so `_read_contact_forces`
negates the full XYZ vector. This fixes the positive-Z contact gate in
`penalty_feet_ori`; slip and undesired-contact penalties use magnitudes/absolute
values. Changing this convention changes reward behavior, not merely logging.

`tools/robot_config.py` reads the robot YAML once into the training snapshot.
Task defaults come from `configs/env.yaml`; robot-specific control,
contact semantics, joint limits, motion extensions and task overrides stay in
the robot YAML. This supports custom robots without a second simulator adapter.
Motion loading uses one concrete `MotionLib` and the robot FK parser; source
mixing, active sampling priorities, interpolation and `history_actor` remain.
Robot XML, robot-config and manifest paths share `tools/robot_config.py:resolve_path`,
which checks the configuration directory before the current working directory.

## Checkpoint format

Current FB-CPR-Aux/TeCH checkpoints use JSON configuration/build arguments,
safetensors weights, optimizer state and explicit training counters. TeCH also
saves its dual parameter and goal-encoder training state. Missing required
weights or optimizer stages are errors. `load_target_networks=False` omits target
networks for inference while
validating its required weights. Retired fields, old pickle build arguments,
old algorithm names and historical configuration migrations are unsupported.

## Training flow

Start with `ufo/train.py:parse_args` and `run_train`, then
`ufo/training/trainer.py:Trainer`:

1. `parse_args` resolves CLI defaults, optional dataset manifests and the robot config path.
2. `run_train` selects device/rank, initializes distributed execution and calls `build_mjlab_config`.
3. `build_mjlab_config` calls the selected algorithm builder and assembles `TrainConfig` with environment/run settings.
4. `TrainConfig.build` creates `Trainer`: build the environment, create or restore the agent, initialize execution/evaluation/logging and write effective configuration.
5. `Trainer.train` owns scheduling; its stage implementations follow it in the same file.

The stages are `_prepare_replay_buffers`, `_evaluate_training`,
`_collect_transition`, `_update_agent`, `_log_training`, `eval` and `save`.
They respectively prepare expert/replay data, evaluate/reset/update priorities,
collect transitions, update/check/synchronize the agent, log metrics and perform
evaluation/checkpoint I/O. Finite checks and `torso_contact_force_metrics` live
in `ufo/training/logging.py`; `_log_training` passes the contact-metric enable
flag and threshold rather than maintaining a second copy of task state.

## Configuration ownership and precedence

The default launch path is:

```text
CLI / data manifest
  → train.py:build_mjlab_config + selected algorithm builder
  → training/config.py:TrainConfig
  → envs/config.py:MJLabEnvConfig.build → resolve_task_config + native mjlab scene
```

| Owner | Parameters and behavior |
| --- | --- |
| `ufo/train.py` | CLI defaults for dataset, environment count, budget, replay capacity, work directory and seed; supported CLI overrides; device/rank and W&B settings |
| `ufo/agents/fb_cpr_aux/config.py`, `ufo/agents/tech/config.py` | Algorithm model/train schemas and builders, latent update defaults and `TRAIN_RUNTIME` presets for logging/update/seed/evaluation intervals and updates per trigger |
| `ufo/agents/common/presets.py` | Shared network architectures, observation normalizers and auxiliary reward defaults used by the algorithm schemas |
| `ufo/training/config.py` | Supported FB/TeCH union, `TrainConfig` validation and shared positive model/replay save-interval defaults |
| `configs/robots/*.yaml`, `ufo/tools/robot_config.py` | Robot XML, joint/body order, controls, initialization, actuator parameters, contact semantics, motion extensions and robot-specific task overrides |
| `configs/env.yaml`, `ufo/envs/config.py` | Shared task defaults, robot/environment resolution and native mjlab scene construction |

`TrainConfig` validates the resolved launch settings. Work directory, seed,
environment count, step budget, replay size/device and logging/update/seed/evaluation
intervals are required fields. `train.py` resolves the CLI values and the selected
algorithm's `TRAIN_RUNTIME` supplies scheduling defaults (16 updates per trigger
for FB, 128 for TeCH). The CLI defaults the work directory to `runs/<robot>_<agent>`.
Direct Python callers should use `build_mjlab_config` or supply the resolved fields;
the schema does not generate a second run directory or alternate schedule.

Algorithm schemas contain the actual model and optimizer defaults. Their builders
resolve robot baselines and supported overrides; they do not define a second preset. Pydantic copies
nested defaults so changing one configuration cannot mutate another run.

FB owns the 1024-wide, six-hidden-layer forward/actor/critic defaults and enables
the shared critic in `agents/fb_cpr_aux/config.py`. G1 and RP1 inherit this topology;
TeCH retains its 2048-wide networks. `configs/env.yaml` owns angular-velocity
observation scaling, now 1.0 for all robots unless explicitly overridden.
Custom robot configs that need the previous defaults must explicitly set FB
widths to 2048, `shared_critic: false`, and/or
`training.task_overrides.obs.obs_scales.base_ang_vel: 0.25` as appropriate.

Environment `rewards.reward_scales` weights the scalar reward returned by the
environment (including its configured penalty curriculum). Agents instead weight
raw replay `aux_rewards` using `aux_rewards_scaling` for the auxiliary critic.
These settings have different consumers and may intentionally differ; neither
overrides the other.

Environment configs store dataset paths in `motion_data_path`, for either one
dataset or a list. Run JSON must use `motion_data_path`; the retired `lafan_tail_path` alias is rejected.

`build_fb_agent` is the only FB configuration resolver: it merges algorithm defaults
and launch values with `training.fb_agent_overrides`, applies explicit width/critic
choices, scales all six resolved learning rates once, and validates the result.
The existing precedence is preserved: robot overrides can replace launch values for
`clip_grad_norm`, `update_z_every_step` and `model.device`; explicit width and critic
choices take precedence over the robot baseline. The optional cartwheel reward
preset rejects a conflicting explicit robot reward baseline.

`build_mjlab_config` combines runtime scheduling with the supported robot schedule
overrides before constructing `TrainConfig`. `--num-agent-updates` overrides the
algorithm schedule. TeCH retains its own learning rates and ignores the FB
learning-rate scale/clipping options.

`tools/robot_config.py:select_robot_config` uses explicit `--robot-config`, otherwise the data manifest's
robot config, otherwise `configs/robots/g1_29dof.yaml`. When both CLI and manifest
provide a robot, `assert_robot_configs_compatible` checks matching identity/XML
before using the CLI config. Manifest loading produces resolved dataset paths
and normalized source weights; it cannot be combined with `--data-path`.

`load_robot_training_spec` reads the robot YAML once and snapshots its metadata
into the environment config. Task resolution starts with `env.yaml`, merges
`training.task_overrides`, then applies the selected robot metadata and environment
flags/data paths. FB's cartwheel-safe option reads both the auxiliary and task
reward settings from `CARTWHEEL_SAFE_REWARDS` in `agents/fb_cpr_aux/config.py`
before task resolution. Noise/DR disable flags apply after the
merge, so task overrides cannot re-enable them. Unknown task-override keys are
rejected. Robot configuration remains extensible through the same YAML path;
motion data must already be adapted to the chosen robot.

Unused legacy `terrain`, `rewards.reward_limit.reward_limits_curriculum`,
`termination_scales`, `termination_probality` and `termination_curriculum`
sections have been removed. External robot YAML files must remove overrides
for these sections; they previously had no effect and now fail strict validation.

## Scheduling and save defaults

Default model saves use `--checkpoint-every-steps 3200000`; complete rank-0
replay snapshots use `--buffer-every-steps 16000000`. Both positive defaults
come from `ufo/training/config.py`. Latent updates default to 100 for FB and
10 for TeCH, owned by their algorithm config modules. Explicit values take
precedence. Retired `checkpoint_buffer` and `checkpoint_rank_buffers` fields
are rejected.

Priority sampling defaults to `exp` in `TrainConfig`; explicit `lin` and `bin`
configurations remain available.

`_step_accounting` supplies local/global statistics to startup messages, logs,
effective configuration and checkpoints. Seed steps and update intervals use
local sample counts; the normal distributed launch accounts for budget, model/
replay saves, evaluation and logging in global samples. Checkpointing precedes
evaluation at a shared boundary, and collection never crosses the sample budget.
Evaluation shares the simulator, so the preceding transition is marked truncated
before the evaluation/reset boundary. Updates use a strict
`local_time > _replay_update_after` condition and require sampleable replay on
all ranks. Fresh runs set this threshold to `num_seed_steps`; replay restore or
fresh-buffer warm-up determines it on resumed runs.

All ranks participate in metric reduction; only artifact-writing ranks format
logs and compute contact diagnostics. Each evaluation writes its CSV rows in
one batch. FPS uses actual completed samples since the previous measurement,
including the current rollout batch, with the first interval starting at the
restored checkpoint step. Its elapsed time includes intervening evaluation,
saving and logging I/O. Logged step labels retain their existing scheduling units.
CSV logging reads the existing header on resume and adds columns when new metrics
appear. Schema expansion streams old rows through a temporary file and replaces
the original only after the rewrite succeeds; missing values remain blank.

## Completion, planned stopping and replay restore

Normal completion writes a final model and rank 0's complete replay snapshot.
Reward inference uses rank 0; routine model saves do not imply replay data from
every rank. For a planned stop, create `SAVE_AND_STOP` in the run's work directory:

```bash
touch runs/ufo/SAVE_AND_STOP
```

At the next model/replay save boundary, all ranks save their replay with the
same model save ID and stop. The marker is removed only after coordinated
saving succeeds. SIGINT/SIGTERM do not provide this planned-stop guarantee.

Model/training-status restoration requires the original world size. Training
restores replay only when every shard matches the model's save ID, global step
and world size. If a shard is missing or mismatched, every rank warms up fresh
replay while retaining saved weights, optimizer state and training counters.
Restored replay honors the saved warm-up threshold. Restored
replay marks its final row truncated because the simulator starts a new episode.
Rank-0 snapshots remain useful for reward inference when insufficient for
multi-rank training restore. Device selection is applied before allocating
loaded replay, so CUDA snapshots can be loaded on CPU.

Training and inference reject checkpoints carrying `.saving`, which marks an
incomplete model save. Environment and RNG state are not saved: restoring replay
does not guarantee step-by-step trajectory reproduction. Historical checkpoint/
configuration layouts are not migrated.

## Launch configuration versus a restored agent

On a fresh run, `cfg.agent` from the selected builder constructs the agent.
On resume, `training/trainer.py:create_agent_or_load_checkpoint` loads the agent from
`RUN/checkpoint/config.json`, its saved build arguments, weights and optimizer
state. The launch device is applied during loading; launch width/critic/loss
hyperparameters do not rewrite the restored agent configuration.

`Trainer.cfg` still contains the current launch configuration. The environment
is built from it before agent loading; scheduling, expert/replay setup and step
accounting also consult it. Resuming does not replace the entire launch config
with historical settings. Keep the selected algorithm and robot/observation
configuration compatible with the saved agent, and compare the two agent
configuration entries in `effective_config.json` when changing launch arguments.

## Inspect the configuration actually used

Rank 0 writes `RUN/effective_config.json` during trainer initialization,
alongside `RUN/config.json` (the launch configuration) and `RUN/config.yaml`
(the resolved task configuration):

| Entry | Meaning |
| --- | --- |
| `training` | Constructed launch configuration, including its requested `training.agent` |
| `agent` | Actual agent configuration, loaded from the checkpoint on resume |
| `task` | Resolved robot/task settings used to create the current environment |
| `resume` | Counters and checkpoint metadata loaded at initialization |
| `step_accounting` | Local/global counts, effective batch size and interval units calculated from the launch config |

This initialization artifact is not a continuously updated training-status file.
For the latest completed save, inspect `RUN/checkpoint/train_status.json`.


## Validation and evidence

```bash
uv run --no-sync pytest -q
uv run --no-sync ruff check ufo tests
uv run --no-sync python -m compileall -q ufo tests
```

Current regression coverage includes [training scheduling](../tests/test_training_flow.py),
[checkpoint/replay policy](../tests/test_checkpoint_replay.py),
[critic topology and resumed updates](../tests/test_critic_topology.py),
[agent updates, optimizers and checkpoint loading](../tests/test_agent_runtime.py) and
[environment snapshots](../tests/test_environment_kernels.py).
GPU physics checks and their opt-in command are documented in
the following section.

### GPU physics checks

On a free GPU, provide an existing robot-compatible LaFAN motion file:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl \
UFO_TEST_MOTION_FILE=/path/to/lafan_motions.pkl \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uv run --no-sync pytest -q tests/test_mjlab_physics_integration.py
```

The suite builds eight G1 worlds with domain randomization on/off, without an
agent or checkpoint. It checks partial-reset isolation of physics, action/history,
motion state, counters and event timers; actual randomized masses/COM/friction;
joint-zero targets; scheduled pushes; and contact-dependent auxiliary/weighted
rewards. Its independent MuJoCo Warp per-contact force sum must agree with UFO
at `atol=0.05 N, rtol=0.002`.

### Recorded validation and limits

These are dated results, not a claim that every later change was GPU-tested:

- **2026-09-28, mjlab 1.6.0 / MuJoCo and MuJoCo Warp 3.11.0:** physics checks
  passed 7 tests and skipped 1 (pushes disabled in the DR-off case). Both contact
  cases had failed before the force-sign correction. Partial reset uses the public
  mjlab reset API. Mass randomization retains mass-only semantics; it does not
  randomize density/inertia. Earlier short eight-rank FB/TeCH runs also exercised
  training, evaluation and synchronized model state. They do not establish
  long-run learning equivalence or unchanged trajectories after a physics upgrade.
- **2026-10-08, environment/logging/config cleanup:** CPU regression passed
  241 tests and 56 subtests, with 8 opt-in GPU physics tests skipped. All 260
  tested FB/TeCH configuration variants and retained task defaults matched the
  previous version. Tests cover snapshot ownership after reset, eight-process
  Gloo evaluation, batched CSV equivalence and FPS accounting. Ruff, compileall
  and whitespace checks passed. GPUs were occupied, so this cleanup had no new
  physics rollout, GPU throughput measurement or long-run quality comparison.
  Artifacts are at `/data/xue/UFO_open_source_runs/code_cleanup_20261008_7de0o56o`
  on H200-2.

Detailed historical performance, precision, communication and refactor reports
are experiment records rather than current launch instructions. Their original
documents are archived on H200-2 under
`/data/xue/UFO_open_source_runs/structure_cleanup_20261008_2a2d84qo/docs_before/`;
the same audit directory contains `before.tar.gz`. Use the fixed execution
policy above; obsolete experimental switches are not supported.

FPS comparisons require the same data, hardware, precision, per-rank environment
count, batch size and update schedule. Report compilation, evaluation and
checkpoint time separately from steady-state FPS. Short numerical checks do
not establish long-run learning-curve equivalence. This layout cleanup makes
no throughput claim. Third-party copyright and license notices remain applicable.

## RP1 and RPO alongside G1

`configs/robots/rp1_3.yaml` and `configs/robots/rpo.yaml` own their physical tasks
and FB baselines. `training.fb_agent_overrides` is validated against the native
FB config; `training.train_overrides` sets seed, run length and schedules. CLI
run lengths, seed and critic choices take precedence. There is no robot-specific
training loop.

- `envs/actuators.py` owns G1 motor calibration and YAML parameter expansion;
  RP1 and RPO use the RoboParty DC motor envelope and sample command delay once per reset.
- `envs/contact.py` uses MuJoCo Warp contact forces to sum the normal component of every contact, with a three-substep history. G1 keeps its native net-force sensor.
- `envs/rewards.py` holds the shared reward kernels; RP1 and RPO allocate jerk/momentum history.
- `tools/rp1_motion_import.py` checks the existing BeyondMimic NPZ dataset and emits standard UFO PKL files. Explicit derivative fields are optional; old G1 data follows the existing derivative path.
- `tools/rpo_motion_import.py` validates the original RPO NPZ manifest, FK and stored velocities before emitting the same UFO PKL format.

The RP1 GPU physics checks are opt-in:

```bash
CUDA_VISIBLE_DEVICES=0 UFO_TEST_RP1_MOTION_FILE=ufo/data/rp1_24dof_lafan_v7_10s-clipped.pkl \
  uv run pytest -q tests/test_rp1_physics_integration.py
```


## User entrypoints and robot selection

`--robot g1|rp1|rpo` resolves built-in robot YAML and training data through
`tools/robot_config.py`; custom robots retain `--robot-config` and explicit data.
`inference/__main__.py` dispatches the three existing parsers without duplicating
rollout logic. Inference validates robot identity against `RUN/config.json` and
preserves its control and observation snapshot while allowing matching asset relocation.
TeCH has no reward-inference method; the reward entrypoint rejects its checkpoints.
`tools/robot_setup.py` composes existing inspection/import tools and reports draft
calibration requirements, without automatically starting training.

Standalone MAE uses `python -m ufo.tools.evaluate_mae tracking|goal`. Tracking MAE
accumulates per-motion errors on the device and transfers each completed chunk
once; final statistics still exclude padded frames and the initial reset frame.
Goal MAE remains the error at the final reached pose. The former separate MAE
modules and `export_backward_encoder_onnx` module have been replaced by these
entrypoints.

## Tracking metric names and units

`distance` is the mean L2 distance between joint-angle vectors, in radians.
`joint_delta_l2_rad_per_frame` compares first temporal differences;
`joint_delta2_l2_rad_per_frame2` compares second temporal differences. They are
per-frame differences, not angular velocity or acceleration in seconds.
`emd`, `obs_state_emd`, and priority sampling keep their existing definitions.

Historical logs used `mpjpe_l = distance * 1000` despite measuring joint angles,
not Cartesian body positions. This duplicate field is removed. Convert old
`vel_dist / 1000` and `accel_dist / 100` to compare against the new first- and
second-difference fields respectively; do not compare their raw logged values.
Historical metric values and W&B runs are not converted. A resumed CSV may gain
new columns, with historical rows left blank for those fields.

Tracking, goal and reward inference stream frames to an MP4 encoder. A temporary
video replaces the destination only after the rollout and encoder complete;
failures close the encoder, remove the temporary file and preserve any prior video.
