# Robot adaptation and motion data

G1, RP1 and RPO share the mjlab environment and inference entrypoints.
G1/RP1 support FB and TeCH; the released RPO baseline uses FB.
Select them with `--robot g1|rp1|rpo`. Custom robots use `--robot-config` and motion
data already retargeted to that robot. The tools generate drafts and convert
motion formats; they do not perform retargeting or hardware calibration.

## Required inputs

- A MuJoCo XML and its referenced assets.
- A robot YAML with the XML's joint/body order and reviewed body semantics.
- `training.init_state`, `training.control` and `training.actuator` settings.
- Matching RobotState motion data; optional `training.task_overrides` and
  `training.motion_extensions` describe task differences.

XML determines qpos/qvel layout, actuator order, action dimensions, control
joint names and body names. Review feet, hands, key bodies, contact semantics,
default pose, PD gains, motor limits, rewards and termination settings before
training. Generated YAML carries `metadata.review_status: draft`; G1/RP1 formal
experiments should use their curated configurations.

All robots use the native task path described in
[configuration ownership](ARCHITECTURE.md#configuration-ownership-and-precedence).
Non-empty historical Hydra robot groups are rejected; translate their settings
into `training.task_overrides`.

## Generate or validate a robot configuration

The combined setup command can generate a robot draft and prepare retargeted data:

```bash
uv run python -m ufo.tools.robot_setup \
  --name my_robot \
  --xml /path/to/robot.xml \
  --output configs/robots/my_robot.yaml \
  --motion-source '/path/to/motions/*.csv' \
  --fps 50
```

It writes the robot YAML, a data manifest and a setup report containing a smoke
training command. It does not start training. Omit `--motion-source` to prepare
only the robot draft. For a reviewed YAML, replace `--xml` and `--output` with
`--robot-config configs/robots/my_robot.yaml`. Existing outputs are protected;
choose fresh paths or explicitly remove obsolete reports before rerunning.

For individual draft generation and inspection options:

```bash
uv run python -m ufo.tools.robot_inspect \
  --xml /path/to/robot.xml \
  --name my_robot \
  --out configs/robots/my_robot.yaml \
  --with-training
```

`--with-training` includes control, actuator and task fields in the same YAML.
Without it, the command writes a structural robot configuration for data tools.
The generated action scale, gains, initialization and semantic body groups are
starting points for review, not measured robot parameters.

An optional `--urdf /path/to/robot.urdf` supplies effort/velocity limits,
physical damping/friction, semantic hints and metadata-only symmetric DOF pairs.
XML remains authoritative for simulation layout. XML joint limits win by default;
conflicts produce warnings. `--prefer-urdf-limits true` selects URDF limits for
the draft. Use `--urdf-joint-name-map mapping.yaml` (or JSON) to map XML joint names to
URDF joint names. Symmetric DOF metadata is not applied automatically. Validate
these hardware hints against the actual robot before training or deployment.

## Inspect and build motion data

The data tools share one entrypoint:

```bash
uv run python -m ufo.tools.motion_data inspect --help
uv run python -m ufo.tools.motion_data build --help
```

`inspect` reports the detected schema, FPS, duration, estimated clips and a
suggested manifest without writing outputs. `build` writes a manifest and builds
full-sequence inference data plus clipped training data.
Inspection uses the same CSV/NPZ readers and validation as cache building,
including finite values, nonzero quaternions, shapes and joint names.

```bash
uv run python -m ufo.tools.motion_data inspect \
  --robot configs/robots/my_robot.yaml \
  --source '/path/to/motions/*.csv' \
  --format robot_state_csv --fps 50

uv run python -m ufo.tools.motion_data build \
  --robot configs/robots/my_robot.yaml \
  --source '/path/to/motions/*.csv' \
  --format robot_state_csv --name my_motion --fps 50 \
  --clip-seconds 10 \
  --out configs/data/my_motion.yaml
```

Both commands accept multiple sources or a glob, `--stride-seconds`,
`--min-clip-seconds` and `--keep-short`/`--drop-short`. Build additionally accepts
`--weight`, `--no-build` (manifest only), `--rebuild-cache` and `--force` to
explicitly replace an existing manifest. Invalid data and mismatched joint
names are rejected.

The RobotState schema contains `root_pos`, `root_quat`, `dof_pos` and FPS/time.
CSV and NPZ readers support this schema; use `--format auto` to detect the
format. Headerless CSV uses root XYZ, quaternion XYZW and controlled joints in
robot-config order, with an optional leading time column. Supply `--fps` when
the source has no timing information.

An explicit `--fps` (or manifest `fps`) takes precedence over source timing in
both inspection and cache building. Without it, CSV uses its time column and
NPZ uses its `fps` field or time array. Inspection and loading share these rules.

Manifests support `robot_state_csv`, `robot_state_npz` and `ufo_pkl`. For an
existing UFO motion PKL, use `--data-path` directly; no import YAML is required.
Use a manifest for reusable source mixtures or automatic CSV/NPZ conversion.
See the [example manifest](../configs/data/example_robot_state_auto_build.yaml).
Cache directories include the manifest's absolute-path hash; dataset names must
remain unique after filename normalization. Training caches use `_train_ufo.pkl`
regardless of clip length. Older caches remain untouched and are rebuilt at the
new paths. After changing source data or clip settings, use `--rebuild-cache`.

## Use converted data

The default commands look for these files relative to the repository root:

| Robot | Training clips | Full motions for inference |
|---|---|---|
| G1 | `ufo/data/lafan_29dof_10s-clipped.pkl` | `ufo/data/lafan_29dof.pkl` |
| RP1 | `ufo/data/rp1_24dof_lafan_v7_10s-clipped.pkl` | `ufo/data/rp1_24dof_lafan_v7.pkl` |
| RPO | `ufo/data/rpo_23dof_lafan_grounded_10s-clipped.pkl` | `ufo/data/rpo_23dof_lafan_grounded.pkl` |

Copy or symlink your matching files to these paths. Alternatively, pass their
actual paths with `--data-path` when training or running inference. Keep full
motions for goal definitions: motion IDs and frame indices differ from clips.
Run `bash scripts/download_data.sh g1_lafan`, `rp1_lafan` or `rpo_lafan` to download the released files from
[UFO-MotionData](https://huggingface.co/datasets/xuewang/UFO-MotionData).
The downloader verifies existing files by SHA256 and skips downloading
verified copies. If Hugging Face is unreachable, obtain the same files on a
machine with access and copy them here; the downloader cannot bypass a network
block. RP1 contains 55 full motions and 1,166 training clips; RPO contains 56
full motions and 1,168 training clips. Both are 50 Hz. The optional import
commands below convert your own original NPZ datasets and manifests.

## RP1 data import

To convert the original RP1 LaFAN v7 BeyondMimic NPZ files, provide their manifest:

```bash
uv run python -m ufo.tools.rp1_motion_import \
  --manifest /path/to/rp1.3_lafan_v7/manifest.json \
  --robot-config configs/robots/rp1_3.yaml \
  --output ufo/data/rp1_24dof_lafan_v7.pkl \
  --train-output ufo/data/rp1_24dof_lafan_v7_10s-clipped.pkl \
  --report /tmp/rp1_import.json
```

The importer checks hashes, joint/body order and FK, preserves joint velocities,
and converts body COM velocities to link-origin velocities. Use `--output` for
full-motion inference and `--train-output` for training. Motion IDs depend on the
loaded dataset; full sequences and training clips have different indices.

## RPO data import

The RPO importer accepts the manifest and NPZ files produced by the original
RPO conversion pipeline:

```bash
uv run python -m ufo.tools.rpo_motion_import \
  --manifest /path/to/rpo_lafan/manifest.json \
  --robot-config configs/robots/rpo.yaml \
  --output ufo/data/rpo_23dof_lafan_grounded.pkl \
  --train-output ufo/data/rpo_23dof_lafan_grounded_10s-clipped.pkl \
  --report /tmp/rpo_import.json
```

It verifies hashes, joint/body order, finite-difference velocities and sampled
FK before preserving the source 50 Hz motion states. Output paths must not
already exist.

## Validate training and inference

After reviewing the robot draft and converted motion:

```bash
./run_train.sh --agent fb \
  --data-manifest configs/data/my_motion.yaml \
  --gpu-ids single --smoke
```

The manifest's robot configuration is selected automatically unless explicitly
supplied on the CLI; when both are present, UFO checks their compatibility.
Built-in robot selection and explicit YAML selection are mutually exclusive.
Remove `--smoke` and choose GPU count and training budget after validation.

Tracking needs matching checkpoint/motion data. Goal inference additionally
needs robot-specific goal JSON with matching motion names, IDs and frame indices.
FB reward inference needs saved rank-0 replay; non-G1 robots currently support
root/locomotion tasks. TeCH supports tracking and goal inference only. Policies
and ONNX exports are tied to their robot's action and observation dimensions.

## Implementation

`tools/robot_config.py` owns XML/YAML validation and training metadata;
`tools/robot_inspect.py` generates drafts. `tools/motion_data.py` owns data inspection
and build commands. `motion/formats.py` reads, validates, converts and clips;
`motion/manifest.py` coordinates sources and caches. `robot_setup.py` composes
these tools. Adding a robot may require additional actuator or reward semantics;
the setup command does not guarantee arbitrary-robot support or cross-robot
shared-policy training.

Robot inspection and training drafts share one semantic inference result.
Explicit `--feet`, `--hands`, and `--key-bodies` selections also apply to training
metadata; an explicit empty selection stays empty and requires manual review.
