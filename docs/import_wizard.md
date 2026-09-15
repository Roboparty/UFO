# Import Wizard CLI

UFO includes a lightweight import wizard layer for bringing new MuJoCo robot XML files and robot-state CSV/NPZ motion data into the existing robot-aware motion data pipeline.

The available commands are:

- `humanoidverse.tools.robot_inspect`
- `humanoidverse.tools.data_inspect`
- `humanoidverse.tools.data_build`
- `humanoidverse.tools.motion_trim_tpose`

These tools generate and check configuration files, then call the existing `RobotSpec`, motion adapters, and manifest auto-build path. They do not change FB/TeCH algorithms, MotionLib sampling, or the training environment.

## Supported Public Motion Formats

UFO's public data manifests intentionally expose only three motion formats:

UFO's recommended import path is based on a RobotState schema: `root_pos`, `root_quat`, `dof_pos`, and `fps` or `time`. CSV and NPZ are currently supported file readers for this schema.

- `ufo_pkl`: already processed UFO motion dictionaries, usually produced by UFO tools or existing retargeting pipelines.
- `robot_state_csv`: robot-state CSV files containing root pose and joint positions, interpreted with a robot YAML config.
- `robot_state_npz`: robot-state NPZ files containing root pose and joint positions, interpreted with a robot YAML config.

For new datasets, prefer `robot_state_csv` or `robot_state_npz` with the import wizard. For data that has already been converted into UFO's internal motion dictionary schema, use `ufo_pkl`.

## Recommended Flow

### XML-derived Training Drafts

`robot_inspect` can generate XML-derived draft robot configs for the training
path. Passing `--hydra-out` writes both the UFO RobotTrainingSpec YAML and a
Hydra robot config draft:

```bash
uv run python -m humanoidverse.tools.robot_inspect \
  --xml /path/to/robot.xml \
  --name my_robot \
  --out configs/robots/my_robot.yaml \
  --hydra-out humanoidverse/config/robot/my_robot/my_robot_auto.yaml
```

The draft extracts structural fields from XML, including body names, actuated
joint order, action dimensions, joint ranges, and the freejoint base body.
Training-specific fields still require review: semantic body groups, contact
bodies, default pose, PD gains, actuator parameters, and reward/termination
fields. Generated configs are marked `metadata.review_status: draft`. Formal
experiments should use curated robot configs.

Optional URDF assistance can be enabled with `--urdf /path/to/robot.urdf`. XML
still controls the MuJoCo layout: qpos/qvel order, actuator order, action
dimension, control joint names, and body names are not taken from URDF. URDF is
used only to fill draft hardware hints such as effort, velocity, physical
damping/friction, semantic name hints, and a metadata-only symmetric DOF pair
draft. If XML and URDF joint limits conflict, XML wins by default and the tool
emits a warning. Use `--urdf-joint-name-map` when XML and URDF joint names do not
match exactly.

1. Generate a draft robot config from a MuJoCo XML:

```bash
uv run python -m humanoidverse.tools.robot_inspect \
  --xml /path/to/robot.xml \
  --name my_robot \
  --out configs/robots/my_robot.yaml
```

2. Inspect robot-state CSV/NPZ motion data:

```bash
uv run python -m humanoidverse.tools.data_inspect \
  --robot configs/robots/my_robot.yaml \
  --source /path/to/motions/*.csv \
  --format robot_state_csv \
  --fps 50
```

3. Generate a data manifest and build the full pkl plus near10s training pkl:

```bash
uv run python -m humanoidverse.tools.data_build \
  --robot configs/robots/my_robot.yaml \
  --source /path/to/motions/*.csv \
  --format robot_state_csv \
  --name my_motion \
  --fps 50 \
  --clip-seconds 10 \
  --out configs/data/my_motion_auto_build.yaml \
  --rebuild-cache
```

4. Train with the generated manifest:

```bash
./run_train.sh \
  --agent fb \
  --data-manifest configs/data/my_motion_auto_build.yaml \
  --gpu-ids single \
  --smoke
```

Remove `--smoke` and set the desired GPU and training options only after the import path has been checked.

## Trim Leading T-Pose Holds

LAFAN1 captures start with a static T-pose for a few seconds. Absolute shoulder angles after
retargeting can wrap by ±π, so the cut is not a fixed frame count. The tool finds the end of
the opening hold as the later of:

- the first frame whose mean arm-joint distance from frame 0 exceeds 0.45 rad (wrapped)
- the first frame whose mean joint speed exceeds 0.004 rad/frame, if that hold lasted at least 0.3 s

then drops an extra 0.2 s. The search is capped at 20 s. Time-series arrays in the NPZ
(`joint_pos`, `base_pos_w`, `base_quat_w`, and any other length-T field) are sliced together.
The number of dropped frames is stored as `tpose_trimmed_frames`; a second run skips the file.
A sibling `*.mp4` with the original (pre-trim) frame count is re-encoded from the same start
frame so preview videos stay aligned with the NPZ.

```bash
uv run python -m humanoidverse.tools.motion_trim_tpose \
  --source humanoidverse/data/lafan1_4ne1_groundoffset/motions \
  --dry-run

uv run python -m humanoidverse.tools.motion_trim_tpose \
  --source humanoidverse/data/lafan1_4ne1_groundoffset/motions
```

`--skip-npz` / `--skip-video` process only one of the two. This was applied in place to all 77
motions under `humanoidverse/data/lafan1_4ne1_groundoffset/motions` (drops of 66–316 frames,
2.2–10.5 s at 30 fps). These NPZs still use `joint_pos` / `base_pos_w` / `base_quat_w` /
`framerate`; map them to `dof_pos` / `root_pos` / `root_quat` / `fps` before `data_build`.

## Ground Alignment for Floor-Unaware Retargets

Some retargeting pipelines match keypoints only and let the target robot's collision
geometry sink below the floor (feet, hands, or the whole body in lying poses). A
`robot_state_*` dataset can opt into a correction at build time:

```yaml
datasets:
  - name: my_motion
    format: robot_state_csv
    source_path: /path/to/motions/*.csv
    fps: 30
    ground_alignment:
      mode: per_frame_lift            # or "none" to only measure
      tolerance_m: 0.02               # penetration left untouched
      smooth_seconds: 0.3             # centered moving average of the per-frame lift
      max_residual_penetration_m: 0.15  # drop training clips still deeper than this; null keeps all
    auto_build:
      train_clip_seconds: 10.0
```

`data_build --ground-align [--ground-tolerance-m 0.02 --ground-smooth-seconds 0.3 --max-ground-penetration-m 0.15]`
writes the same block. The builder runs MuJoCo forward kinematics with the robot XML for
every frame, measures the deepest world-frame point of all collision geoms, lifts the root by
the smoothed excess penetration, and records `metadata.ground_alignment` statistics in each
motion. Clip filtering applies to the auto-built training clips only; the full-length
inference motions are lifted but never dropped. `ufo_pkl` sources are not touched.

Trade-offs: sustained segments (a robot lying on the floor) are lifted fully; brief spikes
(toe digs) are only partially corrected so the root does not bounce; clearly broken clips are
removed. The correct long-term fix is a floor-aware retargeter.

## Important Notes

`robot_inspect` writes a draft robot YAML, not a final curated robot config. The generated semantic fields are heuristic guesses. Users should inspect and edit at least:

- `base_body`
- `feet`
- `hands`
- `key_bodies`

For robots that already have curated configs, such as G1, formal experiments should continue to use reference configs like `configs/robots/g1_29dof.yaml`.

The import wizard currently solves the robot-aware motion data import workflow and can emit XML-derived training drafts. It does not mean the UFO training environment is fully robot-agnostic, and it does not enable cross-robot shared-policy training by itself.
