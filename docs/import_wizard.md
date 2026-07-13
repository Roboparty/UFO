# Import Wizard CLI

UFO includes a lightweight import wizard layer for bringing new MuJoCo robot XML files and robot-state CSV/NPZ motion data into the existing robot-aware motion data pipeline.

The available commands are:

- `humanoidverse.tools.robot_inspect`
- `humanoidverse.tools.data_inspect`
- `humanoidverse.tools.data_build`

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

## Important Notes

`robot_inspect` writes a draft robot YAML, not a final curated robot config. The generated semantic fields are heuristic guesses. Users should inspect and edit at least:

- `base_body`
- `feet`
- `hands`
- `key_bodies`

For robots that already have curated configs, such as G1, formal experiments should continue to use reference configs like `configs/robots/g1_29dof.yaml`.

The import wizard currently solves the robot-aware motion data import workflow and can emit XML-derived training drafts. It does not mean the UFO training environment is fully robot-agnostic, and it does not enable cross-robot shared-policy training by itself.
