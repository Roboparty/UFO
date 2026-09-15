# Changelog

## Unreleased

### Added

- NEURA 4NE-1 G3.3 bring-up: curated robot config, Hydra robot group, LaFAN data manifest, and
  `scripts/train_4ne1_g3_3.sh` (see `docs/superpowers/specs/2026-09-11-neura-4ne1-g3-3-port-design.md`).
- Manifest `ground_alignment` block and `data_build --ground-align` for RobotState sources that
  were retargeted without a floor constraint (per-frame root lift + residual-penetration clip filter).
- Training-time preview videos (`--video-every-checkpoints`, `--video-motion-ids`,
  `--video-max-steps`, `--video-render-size`) rendered by a background `tracking_inference`
  subprocess and uploaded to W&B under `video/`.
- `tags.robot` in the train config.
- `humanoidverse.tools.motion_trim_tpose`: drop the leading T-pose calibration hold from
  retargeted motion NPZ files and matching MP4 previews. Applied in place to
  `humanoidverse/data/lafan1_4ne1_groundoffset/motions` (77 files).

### Fixed

- MJCF files with `--` inside XML comments (accepted by MuJoCo, rejected by `xml.etree`/`lxml`)
  no longer crash MotionLib or the inference relabel utilities.
- torch now installs from the CUDA 12.8 wheel index so Blackwell GPUs (sm_120) are supported.
- `robot_inspect` matched side tokens as raw substrings (`r_` inside `sensor_...`), so robots with
  `sensor_*` foot bodies got a single foot; tokens now match whole name segments.
- `tools/export_backward_encoder_onnx.py` imported a non-existent module.

## v0.1.0

Initial UFO release candidate.

### Added

- Unified UFO training entrypoint with FB and TLDR unsupervised RL presets.
- Documented current `update_z_every_step` defaults: FB uses `100`, TLDR uses `10`.
- Distributed MJLab training support for Unitree G1-style humanoid policies.
- Weighted multi-source motion dataset manifests with source-level sampling weights.
- RobotState motion import pipeline for `root_pos`, `root_quat`, `dof_pos`, and `fps` or `time`.
- CSV and NPZ readers for RobotState data, plus import wizard tools:
  - `humanoidverse.tools.robot_inspect`
  - `humanoidverse.tools.data_inspect`
  - `humanoidverse.tools.data_build`
- Tracking, goal, and reward inference entrypoints with MP4/ONNX export support.
- Release smoke script at `scripts/smoke_release.sh`.

### Hardened

- Manifest dataset names must be unique.
- Manifest weights must be finite, non-negative, and sum to a positive value.
- User-facing failed-motion sampling text now uses "PMCP-inspired failed-motion prioritization".

### Known Limitations

- UFO currently supports robot-aware motion data import, but the training environment is still G1/MJLab-centered.
- `robot_inspect` generates a draft robot YAML and cannot fully infer semantic fields such as feet, hands, and key bodies without user review.
- RobotState import does not perform FPS resampling.
- RobotState-to-UFO conversion currently supports hinge control joints.
- Cross-robot shared-policy training and arbitrary-robot training environments are not part of v0.1.0.
