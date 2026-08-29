# D435i Terrain Bring-up

The hardware path is deliberately separated from policy inference:

```text
D435i depth (480x270 Z16 at 60 Hz, native units)
  -> meters / invalid=NaN
  -> full-FOV area downsample (64x36)
  -> calibrated K'
  -> camera-to-torso T + synchronized G1 pose
  -> DepthTerrainAdapter
  -> temporal completion (optional)
  -> 273D terrain_actor
```

The adapter expects optical camera axes `+x right, +y down, +z forward` and
world-frame `xyzw` quaternions. Do not crop, inpaint, hole-fill, or spatially
filter the native image. The calibration JSON contains the active D435i depth
profile's intrinsics and depth scale plus the explicitly named
`T_torso_link_from_camera_optical`. Intrinsics are always read from the active
device profile; they are never copied from a different resolution or inferred
from a nominal field of view.

The logger retains every 60 Hz camera frame and writes a timestamp-derived
`runtime_frame_index` selecting the latest camera exposure at each 50 Hz
runtime tick. Offline replay honors this index automatically. Thus a 31-frame
model window still spans approximately 0.6 s while the raw NPZ keeps every
sensor frame.

## Nominal G1 camera transform

Project Instinct's pinned G1 configuration provides
`T_torso_link_from_camera_depth_link` with translation
`(0.0487988662, 0.015, 0.4378029938)` m and a roughly 48-degree downward
mount. Its `camera_depth_link` uses robot axes `+x forward, +y left, +z up`,
not the standard optical axes consumed by `DepthTerrainAdapter`.

The logger's nominal preset therefore composes

```text
T_torso_link_from_camera_optical
  = T_torso_link_from_camera_depth_link
  @ T_camera_depth_link_from_camera_optical
```

with the fixed mapping

```text
(X, Y, Z)_optical -> (Z, -X, -Y)_camera_depth_link

R_camera_depth_link_from_camera_optical =
  [ 0  0  1 ]
  [-1  0  0 ]
  [ 0 -1  0 ]
```

The source commit and both component transforms are embedded in every capture
that uses the nominal preset. This is a nominal bring-up reference, not a
claim of measured calibration.

## Synchronized live capture on G1

`humanoidverse/tools/realsense_g1_live_logger.py` is a standalone Python 3.8
logger for the G1 Jetson. It has no Actor, LowCmd, or motor publisher. A single
Unitree SDK DDS participant subscribes to:

- `rt/dog_odom` for `odom -> robot_center` translation and orientation;
- `rt/lowstate` for robot tick, pelvis IMU, and 29 joint states;
- `rt/secondary_imu` for the raw torso IMU cross-check.

The D435i is opened directly through `pyrealsense2`. The logger stores native
Z16 frames, active-profile K, distortion model and coefficients, depth scale,
frame number, camera timestamp and domain, host monotonic/realtime receipt
times, supported frame metadata, and all three unmodified G1 state streams.
It never falls back to the example intrinsics.

On the current G1 image, expose the onboard Unitree SDK before running the
read-only source probe:

```bash
export PYTHONPATH=/home/unitree/hhtools-bridge/unitree_sdk2_python:\
/home/unitree/hhtools-bridge/.venv/lib/python3.8/site-packages

python3 /home/unitree/realsense_g1_live_logger.py \
  --probe --duration-s 2 --network-interface eth0 \
  --odom-topic /dog_odom --serial 254322076596
```

For the first static geometry captures, explicitly select the pinned Project
Instinct nominal `T_torso_link_from_camera_optical`:

```bash
python3 /home/unitree/realsense_g1_live_logger.py \
  --output /home/unitree/data/flat_001.npz \
  --use-instinct-nominal-torso-from-optical \
  --scene flat --duration-s 10 \
  --accept-robot-center-egomotion-proxy \
  --network-interface eth0 --odom-topic /dog_odom \
  --serial 254322076596
```

A later measured override uses an explicit transform contract:

```json
{
  "transform_name": "T_torso_link_from_camera_optical",
  "torso_from_camera_optical_translation_m": [0.0, 0.0, 0.0],
  "torso_from_camera_optical_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]
}
```

Pass it with `--torso-from-optical-json`. The placeholder values above are
schema documentation only and must not be used as a calibration.

Position is linearly interpolated and orientation uses quaternion SLERP at the
camera exposure time. Raw `/dog_odom` is retained as `odom -> robot_center`
and used as an egomotion proxy; the capture does not assert
`robot_center == pelvis`. Compatibility `pelvis_*` replay arrays currently
alias that proxy, and the torso pose is a nominal G1 waist yaw-roll-pitch FK.
Raw pelvis and torso IMUs remain in the NPZ; their physical mounting offsets
are not subtracted implicitly. Every frame records signed nearest-sample
`sync_dt` and the full interpolation bracket span. A missing bracket or
excessive synchronization gap aborts the capture.

A fixed `robot_center -> pelvis` offset cancels for straight translation, so a
slow supported forward/backward `sync_motion` test can validate temporal
alignment now. It does not cancel exactly during rotation: yaw produces a
lever-arm translation unless that fixed transform is known. Keep yaw motion
diagnostic-only until the frame transform is measured or otherwise verified.

The capture produces the raw NPZ, a replay-ready
`.runtime_calibration.json`, and a `.summary.json`. Use a separate file and a
fresh temporal reset for `flat`, `stair_10cm`, `stair_14cm`, `stair_18cm`, and
`sync_motion`. The raw arrays remain 60 Hz; `runtime_frame_index`,
`runtime_tick_timestamp_s`, and `runtime_camera_age_s` make the 50 Hz selection
auditable without duplicating or discarding the source depth frames.

## Static replay

Create an `.npz` with these arrays, all indexed by frame `T`:

- `depth`: `[T, native_height, native_width]`, normally `uint16` native units;
- `torso_pos_w`: `[T, 3]`;
- `torso_quat_w_xyzw`: `[T, 4]`;
- `pelvis_pos_w`: `[T, 3]`;
- `pelvis_heading_quat_w_xyzw`: `[T, 4]`;
- `timestamp_s`: `[T]`;
- optional `proprio`: `[T, proprio_dim]`;
- optional `reset_mask`: `[T]`, with the first frame set to `true`.

Run projection-only geometry validation with:

```bash
python -m humanoidverse.tools.realsense_terrain_bringup \
  --input-npz /path/to/d435i_static.npz \
  --calibration /path/to/d435i_calibration.json \
  --output /path/to/bringup/summary.json \
  --device cuda:0
```

Add `--perception-checkpoint` only after the partial-map geometry is correct.
The command writes `summary.npz` beside the summary JSON with `partial_map`,
`visible_mask`, and `terrain_actor`. A missing pose or timestamp is a hard
error; a depth miss remains NaN and is never interpreted as flat terrain.

For a paired 2 m range-gate and raw-resolution Gaussian replay, use:

```bash
python -m humanoidverse.tools.realsense_terrain_bringup \
  --input-npz /path/to/d435i_static.npz \
  --calibration /path/to/d435i_calibration.json \
  --output /path/to/bringup/summary_blur.json \
  --depth-gate-max 2.0 \
  --blur-probability 0.5 \
  --blur-sigma-min-px 0 \
  --blur-sigma-max-px 3
```

This augmentation is validity-preserving: an invalid source pixel stays
invalid after blur. It is not image inpainting, and no depth normalization is
performed before the metric-space adapter.

The first hardware milestone remains `flat -> stair_10cm -> stair_14cm ->
stair_18cm` static map geometry with completion disabled first. Follow it with
a slow, supported, approximately straight forward/backward motion test. Do not
enable policy motor control during either step.
