# D435i Terrain Bring-up

The hardware path is deliberately separated from policy inference:

```text
D435i depth (native units)
  -> meters / invalid=NaN
  -> full-FOV area downsample (64x36)
  -> calibrated K'
  -> camera-to-torso T + synchronized G1 pose
  -> DepthTerrainAdapter
  -> temporal completion (optional)
  -> 273D terrain_actor
```

The adapter expects optical camera axes `+x right, +y down, +z forward` and
world-frame `xyzw` quaternions. Do not crop the native image. The calibration
JSON contains the native D435i depth intrinsics, depth scale, and the optical
camera-to-torso transform. The actual device intrinsics must replace the
example values; they are not inferred from the FOV.

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

`RealSenseDepthSource` provides an optional `pyrealsense2` live depth reader,
but pose synchronization remains external and must be supplied by the G1 state
estimator. This is intentional: the first hardware milestone is static map
geometry, followed by manually moving the robot with synchronized poses.
