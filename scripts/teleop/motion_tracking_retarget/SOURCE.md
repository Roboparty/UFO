# Source

This package vendors the PICO/XRobot-to-G1 qpos retargeting path from:

- Repository: `https://github.com/Axellwppr/motion_tracking`
- Branch: `sim2real`
- Commit: `0d5ba31e33397f3543d350d98b637e26d92f470a`
- License: MIT, copied in `LICENSE`

## Canonical Boundary

Vendored scope:

```text
PICO/XRobot raw body data -> 24-joint XRobot snapshot -> G1 qpos
```

UFO-specific downstream code is not vendored from motion_tracking:

```text
G1 qpos -> scripts/realtime/realtime_z_server.py -> backward_encoder.onnx -> latent z
-> rl_policy/ufo_policy.py -> Unitree G1
```

## Vendored Files And Assets

The following motion_tracking files were adapted into this package:

- `sim2real/teleop/retarget/xrobot_retarget.py`
- `sim2real/teleop/retarget/params.py`
- `sim2real/teleop/retarget/fk.py`
- `sim2real/teleop/utils/helper.py`
- `sim2real/teleop/utils/math.py`
- `sim2real/teleop/utils/robot_config.py`
- `sim2real/config/g1/retarget/xrobot_to_g1.json`
- `sim2real/config/g1/assets/g1.xml`
- `sim2real/config/g1/assets/meshes/*.STL`

The full G1 XML and mesh assets are copied instead of generating a retarget-only XML, so
MuJoCo can load the canonical model without depending on an external motion_tracking
checkout.

## UFO-Side Changes

- Top-level imports such as `utils.*`, `retarget.*`, and `paths` were replaced with
  explicit package-relative imports.
- Path resolution was changed to use this package's local `assets/` directory and UFO's
  `config/teleop/g1.yaml`.
- XR snapshot parsing additionally rejects non-finite pose values before retargeting.
- Joint-order handling is explicit: canonical G1 XML joint order is permuted by joint name
  to UFO's expected 29-DoF output order before JSON replies are sent to `realtime_z_server.py`.
- The UFO teleop server keeps its existing ZMQ protocol, JSON reply schema, callback and
  polling SDK modes, multiprocessing worker shell, and optional viewer/debug channels.
