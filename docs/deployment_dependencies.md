# Deployment Dependencies

Current deploy does not require the external `general_motion_retargeting`/GMR
package.

## Python Environment Policy

Release-supported defaults:

| Host | Python environment |
|-|-|
| Workstation | Conda with Python 3.10 |
| G1 onboard | Python 3.10 venv |

Use workstation Conda for local MuJoCo sim2sim, split-workstation teleop,
development checks, and packaging validation. Use the G1 onboard Python 3.10
venv for ordinary Sim2Real, onboard PICO teleop Sim2Real, and readonly onboard
diagnostics. The validated G1 onboard deployment uses venv by default because
native Unitree, XRoboToolkit, and CycloneDDS libraries depend on system ABI
compatibility.

## Flow Matrix

| Flow | Human pose | XRoboToolkit | Retarget | GMR | Torch |
|-|-|-|-|-|-|
| MuJoCo sim2sim | No | No | No | No | Training only |
| Ordinary onboard Sim2Real | No | No | No | No | No |
| PICO Teleop Sim2Real | Yes | Yes | `motion_tracking_retarget` + Mink IK | No | No for retarget |

## Ordinary Sim2Real

Ordinary onboard Sim2Real does not consume PICO data, XRoboToolkit streams,
human pose, or retargeted qpos. Its deployment dependency path is:

```text
policy ONNX -> observation -> backward encoder -> latent z -> UFO policy -> G1
```

Use `requirements/runtime.txt` for this flow.

## PICO Teleop Sim2Real

PICO teleop Sim2Real consumes live body data and retargets it with the vendored
retargeter:

```text
PICO -> XRoboToolkit -> xrobot body joints
-> scripts/teleop/motion_tracking_retarget -> Mink IK
-> G1 qpos -> realtime z -> UFO policy
```

This flow requires the XRoboToolkit service, `xrobotoolkit_sdk`, vendored
`motion_tracking_retarget` assets, and Mink dependencies. Direct onboard PICO
teleop Sim2Real also needs runtime dependencies because the same host runs
realtime `z`, `backward_encoder.onnx`, ONNX Runtime policy inference, and the G1
runtime. It does not require `general_motion_retargeting`, the
`GeneralMotionRetargeting` class, or torch for retargeting.

Use `requirements/teleop.txt` for PICO teleop deployment. It inherits
`requirements/runtime.txt`. The `xrobotoolkit_sdk` native binding is installed
separately with the platform-specific SDK installer.

## Legacy GMR

The old GMR architecture is not used by the current deploy runtime. It is not a
deployment dependency for ordinary Sim2Real or PICO teleop Sim2Real.
