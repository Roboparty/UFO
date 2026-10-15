# Deployment dependencies

| Host | Python environment |
|---|---|
| Workstation | Conda with Python 3.10 |
| G1 onboard | Python 3.10 venv |

The onboard venv keeps native Unitree, XRoboToolkit and CycloneDDS bindings
compatible with the system ABI.

| Flow | Python requirements | Additional native components |
|---|---|---|
| Tracking / goal / reward sim2sim | `requirements/runtime.txt` | None |
| Tracking / goal / reward sim2real | `requirements/runtime.txt` | `g1_interface`, CycloneDDS |
| Teleop sim2sim | `requirements/teleop.txt` | XRoboToolkit service, `xrobotoolkit_sdk` |
| Teleop sim2real (onboard) | `requirements/teleop.txt` | Both SDK stacks above |
| Read-only robot state diagnostic | Runtime + optional `unitree_sdk2py` | CycloneDDS |

`requirements/teleop.txt` inherits `requirements/runtime.txt`. The vendored
`motion_tracking_retarget` uses Mink IK; it needs neither external GMR nor Torch.
`requirements/training.txt` is for export/development checks and is not needed
on the robot.

Ordinary inference reads already-exported z:

```text
robot observations + task z → actor ONNX → joint targets
```

Teleop adds a reference source on the same host:

```text
PICO → XRoboToolkit → retarget → G1 pose → backward ONNX → live z
```

Installation details: [onboard](../scripts/onboard/README.md),
[teleop](../scripts/teleop/README.md).
