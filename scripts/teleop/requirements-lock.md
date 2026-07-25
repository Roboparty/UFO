# Teleop External Revision Lock

Captured: 2026-07-25

These are the upstream revisions used by the UFO-Deploy teleop setup instructions. Pinning
them keeps new deployments from silently drifting with upstream `main` or `master`.

| Component | Repository | Ref used by UFO-Deploy docs |
| --- | --- | --- |
| GMR | `https://github.com/YanjieZe/GMR.git` | `bb1bbe40774794fceb2a7c579a3464a28e68c844` |
| XRoboToolkit Python binding | `https://github.com/Axellwppr/XRoboToolkit-PC-Service-Pybind.git` | `a7ae949849ef335f0fa7bbef1e741c5b35e2124e` |
| XRoboToolkit service source / SDK source | `https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git` | `85bac4dbc1fd5cef42c74a160d9c30aa3491f122` |
| XRoboToolkit arm64 headless `.deb` | `https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/tag/v1.0.0` | tag `v1.0.0`, commit `6386009475369615fa984bca0a6e9902cd81eb2a`, asset SHA256 `532c605dfa1a02b05b7c285b856c91771c78623cded30ef5b16ea371de49ed5f` |

When updating any of these revisions, rerun `scripts/teleop/check_teleop_env.py` in the
target teleop environment and verify that `xrobot -> unitree_g1` GMR assets and
`xrobotoolkit_sdk` APIs are still compatible.
