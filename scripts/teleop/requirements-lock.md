# Teleop External Revision Lock

Captured: 2026-07-25

These are the upstream revisions used by the UFO-Deploy teleop setup instructions. Pinning
them keeps new deployments from silently drifting with upstream `main` or `master`.

| Component | Repository | Ref used by UFO-Deploy docs |
| --- | --- | --- |
| Canonical PICO/XRobot -> G1 qpos retargeting source | `https://github.com/Axellwppr/motion_tracking.git` | branch `sim2real`, commit `0d5ba31e33397f3543d350d98b637e26d92f470a` |
| XRoboToolkit Python binding | `https://github.com/Axellwppr/XRoboToolkit-PC-Service-Pybind.git` | `a7ae949849ef335f0fa7bbef1e741c5b35e2124e` |
| XRoboToolkit service source / SDK source | `https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git` | `85bac4dbc1fd5cef42c74a160d9c30aa3491f122` |
| XRoboToolkit arm64 headless `.deb` | `https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/tag/v1.0.0` | tag `v1.0.0`, commit `6386009475369615fa984bca0a6e9902cd81eb2a`, asset SHA256 `532c605dfa1a02b05b7c285b856c91771c78623cded30ef5b16ea371de49ed5f` |

Canonical scope is `PICO/XRobot raw body data -> G1 qpos`. UFO's realtime `z` server,
backward encoder, policy inference, and robot command bridge are not imported from
motion_tracking.

When updating any of these revisions, rerun `scripts/teleop/check_teleop_env.py` in the
target teleop environment and verify that the vendored `motion_tracking_retarget` assets,
`xrobot_to_g1.json`, canonical G1 XML, joint permutation, and `xrobotoolkit_sdk` APIs are
still compatible.
