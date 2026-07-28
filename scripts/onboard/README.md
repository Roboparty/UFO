# Onboard Diagnostics And Deployment

These scripts support reproducible Unitree G1 onboard deployment. They are
diagnostic and setup helpers; by default they do not start real robot control.

## Clean Checkout Flow

From a fresh onboard clone:

```bash
git clone --branch deploy --single-branch https://github.com/Roboparty/UFO.git UFO-Deploy
cd UFO-Deploy
git rev-parse HEAD
```

For the 2026-07-25 validated deployment, the expected base commit is:

```text
d92bd38e914f888e90da3a303d6ec2a73ad6e60d
```

## External Runtime Dependencies

UFO deploy has three different external dependency categories.

### A. Required For Any Real G1 Control

Required:

- `g1_interface`

Use:

```text
UFO policy
-> g1_interface
-> G1 hardware control
```

Source:

```text
https://github.com/EGalahad/unitree_sdk2
```

This binding is required for ordinary onboard Sim2Real and onboard teleop
Sim2Real. It must match the UFO Python runtime ABI. The current runtime Python
is Python 3.10, so use:

```text
g1_interface.cpython-310-aarch64-linux-gnu.so
```

Do not use:

```text
g1_interface.cpython-38-aarch64-linux-gnu.so
g1_interface.cpython-310-x86_64-linux-gnu.so
```

Verify:

```bash
python - <<'PY'
import g1_interface
print(g1_interface.G1_NUM_MOTOR)
assert g1_interface.G1_NUM_MOTOR == 29
PY
```

### B. Required Only For Onboard PICO Teleop

Optional for ordinary Sim2Real. Required only for onboard PICO teleop:

- `xrobotoolkit_sdk`
- XRoboToolkit headless service

Use:

```text
PICO
-> XRoboToolkit
-> xrobotoolkit_sdk
-> motion_tracking_retarget
-> G1 qpos
```

Ordinary Sim2Real does not need `xrobotoolkit_sdk`, XRoboToolkit service, PICO,
or retargeted qpos. This dependency is not the policy runtime and is not the G1
control binding.

### C. Optional Diagnostics

Optional:

- `unitree_sdk2py`

Source:

```text
https://github.com/unitreerobotics/unitree_sdk2_python
```

Use only:

```text
scripts/onboard/check_g1_state_readonly.py
```

This optional dependency is for readonly low-state subscriber checks: `q`, `dq`,
IMU, and wireless remote state. It is not used by UFO policy control. Missing
`unitree_sdk2py` does not block ordinary Sim2Real or PICO teleop.

Set up ordinary onboard Sim2Real in this order:

1. Create a Python 3.10 venv.
2. Install ARM64 runtime dependencies from `requirements/runtime.txt` or an
   onboard wheelhouse. Ordinary Sim2Real does not need human pose,
   XRoboToolkit, `xrobotoolkit_sdk`, PICO, retargeting, `unitree_sdk2py`, GMR,
   or torch.
3. Restore/download model artifacts under `model/g1_policy/`.
4. Verify artifacts:

   ```bash
   python scripts/onboard/check_deploy_artifacts.py
   ```

5. Install or expose the CPython 3.10 aarch64 `g1_interface` binding in the
   runtime environment.
6. Configure the low-level G1 DDS interface explicitly:

   ```bash
   export G1_INTERFACE=<low-level-dds-interface>
   ```

7. If the current prebuilt `g1_interface` needs OpenSSL 1.1, provide:

   ```bash
   export OPENSSL11_LIB=/path/to/openssl-1.1/lib
   ```

8. Run no-actuation ordinary preflight:

   ```bash
   scripts/onboard/run_preflight_suite.sh --profile ordinary
   ```

For onboard PICO teleop, start from the ordinary runtime above, then install the
ARM64 XRoboToolkit SDK and teleop dependencies. `requirements/teleop.txt`
inherits `requirements/runtime.txt` because onboard teleop also runs realtime
`z`, ONNX inference, policy code, and G1 runtime:

   ```bash
   python -m pip install -r requirements/teleop.txt
   scripts/onboard/install_xrobot_sdk.sh \
     --sdk-root /path/to/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64 \
     --venv /path/to/ufo_deploy_venv
   ```

Start/verify the XRoboToolkit service, then run:

   ```bash
   scripts/onboard/run_preflight_suite.sh --profile teleop
   scripts/onboard/run_preflight_suite.sh --profile teleop --require-body
   ```

Optional readonly diagnostics additionally require `unitree_sdk2py`:

```bash
scripts/onboard/run_preflight_suite.sh --profile diagnostic
```

Run ordinary onboard Sim2Real or onboard PICO teleop Sim2Real only after the
physical robot safety setup is ready.

## Deployment Dependencies

Current deploy does not require the external `general_motion_retargeting`/GMR
package. Ordinary onboard Sim2Real has no retargeting dependency. Onboard PICO
teleop Sim2Real uses XRoboToolkit body joints, the vendored
`scripts/teleop/motion_tracking_retarget/` package, and Mink IK.

The legacy GMR architecture is no longer used by the deploy runtime:

```text
general_motion_retargeting installed: no
GMR required: no
torch required for teleop retargeting: no
```

Old GMR workspaces and `general_motion_retargeting` are not part of this deploy
path. Online teleop retargeting is provided by:

```text
scripts/teleop/motion_tracking_retarget/
```

`qpsolvers` and `daqp` are Mink solver dependencies, not legacy GMR
dependencies. See `docs/deployment_dependencies.md` for the dependency matrix.

## Diagnostics

Run from the repository root with the onboard venv:

```bash
source /path/to/ufo_deploy_venv/bin/activate
```

Recommended profiled suites:

```bash
scripts/onboard/run_preflight_suite.sh --profile ordinary
scripts/onboard/run_preflight_suite.sh --profile teleop
scripts/onboard/run_preflight_suite.sh --profile diagnostic
scripts/onboard/run_preflight_suite.sh --profile all
```

Individual ordinary control checks:

```bash
python scripts/onboard/check_g1_onboard_env.py --g1-interface "$G1_INTERFACE"
python scripts/onboard/check_deploy_artifacts.py
python scripts/onboard/check_policy_preflight.py --task config/exp/tracking/tracking.yaml
```

Individual onboard PICO teleop checks:

```bash
python scripts/onboard/check_g1_onboard_env.py --profile teleop --g1-interface "$G1_INTERFACE"
python scripts/onboard/check_policy_preflight.py --task config/exp/tracking/teleop.yaml
python scripts/onboard/check_xrobot_sdk.py --duration 5
python scripts/onboard/check_xrobot_sdk.py --duration 5 --require-body
python scripts/onboard/check_teleop_qpos.py --duration 5 --min-valid 10
python scripts/onboard/check_z_stream.py --addr tcp://127.0.0.1:28711 --duration 3
```

Individual optional diagnostic checks:

```bash
python scripts/onboard/check_g1_onboard_env.py --profile diagnostic
```

`check_g1_state_readonly.py` is optional and subscribes only to live low state:

```bash
export CYCLONEDDS_HOME=/path/to/cyclonedds/install
export OPENSSL11_LIB=/path/to/openssl-1.1/lib  # only if libddsc/g1_interface needs OpenSSL 1.1
export LD_LIBRARY_PATH="$OPENSSL11_LIB:$CYCLONEDDS_HOME/lib:${LD_LIBRARY_PATH:-}"

UNITREE_SDK_PYTHON=/path/to/unitree_sdk2_python \
G1_INTERFACE="$G1_INTERFACE" \
python scripts/onboard/check_g1_state_readonly.py --duration 3
```

It does not import `g1_interface`, create a command publisher, set PR mode, or
write commands. If `cyclonedds` is not installed in the venv, install it against
the same native CycloneDDS runtime used by the robot:

```bash
CYCLONEDDS_HOME=/path/to/cyclonedds/install \
python -m pip install cyclonedds==0.10.2
```

`check_xrobot_sdk.py --require-body` fails when XRoboToolkit is reachable but
full-body data is not streaming from the PICO app.

`check_teleop_qpos.py` expects `scripts/teleop/teleop_pose_50hz_onboard.sh` to
already be running. It validates finite 36-D G1 qpos frames from live retargeted
body data. Use `--allow-fallback` only when explicitly testing the static
fallback path.

`check_policy_preflight.py` validates policy/task/model shapes without
instantiating the real robot policy class, `G1Interface`, or command sender.

`run_preflight_suite.sh` runs safe checks by dependency profile. It never starts
robot control, teleop bridge, or realtime z server.

```bash
scripts/onboard/run_preflight_suite.sh --profile ordinary
scripts/onboard/run_preflight_suite.sh --profile teleop
scripts/onboard/run_preflight_suite.sh --profile diagnostic
scripts/onboard/run_preflight_suite.sh --profile all
```

`ordinary` checks Python version, runtime imports, artifacts, ONNX files,
`g1_interface`, and the robot interface. It skips `xrobotoolkit_sdk`, PICO, and
`unitree_sdk2py`.

`teleop` includes ordinary checks and adds `xrobotoolkit_sdk`, XRoboToolkit
service/API checks, vendored `motion_tracking_retarget`, and an already-running
qpos stream check.

`diagnostic` checks the optional `unitree_sdk2py` low-state subscriber path. It
does not require `g1_interface` or `xrobotoolkit_sdk`.

`all` runs all profiles. If `G1_INTERFACE` is set for a control profile, the
suite also checks launcher `--help` with interface/OpenSSL resolution while
still avoiding policy construction.

## Real G1 Validation

The validated d92 deployment was user-confirmed on a real Unitree G1:

- ordinary onboard Sim2Real ran on the real G1;
- onboard PICO teleop Sim2Real ran on the real G1;
- the PICO/XRobot to retarget qpos to realtime z to UFO policy to real G1
  actuation path was validated;
- no obvious functional issue was observed in the user-confirmed real-robot
  test.

This does not claim that R2 fault injection, physical e-stop injection, PICO
disconnect, process kill behavior, long-duration free walking, or quantified
impact limits were systematically tested.

See `DEPLOYMENT_STATUS_20260725.md` for the current evidence summary and
remaining validation boundaries.
