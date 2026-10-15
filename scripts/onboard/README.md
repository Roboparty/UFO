# G1 onboard setup

Run tracking, goal, reward and teleop policies on the G1 Jetson. Teleop sim2real
is onboard-only: the pose bridge, z encoder and policy run on this computer.
See the [main commands](../../README.md#sim2real--onboard) after setup.

## Python and native libraries

Use a Python 3.10 venv on the robot. Workstation simulation uses Conda.

```bash
python3.10 -m venv /home/unitree/ufo_deploy_venv
source /home/unitree/ufo_deploy_venv/bin/activate
python -m pip install -r requirements/runtime.txt
```

For teleop, install `requirements/teleop.txt` instead; it includes runtime
requirements. Follow the [teleop guide](../teleop/README.md) for XRoboToolkit.

| Native dependency | Needed for |
|---|---|
| `g1_interface`, CycloneDDS | All real G1 control |
| `xrobotoolkit_sdk`, XRoboToolkit service | Teleop only |
| `unitree_sdk2py` | Optional read-only state diagnostics |

Ordinary tracking, goal and reward deployment need no PICO, retargeter or Torch.

The control binding comes from [EGalahad/unitree_sdk2](https://github.com/EGalahad/unitree_sdk2).
Use a binding built for the robot's Python and architecture:
`g1_interface.cpython-310-aarch64-linux-gnu.so`.
Expose the received state tick before rebuilding an existing SDK checkout:

```bash
python scripts/onboard/patch_g1_sdk.py /home/unitree/unitree_sdk2_bfm
cmake --build /home/unitree/unitree_sdk2_bfm/build --target g1_interface -j2
```

The preflight checks `g1_interface.LowState().tick` without opening hardware.
The runtime refuses old bindings without this field. It detects cached state
through the tick; packets older than the configured timeout latch a stop.

`scripts/onboard/run_policy.sh` prepares library paths before invoking the shared
`deploy.py` entry. Its defaults are:

```text
CYCLONEDDS_HOME=/home/unitree/cyclonedds_ws/install/cyclonedds
UNITREE_SDK_LIB=/home/unitree/unitree_sdk2_bfm/build/lib
UNITREE_SDK_THIRDPARTY_LIB=/home/unitree/unitree_sdk2_bfm/thirdparty/lib/aarch64
CUDA_LIB=/usr/local/cuda-11.4/lib64
```

Override those variables for your installation. The launcher tries the binding
first and only adds `OPENSSL11_LIB` if it reports missing `libssl.so.1.1` or
`libcrypto.so.1.1`; the default is `external/openssl-1.1-aarch64` under this repo.
Keep Python, architecture and native libraries matched; do not replace system
libc/libstdc++ to force an incompatible binary to load.

## Interface and artifacts

Find the low-level DDS NIC:

```bash
ip -br addr
export G1_INTERFACE=eth0  # Use the actual robot control NIC.
python scripts/onboard/interface_config.py --interface "$G1_INTERFACE"
```

The interface must exist, be UP, have IPv4, and normally must not be the default
route used by Wi-Fi/PICO. An intentional exception requires
`--allow-default-route-interface` on `deploy.py`, or
`ALLOW_DEFAULT_ROUTE_INTERFACE=1` on the onboard launcher.

Copy your model, `.meta.json`, `.deploy.yaml` and task z files as shown in the
[root README](../../README.md#model-and-task-files). Check each task you will run:

```bash
python scripts/onboard/check_policy_preflight.py --task config/tasks/tracking.yaml
python scripts/onboard/check_policy_preflight.py --task config/tasks/goal.yaml
python scripts/onboard/check_policy_preflight.py --task config/tasks/reward.yaml
python scripts/onboard/check_policy_preflight.py --task config/tasks/teleop.yaml
```

For a custom export, pass `--model-path /path/to/Policy.onnx`; preflight reads its
sibling `.deploy.yaml` automatically. Give `--task` a custom YAML when selecting
different motion, reward or goal names. `check_deploy_artifacts.py` additionally
checks the release artifact manifest when using that layout.

## Start policy

After model and interface checks, keep the robot supported and the physical
stop available:

```bash
export UFO_REAL_ROBOT_OK=1
scripts/onboard/run_policy.sh --task tracking
```

Replace `tracking` with `goal`, `reward` or `teleop`. For teleop, start the
[onboard pose bridge and encoder](../teleop/README.md#startup) first. No mode is
automatically enabled: the G1 remote controls standing and policy activation.
Remote/button meanings are listed in the [main README](../../README.md#sim2real--onboard).

Pass `--model`, `--task-config`, `--policy-config`, or `--robot-config` directly to
the launcher. `MODEL_PATH`, `POLICY_CONFIG`, `ROBOT_CONFIG`, `TASK_CONFIG` and
`TASK` also set launcher defaults. `VENV_PATH=/path/to/bin/activate` optionally
activates a venv; otherwise the launcher uses the current Python.

## Diagnostics

Environment checks do not instantiate the control interface:

```bash
python scripts/onboard/check_g1_onboard_env.py --target g1-onboard
python scripts/onboard/check_g1_onboard_env.py --profile teleop --target g1-onboard
```

The suite uses `/home/unitree/ufo_deploy_venv/bin/python` by default; override
`ONBOARD_PY` for another venv. It never sends joint commands. Teleop checks may
connect to XRoboToolkit and query an already-running pose bridge.

```bash
scripts/onboard/run_preflight_suite.sh --profile ordinary
scripts/onboard/run_preflight_suite.sh --profile teleop
scripts/onboard/run_preflight_suite.sh --profile diagnostic
```

The diagnostic profile needs the optional `unitree_sdk2py` package. Without it,
ordinary control and teleop remain available. Individual stream checks:

```bash
python scripts/onboard/check_xrobot_sdk.py --no-init
python scripts/onboard/check_teleop_qpos.py --duration 3 --min-valid 1
python scripts/onboard/check_z_stream.py --addr tcp://127.0.0.1:28711 --duration 3
python scripts/onboard/check_g1_state_readonly.py --interface "$G1_INTERFACE" --duration 3
```

Regression tests use mocked hardware. Real e-stop, state-loss, PICO-disconnect
and free-standing behavior still require validation on the actual robot.
