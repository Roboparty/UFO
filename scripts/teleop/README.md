# PICO teleop

Teleop has two supported placements. All UFO processes run on the same host:

| Mode | PICO target | Retarget, z encoder and policy |
|---|---|---|
| sim2sim | Workstation IP | Workstation; MuJoCo also runs here |
| sim2real | G1 onboard IP | G1 onboard computer |

The pose bridge reads XRoboToolkit, retargets to G1 using the vendored
`motion_tracking_retarget`, and serves poses over local ZMQ. The encoder turns
poses into z; `deploy.py --task teleop` feeds live z to the shared actor.

## Environment

Use Python 3.10 Conda on the workstation; use a Python 3.10 venv on G1.
The onboard default is `/home/unitree/ufo_deploy_venv/bin/python`.

```bash
pip install -r requirements/teleop.txt
export TELEOP_WORKSPACE=${TELEOP_WORKSPACE:-$HOME/teleop_ws}
mkdir -p "$TELEOP_WORKSPACE"
```

This requirements set includes policy/runtime dependencies. Install the native
XRoboToolkit service and `xrobotoolkit_sdk` separately below. The service must
be running; importing the SDK alone does not start it. Pinned upstream commits
and assets are recorded in [requirements-lock.md](requirements-lock.md).

## Install XRoboToolkit Service

### x86 Ubuntu Workstation

Install the XRoboToolkit PC Service on the workstation that the PICO connects to. A common
path is the Ubuntu `.deb` package from the XRoboToolkit PC Service release page:

```bash
cd "$TELEOP_WORKSPACE"
wget https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
sudo dpkg -i XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
```

Start the PC service from the Ubuntu application launcher, or with the service start
command provided by your XRoboToolkit installation.

### G1 Jetson Arm64 Headless

For onboard teleop, install the arm64/headless XRoboToolkit service on the G1 Jetson. The
commands below use the official XR-Robotics v1.0.0 release asset:

```bash
cd "$TELEOP_WORKSPACE"
wget -O XRoboToolkit-PC-Service-headless_1.0.0.0_arm64.deb \
  https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit-PC-Service-headless_1.0.0.0_arm64.deb
echo "532c605dfa1a02b05b7c285b856c91771c78623cded30ef5b16ea371de49ed5f  XRoboToolkit-PC-Service-headless_1.0.0.0_arm64.deb" | sha256sum -c -
sudo dpkg -i XRoboToolkit-PC-Service-headless_1.0.0.0_arm64.deb
```

After installation the G1 should have this headless service launcher:

```text
/opt/apps/roboticsservice/runService.sh
```

Start the service before launching the UFO teleop bridge:

```bash
bash /opt/apps/roboticsservice/runService.sh
```

Confirm that an XRoboToolkit/RoboticsService process is running:

```bash
pgrep -af 'RoboticsServiceProcess|roboticsservice|XRoboToolkit'
```

Compatibility note: this package is an upstream prebuilt binary. UFO-Deploy cannot
guarantee that it is compatible with every Unitree Jetson Ubuntu image. If installation or
startup fails with `GLIBC`, `GLIBCXX`, Qt, or QML library errors, do not upgrade the
robot's system libc/libstdc++ to force compatibility. Use a service release built for the
robot OS, or build XRoboToolkit-PC-Service on the target system or in a container matching
the target Ubuntu image.

Upstream source-build entry point for arm64 is `RoboticsService/qt-gcc_aarch64.sh`, which
requires a matching Qt installation path before running:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git pkg-config
git clone --branch main --single-branch https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git
cd XRoboToolkit-PC-Service
git checkout 85bac4dbc1fd5cef42c74a160d9c30aa3491f122
# Edit RoboticsService/qt-gcc_aarch64.sh so QT_GCC_ARM64, QT6_TOOLS, and every
# hard-coded /media/bytedance/... PATH entry point to the G1 Qt/toolchain install.
# The upstream script checks for Qt 6.6.2; confirm the Qt version and toolchain are
# compatible with the target Unitree Ubuntu image before building.
bash RoboticsService/qt-gcc_aarch64.sh
```

Package the resulting arm64 build with the upstream `RoboticsService/Package/debPackAArch64`
scripts if you need an installable `.deb`.

## Install And Prepare The PICO App

Install the XRoboToolkit PICO-side app from:

- `https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/`

Prepare the headset and trackers:

1. Put on the motion trackers.
2. Put the controllers on the wrists.
3. Start VR on the headset.
4. Calibrate the whole-body motion tracking.
5. Open the XRoboToolkit / XRobot app on the headset.
6. Connect the app to the teleop host IP.
7. Start body data streaming.
8. Start headset/controller pose streaming.

Use the workstation IP for workstation teleop. Use the G1 Jetson IP for onboard teleop.
The PICO and teleop host must be on a reachable network. If data does not arrive, check
firewalls, VPN/TUN interfaces, proxy settings, and whether the PICO can route to the host
IP entered in the app.

## Onboard Network Target

For G1 onboard mode, set the XRoboToolkit PICO client target IP to the G1 onboard computer
IP. Do not use the workstation IP in this mode.

```text
PICO XRoboToolkit client
 |
 WiFi / LAN
 |
 target IP = <G1_JETSON_IP>
 |
 G1 Jetson XRoboToolkit headless service
```

Find the robot-side IP on the G1 Jetson:

```bash
ip -br addr
```

Choose the IP on the WiFi/LAN interface reachable from the PICO headset, then enter that IP
in the XRoboToolkit PICO app before starting body/controller streaming.

## Install xrobotoolkit_sdk

Build and install the Python binding in the same environment used by the bridge. Install
the system build dependencies first:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  git \
  python3-dev \
  pkg-config

cd "$TELEOP_WORKSPACE"
git clone https://github.com/Axellwppr/XRoboToolkit-PC-Service-Pybind
cd XRoboToolkit-PC-Service-Pybind
git checkout a7ae949849ef335f0fa7bbef1e741c5b35e2124e
```

On Ubuntu 20.04, `python3-dev` follows the system Python version. If your target teleop
Python is a non-Conda Python 3.10, install the matching headers as well, for example
`python3.10-dev`. Conda Python environments normally include matching headers.

Build the native SDK used by the binding:

```bash
cd "$TELEOP_WORKSPACE"
git clone --branch main --single-branch https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git
cd XRoboToolkit-PC-Service
git checkout 85bac4dbc1fd5cef42c74a160d9c30aa3491f122
cd RoboticsService/PXREARobotSDK
bash build.sh
```

Install the Python binding:

```bash
cd "$TELEOP_WORKSPACE/XRoboToolkit-PC-Service-Pybind"
python -m pip uninstall -y xrobotoolkit_sdk
python -m pip install .
```

`XRoboToolkit-PC-Service-Pybind` declares `pybind11` in `pyproject.toml`, so modern pip
installs the build dependency automatically. Do not invoke `setup.py` directly for new
environments.

If the XRoboToolkit-PC-Service source checkout is not next to the pybind repository, set
`PXREA_SDK_ROOT` explicitly:

```bash
export PXREA_SDK_ROOT=/abs/path/to/XRoboToolkit-PC-Service
python -m pip install .
```

Remember that this step installs the Python binding only. The XRoboToolkit service must be
installed and running separately.

## Startup

Commands below run from the repository root. Set the actor export used by all
three stages; the pose bridge reads its `.deploy.yaml` for joint order:

```bash
export MODEL_PATH=/path/to/exported/FBcprAuxModel.onnx
export BACKWARD_ONNX=/path/to/exported/backward_encoder.onnx
```

The default model directory is `model/g1_policy`. Set these variables in each
terminal when using another export. The actor and backward encoder must come
from the same checkpoint.

For **sim2sim**, start MuJoCo and then the following three processes in separate
workstation terminals with the appropriate Python environment activated:

```bash
# Terminal A
python -m sim_env.simulator
# Terminal B — PICO pose bridge
scripts/teleop/teleop_pose_50hz.sh
# Terminal C — live z (uses the active Python, or Z_PY)
scripts/realtime/run_z_encoder.sh
# Terminal D — policy
python deploy.py --mode sim2sim --task teleop --model "$MODEL_PATH"
```

For **sim2real**, run all three processes on the G1 after the
[onboard setup](../onboard/README.md). Start the XRoboToolkit headless service
and connect PICO to the robot IP first:

```bash
# Terminal A — onboard pose bridge
TELEOP_PY=/home/unitree/ufo_deploy_venv/bin/python scripts/teleop/teleop_pose_50hz_onboard.sh
# Terminal B — live z
Z_PY=/home/unitree/ufo_deploy_venv/bin/python scripts/realtime/run_z_encoder.sh
# Terminal C — policy, with the deployment venv activated
UFO_REAL_ROBOT_OK=1 G1_INTERFACE=eth0 scripts/onboard/run_policy.sh --task teleop
```

Use the actual robot DDS interface in place of `eth0`. The pose bridge uses
localhost ports 28701–28703; live z uses localhost 28711. The real policy
rejects a task config pointing to a remote z publisher.

The G1 remote owns standing, activation and stopping. PICO right-hand A
(`right_key_one`) follows/resumes the live reference; left-hand X
(`left_key_one`) freezes it. The z encoder starts frozen and blends over 500 ms
when resuming. PICO does not enable the robot policy in the default flow.
The optional legacy PICO policy-control channel stays disabled by default.

## Calibration and timing

Keep whole-body trackers calibrated and stand neutrally for the first few
frames. Startup foot-height alignment uses 10 frames; crouching or moving at
startup changes the estimated height offset. The default operator height is
1.75 m; set `ACTUAL_HUMAN_HEIGHT` to the operator's height when needed.

Retarget settings live in `config/teleop/g1.yaml`: 50 Hz control, 50 ms pose
lookback, 0.5 s pose buffer, and optional visualization. `ACTUAL_HUMAN_HEIGHT`,
`LOOKBACK_MS`, `CTRL_FPS`, `RETARGET_BUFFER_WINDOW_S`, `LOG_INTERVAL_S` and
`VIS_FPS` override their YAML values only when explicitly set. The encoder's
additional lookback defaults to zero; its validity timeout defaults to 200 ms.

`TELEOP_POLICY_CONFIG` overrides the pose bridge's joint-order config; otherwise
it uses `POLICY_CONFIG`, then the actor's `.deploy.yaml`. Preserve the matching
export's joint order, observation scales and metadata. `DEVICE=cpu` is the z
encoder default. CUDA requires a compatible ONNX Runtime GPU installation.

`WEB_VISUALIZE=1` enables the onboard retarget viewer on port 8080. The viewer's
floor is for display and does not affect FK, retargeting or policy control.
`START_XROBOT_SERVICE=1` is an optional service-start helper after the installed
service has been verified; by default the service must already be running.

## Diagnostics

```bash
python scripts/teleop/check_teleop_env.py
python scripts/onboard/check_xrobot_sdk.py --duration 3 --require-body
python scripts/onboard/check_teleop_qpos.py --duration 3 --min-valid 1
python scripts/onboard/check_z_stream.py --addr tcp://127.0.0.1:28711 --duration 3
```

Missing or stale poses stop valid z publication; stale z stops policy action.
These checks do not send joint commands. Test the real stop behavior with
support before free-standing operation.
