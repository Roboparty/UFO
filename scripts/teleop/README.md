# Teleop Bridge Setup Guide

This guide covers the PICO/XRoboToolkit canonical teleoperation bridge used by UFO-Deploy.
It is written for two deployment modes:

- workstation teleop, where the PICO streams to an Ubuntu PC
- G1 onboard teleop, where the PICO streams directly to the Unitree G1 onboard Jetson

The PC-side teleop chain is already validated in UFO-Deploy. The onboard flow uses the
same pose bridge, realtime `z` server, backward encoder, and UFO policy interfaces; only
the teleop host moves from the workstation to the robot.

## What This Folder Owns

The scripts in this folder only do the PICO-to-G1-pose part of the stack:

1. read live body, headset, and controller data from XRoboToolkit,
2. retarget the human motion to Unitree G1 with UFO's vendored `motion_tracking_retarget`,
3. publish retargeted G1 pose frames over ZMQ for `scripts/realtime/realtime_z_server.py`.

They do not start the realtime latent `z` encoder and they do not start the UFO policy.
The runtime split is:

```text
scripts/teleop/teleop_pose_50hz_onboard.sh:
  PICO -> XRoboToolkit -> xrobotoolkit_sdk -> motion_tracking_retarget -> ZMQ pose server

scripts/realtime/realtime_z_server.py:
  ZMQ pose -> backward_encoder.onnx -> latent z

run_g1_teleop_policy_onboard.sh:
  latent z -> UFO policy inference -> G1 command
```

Included files:

- `xrobot_teleop_to_pose_zmq_server.py`
- `default_mimic_obs.py`
- `teleop_pose_50hz.sh`
- `teleop_pose_50hz_onboard.sh`
- `check_teleop_env.py`

## Runtime Modes

### Workstation Teleop

Use this mode for local teleop sim2sim and split workstation/robot debugging. The PICO
connects to the PC, so the teleop host is the workstation.

```text
PICO
 |
 XRoboToolkit
 |
 PC
 |
 motion_tracking_retarget
 |
 xrobot_teleop_to_pose_zmq_server.py
 |
 realtime_z_server.py
 |
 UFO policy
```

Run the bridge with:

```bash
cd "$UFO_ROOT"
conda activate ufo-teleop
scripts/teleop/teleop_pose_50hz.sh
```

### G1 Onboard Teleop

Use this mode for direct PICO-to-robot teleop. The PICO connects to the G1 Jetson IP, so
the teleop host moves from the PC to the G1 onboard computer.

```text
PICO XRoboToolkit client
 |
 WiFi / LAN target IP = <G1_JETSON_IP>
 |
 G1 Jetson / onboard computer
 |
 XRoboToolkit headless service
 |
 xrobotoolkit_sdk
 |
 motion_tracking_retarget
 |
 xrobot_teleop_to_pose_zmq_server.py
 |
 realtime_z_server.py
 |
 UFO policy
```

Run the pose bridge on the robot with:

```bash
cd /home/unitree/UFO-Deploy
scripts/teleop/teleop_pose_50hz_onboard.sh
```

## Required Components

Install these components on the machine that receives the PICO stream. For workstation
teleop this is the Ubuntu PC. For onboard teleop this is the G1 Jetson.

- XRoboToolkit PICO app, installed on the headset
- XRoboToolkit PC Service on x86 Ubuntu, or XRoboToolkit headless service on G1 Jetson
- `xrobotoolkit_sdk` Python binding
- vendored `motion_tracking_retarget` code and G1 assets included in UFO-Deploy
- `mink`, `mujoco`, `numpy`, `scipy`, `pyyaml`
- `pyzmq`

Important: `xrobotoolkit_sdk` is only the Python binding. It depends on the XRoboToolkit
system service already running on the host. Installing the Python package alone does not
start or replace the XRoboToolkit service.

The upstream teleop dependencies are pinned in
[requirements-lock.md](requirements-lock.md). Use those commits for release deployments so
canonical retarget assets, XRoboToolkit SDK ABI, and `xrobotoolkit_sdk` APIs do not
silently drift with upstream `main` or `master`. Users only need to clone UFO-Deploy; no
runtime clone of `motion_tracking` is required.

## Python Environment

Create a Python 3.10 environment on the teleop host:

```bash
conda create -n ufo-teleop python=3.10 -y
conda activate ufo-teleop
cd "$UFO_ROOT"
python -m pip install -r requirements/teleop.txt
```

Use this `ufo-teleop` environment for `scripts/teleop/teleop_pose_50hz.sh` and the onboard
teleop bridge. Keep the main `ufo-deploy` environment for MuJoCo, realtime `z`, and policy
inference. `requirements/teleop.txt` inherits `requirements/runtime.txt` so a
direct onboard teleop venv also has the realtime `z`, ONNX inference, policy,
and G1 runtime dependencies.

On the G1 Jetson you can also use a venv, for example:

```bash
python3 -m venv /home/unitree/ufo_teleop_venv
source /home/unitree/ufo_teleop_venv/bin/activate
pip install --upgrade pip
cd /home/unitree/UFO-Deploy
python -m pip install -r requirements/teleop.txt
```

If the XRoboToolkit binding ships a native `.so`, make sure its directory is in
`LD_LIBRARY_PATH` before importing `xrobotoolkit_sdk`.

Use a workspace for external XRoboToolkit service/source checkouts:

```bash
export TELEOP_WORKSPACE=${TELEOP_WORKSPACE:-$HOME/teleop_ws}
mkdir -p "$TELEOP_WORKSPACE"
```

## Canonical Retarget Assets

UFO-Deploy vendors the minimal motion_tracking retarget path under:

```text
scripts/teleop/motion_tracking_retarget/
```

This includes the canonical XRobot parser, G1 Mink IK retargeter, `xrobot_to_g1.json`,
G1 XML, and required mesh assets. The canonical boundary is `PICO/XRobot raw body data ->
G1 qpos`; UFO's realtime `z`, backward encoder, policy, and robot command path remain UFO
code.

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

## Verify The Environment

Run the automated preflight checker from the UFO-Deploy checkout:

```bash
cd "$UFO_ROOT"
python scripts/teleop/check_teleop_env.py
```

Expected healthy preflight output includes lines like:

```text
[OK] mink installed
[OK] mujoco installed
[OK] numpy installed
[OK] scipy installed
[OK] pyyaml installed
[OK] xrobotoolkit_sdk installed
[OK] pyzmq installed
[OK] motion_tracking_retarget package available
[OK] canonical G1 XML loads with MuJoCo
[OK] xrobot_to_g1.json required fields present
[OK] canonical -> UFO joint permutation valid
[OK] XRoboToolkit service running
[OK] port 28701 available
[OK] port 28702 available
[OK] port 28703 available
```

Before starting the teleop bridge, ports `28701`, `28702`, and `28703` should normally be
available. If a previous teleop process is still running, the checker will report the
occupied port so you can stop the stale process.

The canonical retarget check is more than an import check. It verifies the vendored
24-joint XRobot names, G1 XML, `xrobot_to_g1.json`, toe bodies, canonical 29-joint order,
UFO output joint order, qpos size, and joint-name permutation before
`xrobot_teleop_to_pose_zmq_server.py` starts its retarget worker.

To check the realtime `z` server port separately, use the realtime profile:

```bash
python scripts/teleop/check_teleop_env.py --port-profile realtime
```

This checks port `28711`, which belongs to `scripts/realtime/realtime_z_server.py`, not to
the teleop bridge.

After the teleop bridge and realtime `z` server are already running, use:

```bash
python scripts/teleop/check_teleop_env.py --mode running --port-profile all
```

That mode expects the teleop and realtime ZMQ ports to be occupied by the running services.

## Verify Python Imports

The minimum import check is:

```bash
python - <<'PY'
import mink
import mujoco
import numpy
import scipy
import yaml
import xrobotoolkit_sdk
import zmq
import motion_tracking_retarget
print("canonical teleop deps: OK")
print("xrobotoolkit_sdk: OK")
PY
```

If this fails, fix the Python environment before starting the PICO stream.

## Verify XR Data

Start the XRoboToolkit service, connect the PICO app to the host IP, calibrate trackers,
and start body/controller streaming. Then run:

```bash
python scripts/teleop/check_teleop_env.py --xr-data
```

The checker attempts to verify:

- body data
- headset pose
- left controller pose
- right controller pose

You can also run the direct SDK check:

```bash
python - <<'PY'
import xrobotoolkit_sdk as xrt

xrt.init()
print("Body data available:", xrt.is_body_data_available())
print("Headset pose:", xrt.get_headset_pose())
print("Left controller pose:", xrt.get_left_controller_pose())
print("Right controller pose:", xrt.get_right_controller_pose())
if hasattr(xrt, "close"):
    xrt.close()
PY
```

If body data is not available, check that:

- the PICO is connected to the correct host IP,
- trackers/controllers are paired and calibrated,
- full-body streaming is enabled in the PICO XRoboToolkit app,
- firewalls, VPN/TUN interfaces, or proxy settings are not blocking the local connection.

## ZMQ Ports

UFO-Deploy uses these local ZMQ ports:

| Port | Owner | Purpose |
| --- | --- | --- |
| 28701 | teleop bridge | pose request socket from `realtime_z_server.py` |
| 28702 | teleop bridge | pose reply socket to `realtime_z_server.py` |
| 28703 | teleop bridge | PICO button/control channel to `realtime_z_server.py` |
| 28704 | teleop bridge | legacy/debug optional PICO button PUB channel to onboard policy |
| 28711 | realtime `z` server | latent `z` PUB channel to policy |

The preflight checker uses the `teleop` port profile by default, so it checks only
`28701`, `28702`, and `28703`. Use `--port-profile realtime` to check `28711`, or
`--port-profile all` to check both groups. Port `28704` is off by default. It is checked
only when `CTRL_PUB_BIND_ADDR` is explicitly set. The onboard launchers require
`ENABLE_PICO_POLICY_CONTROL=1` alongside that address before enabling legacy/debug PICO
policy-control compatibility.

## Run The Teleop Bridge

### Workstation

```bash
cd "$UFO_ROOT"
conda activate ufo-teleop
scripts/teleop/teleop_pose_50hz.sh
```

### G1 Onboard

```bash
cd /home/unitree/UFO-Deploy
source /home/unitree/ufo_teleop_venv/bin/activate
scripts/teleop/teleop_pose_50hz_onboard.sh
```

Useful onboard environment variables:

```bash
export UFO_ROOT=/home/unitree/UFO-Deploy
export TELEOP_PY=/home/unitree/ufo_teleop_venv/bin/python
export WEB_VISUALIZE=0
export WEB_PORT=8080
```

The onboard launcher automatically derives `UFO_ROOT` from its own path if `UFO_ROOT` is
not set. It checks the Python executable, `mink`, `mujoco`, `numpy`, `scipy`, `pyyaml`,
`pyzmq`, `xrobotoolkit_sdk`, the vendored canonical retarget assets, XRoboToolkit service,
and required local ZMQ ports before starting `xrobot_teleop_to_pose_zmq_server.py`.
Current deploy does not require the external `general_motion_retargeting`/GMR
package. The legacy GMR architecture is no longer used by PICO teleop Sim2Real.
Old GMR workspaces are not part of this deploy path.

PICO teleop Sim2Real uses XRoboToolkit body joints, the vendored
`scripts/teleop/motion_tracking_retarget/` package, and Mink IK. Torch is not a
teleop retarget dependency. `qpsolvers` and `daqp` are Mink dependencies, not
legacy GMR dependencies.
The web viewer is off by default; enable it only for debugging:

```bash
WEB_VISUALIZE=1 scripts/teleop/teleop_pose_50hz_onboard.sh
```

It does not auto-start the XRoboToolkit service by default. Start the service manually
first so version, library, and runtime failures are visible before real-robot bring-up.
If you have already verified the installed service and want the launcher to start it for a
debug session, run:

```bash
START_XROBOT_SERVICE=1 scripts/teleop/teleop_pose_50hz_onboard.sh
```

The teleop bridge defaults come from `config/teleop/g1.yaml`. The shell launchers only
override values such as `ACTUAL_HUMAN_HEIGHT`, `LOOKBACK_MS`, `CTRL_FPS`,
`RETARGET_BUFFER_WINDOW_S`, `LOG_INTERVAL_S`, and `VIS_FPS` when the corresponding
environment variable is explicitly set. `TELEOP_POLICY_CONFIG` defaults to
`POLICY_CONFIG` and then to `config/policy/g1_policy.yaml`; it is passed to the teleop
server so the qpos joint permutation is checked against the same `policy_joint_names`
used by policy inference. `server.visualize` in `config/teleop/g1.yaml` is the web viewer
default, and `WEB_VISUALIZE=1` still enables it from the launcher.

It probes UFO-specific environments in this order:

```text
${UFO_ROOT}/.venv/bin/python
${UFO_ROOT}/venv/bin/python
/home/unitree/ufo_teleop_venv/bin/python
/home/unitree/ufo_deploy_venv/bin/python
/home/unitree/miniconda3/envs/ufo-teleop/bin/python
/home/unitree/miniconda3/envs/ufo-deploy/bin/python
```

The launcher probes these candidates in order with
`check_teleop_env.py --skip-service --skip-ports`. It selects the first Python that can
import the canonical retarget dependencies, `xrobotoolkit_sdk`, and `zmq`, has a compatible
`xrobotoolkit_sdk` API, and can load the vendored G1 XML/assets. A Python
executable that exists but cannot run the teleop bridge is skipped so a later valid teleop
environment can be used.

If none of those candidates passes, set `TELEOP_PY` explicitly. The launcher refuses to
fall back to system `python3` by default because the Jetson system Python usually does not
contain `mink`, `mujoco`, `xrobotoolkit_sdk`, and `numpy`. Set `TELEOP_ALLOW_SYSTEM_PY=1`
only for manual debugging; even with this flag, system Python must pass the same import
probe.

During the first few seconds after starting the bridge, stand in a stable neutral posture.
The bridge uses startup foot height to align the z-axis offset; starting from a crouched or
moving pose can affect gait quality.

## Complete Onboard Startup Order

Run these steps on the G1 Jetson:

1. Start XRoboToolkit headless service.

   ```bash
   bash /opt/apps/roboticsservice/runService.sh
   ```

   This receives the PICO stream on the robot.
   The teleop launcher expects this service to already be running. Automatic service start
   is opt-in with `START_XROBOT_SERVICE=1` and should be used only after the installed
   headless service has been verified.

   In the PICO XRoboToolkit app, set the target IP to the G1 onboard computer IP reported
   by `ip -br addr`. In onboard mode the PICO connects to the robot, not to the PC.

2. Start the teleop pose bridge.

   ```bash
   cd /home/unitree/UFO-Deploy
   scripts/teleop/teleop_pose_50hz_onboard.sh
   ```

   This converts PICO body/controller data into retargeted G1 poses on ZMQ ports
   `28701`, `28702`, and `28703`. The legacy/debug `28704` PICO policy-control PUB channel
   is disabled by default.

3. Start the realtime latent `z` server.

   ```bash
   cd /home/unitree/UFO-Deploy
   Z_PY=/home/unitree/ufo_deploy_venv/bin/python \
     scripts/realtime/run_realtime_z_server_onboard.sh
   ```

   This requests poses from the teleop bridge, runs `backward_encoder.onnx`, and publishes
   latent `z` on port `28711`. The wrapper verifies that the selected deployment Python
   can import `numpy`, `mujoco`, `onnxruntime`, and `zmq` before starting. The onboard
   default is `INITIAL_MODE=freeze`, so no live PICO reference is followed until PICO
   right-hand A switches realtime `z` to follow mode.

4. Start the G1 teleop policy.

   ```bash
   cd /home/unitree/UFO-Deploy
   source /home/unitree/ufo_deploy_venv/bin/activate
   G1_INTERFACE=<low-level-dds-interface> \
   UFO_REAL_ROBOT_OK=1 VENV_PATH=/home/unitree/ufo_deploy_venv/bin/activate \
     ./run_g1_teleop_policy_onboard.sh
   ```

   This subscribes to realtime `z` and runs UFO policy inference.
   The G1 DDS interface must be explicit. The launcher validates the interface
   and can generate a temporary robot-config overlay instead of committing a
   machine-specific `INTERFACE` value.

5. Use the G1 wireless remote to manage robot and policy state.

   ```text
   G1 A    interpolate to default standing pose
   G1 R1   enable policy action
   G1 B    start tracking
   G1 X    reset tracking/reference
   G1 R2   global stop latch
   ```

6. Use PICO only for the live motion reference stream.

   ```text
   PICO right_key_one / right-hand A   follow or resume live reference
   PICO left_key_one / left-hand X     freeze current reference/z
   ```

   On freeze -> follow resume, realtime `z` resets its previous-pose velocity history.
   The first live frame initializes history while holding the frozen `z`, then the server
   blends from the old `z` to the new live `z` over `RESUME_RAMP_MS` milliseconds.

   PICO buttons do not enable policy, clear R2, enter default pose, reset the real policy
   state machine, or bypass the physical e-stop in the default flow.

The G1 wireless remote controls robot and policy state. PICO controls only the
live reference follow/freeze state for realtime `z`.

Legacy/debug PICO policy-control compatibility is available only when both launchers are
explicitly opted in:

```bash
ENABLE_PICO_POLICY_CONTROL=1 CTRL_PUB_BIND_ADDR=tcp://*:28704 \
  scripts/teleop/teleop_pose_50hz_onboard.sh
ENABLE_PICO_POLICY_CONTROL=1 CTRL_PUB_BIND_ADDR=tcp://*:28704 \
  ./run_g1_teleop_policy_onboard.sh
```

Keep the robot on support for first bring-up and test the physical e-stop, wireless R2
stop latch, stale-teleop watchdog, and PICO disconnect behavior before free walking.
