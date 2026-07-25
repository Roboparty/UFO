# Teleop Bridge Setup Guide

This guide covers the PICO/XRoboToolkit/GMR teleoperation bridge used by UFO-Deploy.
It is written for two deployment modes:

- workstation teleop, where the PICO streams to an Ubuntu PC
- G1 onboard teleop, where the PICO streams directly to the Unitree G1 onboard Jetson

The PC-side teleop chain is already validated in UFO-Deploy. The onboard flow uses the
same pose bridge, realtime `z` server, backward encoder, and UFO policy interfaces; only
the teleop host moves from the workstation to the robot.

## What This Folder Owns

The scripts in this folder only do the PICO-to-G1-pose part of the stack:

1. read live body, headset, and controller data from XRoboToolkit,
2. retarget the human motion to Unitree G1 with `general_motion_retargeting` (GMR),
3. publish retargeted G1 pose frames over ZMQ for `scripts/realtime/realtime_z_server.py`.

They do not start the realtime latent `z` encoder and they do not start the UFO policy.
The runtime split is:

```text
scripts/teleop/teleop_pose_50hz_onboard.sh:
  PICO -> XRoboToolkit -> xrobotoolkit_sdk -> GMR -> ZMQ pose server

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
 GMR
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
 GMR
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
- `general_motion_retargeting` (GMR)
- `pyzmq`

Important: `xrobotoolkit_sdk` is only the Python binding. It depends on the XRoboToolkit
system service already running on the host. Installing the Python package alone does not
start or replace the XRoboToolkit service.

## Python Environment

Create a Python 3.10 environment on the teleop host:

```bash
conda create -n ufo-teleop python=3.10 -y
conda activate ufo-teleop
pip install pyzmq
```

Use this `ufo-teleop` environment for `scripts/teleop/teleop_pose_50hz.sh` and the onboard
teleop bridge. Keep the main `ufo-deploy` environment for MuJoCo, realtime `z`, and policy
inference.

On the G1 Jetson you can also use a venv, for example:

```bash
python3 -m venv /home/unitree/ufo_teleop_venv
source /home/unitree/ufo_teleop_venv/bin/activate
pip install --upgrade pip
pip install pyzmq
```

If the XRoboToolkit binding ships a native `.so`, make sure its directory is in
`LD_LIBRARY_PATH` before importing `xrobotoolkit_sdk`.

## Install GMR

Install GMR into the same Python environment used by the teleop bridge:

```bash
export TELEOP_WORKSPACE=${TELEOP_WORKSPACE:-$HOME/teleop_ws}
mkdir -p "$TELEOP_WORKSPACE"
cd "$TELEOP_WORKSPACE"

git clone https://github.com/YanjieZe/GMR.git
cd GMR
pip install -e .
```

Verify:

```bash
python -c "import general_motion_retargeting; print('GMR OK')"
```

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
# Edit RoboticsService/qt-gcc_aarch64.sh so QT_GCC_ARM64 and QT6_TOOLS match the G1.
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
```

Build the native SDK used by the binding:

```bash
cd "$TELEOP_WORKSPACE"
git clone --branch main --single-branch https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git
cd XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK
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
[OK] GMR installed
[OK] GMR robot registered: unitree_g1
[OK] GMR robot XML exists: ...
[OK] GMR robot XML loads: unitree_g1
[OK] GMR IK config exists: xrobot -> unitree_g1: ...
[OK] GMR IK config valid: xrobot -> unitree_g1
[OK] xrobotoolkit_sdk installed
[OK] pyzmq installed
[OK] XRoboToolkit service running
[OK] port 28701 available
[OK] port 28702 available
[OK] port 28703 available
```

Before starting the teleop bridge, ports `28701`, `28702`, and `28703` should normally be
available. If a previous teleop process is still running, the checker will report the
occupied port so you can stop the stale process.

The GMR check is more than an import check. By default it verifies that the installed GMR
package has the `unitree_g1` robot XML, that the XML can be loaded by MuJoCo, and that the
`xrobot -> unitree_g1` IK config JSON exists and contains the required retargeting keys.
This catches incomplete GMR installs before `xrobot_teleop_to_pose_zmq_server.py` starts
its retarget worker.

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
import general_motion_retargeting
import xrobotoolkit_sdk
import zmq
print("general_motion_retargeting: OK")
print("xrobotoolkit_sdk: OK")
print("pyzmq: OK")
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
| 28704 | teleop bridge | optional PICO button PUB channel to onboard policy |
| 28711 | realtime `z` server | latent `z` PUB channel to policy |

The preflight checker uses the `teleop` port profile by default, so it checks only
`28701`, `28702`, and `28703`. Use `--port-profile realtime` to check `28711`, or
`--port-profile all` to check both groups. The onboard teleop launcher also checks `28704`
when the PICO policy-control PUB channel is enabled.

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
not set. It checks the Python executable, GMR, `xrobotoolkit_sdk`, XRoboToolkit service,
and required local ZMQ ports before starting `xrobot_teleop_to_pose_zmq_server.py`.
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
import `general_motion_retargeting`, `xrobotoolkit_sdk`, and `zmq`, has a compatible
`xrobotoolkit_sdk` API, and includes the GMR `xrobot -> unitree_g1` assets. A Python
executable that exists but cannot run the teleop bridge is skipped so a later valid teleop
environment can be used.

If none of those candidates passes, set `TELEOP_PY` explicitly. The launcher refuses to
fall back to system `python3` by default because the Jetson system Python usually does not
contain GMR, `xrobotoolkit_sdk`, `torch`, and `numpy`. Set `TELEOP_ALLOW_SYSTEM_PY=1` only
for manual debugging; even with this flag, system Python must pass the same import probe.

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
   `28701`, `28702`, `28703`, and optionally `28704`.

3. Start the realtime latent `z` server.

   ```bash
   cd /home/unitree/UFO-Deploy
   scripts/realtime/run_realtime_z_server_onboard.sh
   ```

   This requests poses from the teleop bridge, runs `backward_encoder.onnx`, and publishes
   latent `z` on port `28711`.

4. Start the G1 teleop policy.

   ```bash
   cd /home/unitree/UFO-Deploy
   source /home/unitree/ufo_deploy_venv/bin/activate
   UFO_REAL_ROBOT_OK=1 VENV_PATH=/home/unitree/ufo_deploy_venv/bin/activate \
     ./run_g1_teleop_policy_onboard.sh
   ```

   This subscribes to realtime `z` and runs UFO policy inference.

Keep the robot on support for first bring-up and test the physical e-stop, wireless R2
stop latch, stale-teleop watchdog, and PICO disconnect behavior before free walking.
