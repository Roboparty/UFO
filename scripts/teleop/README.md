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
conda activate ufo-deploy
scripts/teleop/teleop_pose_50hz.sh
```

### G1 Onboard Teleop

Use this mode for direct PICO-to-robot teleop. The PICO connects to the G1 Jetson IP, so
the teleop host moves from the PC to the G1 onboard computer.

```text
PICO
 |
 XRoboToolkit
 |
 G1 Jetson
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
exact package path depends on the XRoboToolkit release you use; after installation the G1
should have a headless service launcher such as:

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

## Install xrobotoolkit_sdk

Build and install the Python binding in the same environment used by the bridge.

```bash
cd "$TELEOP_WORKSPACE"
git clone https://github.com/Axellwppr/XRoboToolkit-PC-Service-Pybind
cd XRoboToolkit-PC-Service-Pybind
```

Build the native SDK used by the binding:

```bash
mkdir -p tmp
cd tmp
git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git
cd XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK
bash build.sh
cd ../../../..
```

Copy the headers and shared library into the pybind repository:

```bash
mkdir -p lib include
cp tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/PXREARobotSDK.h include/
cp -r tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/nlohmann include/nlohmann/
cp tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so lib/
```

Install:

```bash
pip uninstall -y xrobotoolkit_sdk
python setup.py install
```

For Jetson/headless installs, use the arm64 native library from the XRoboToolkit service
release. Prefer keeping it under the UFO checkout, for example:

```bash
mkdir -p /home/unitree/UFO-Deploy/external/XRoboToolkit-PC-Service-Pybind/lib/aarch64
# copy libPXREARobotSDK.so into that directory
export LD_LIBRARY_PATH=/home/unitree/UFO-Deploy/external/XRoboToolkit-PC-Service-Pybind/lib/aarch64:$LD_LIBRARY_PATH
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
[OK] xrobotoolkit_sdk installed
[OK] pyzmq installed
[OK] XRoboToolkit service running
[OK] port 28701 available
[OK] port 28702 available
[OK] port 28703 available
[OK] port 28711 available
```

Before starting teleop, ports `28701`, `28702`, `28703`, and `28711` should normally be
available. If a previous teleop or realtime `z` process is still running, the checker will
report the occupied port so you can stop the stale process.

After the teleop bridge and realtime `z` server are already running, use:

```bash
python scripts/teleop/check_teleop_env.py --mode running
```

That mode expects the ZMQ ports to be occupied by the running services.

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

The preflight checker includes `28701`, `28702`, `28703`, and `28711` by default because
they are required for the normal teleop-to-policy pipeline. The onboard teleop launcher
also checks `28704` when the PICO policy-control PUB channel is enabled.

## Run The Teleop Bridge

### Workstation

```bash
cd "$UFO_ROOT"
conda activate ufo-deploy
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
export WEB_VISUALIZE=1
export WEB_PORT=8080
```

The onboard launcher automatically derives `UFO_ROOT` from its own path if `UFO_ROOT` is
not set. It checks the Python executable, GMR, `xrobotoolkit_sdk`, XRoboToolkit service,
and required local ZMQ ports before starting `xrobot_teleop_to_pose_zmq_server.py`.
It searches UFO-specific environments first:

```text
${UFO_ROOT}/.venv/bin/python
${UFO_ROOT}/venv/bin/python
/home/unitree/ufo_teleop_venv/bin/python
/home/unitree/ufo_deploy_venv/bin/python
/home/unitree/miniconda3/envs/ufo-teleop/bin/python
/home/unitree/miniconda3/envs/ufo-deploy/bin/python
```

If none of those exists, set `TELEOP_PY` explicitly. The launcher refuses to fall back to
system `python3` by default because the Jetson system Python usually does not contain GMR,
`xrobotoolkit_sdk`, `torch`, and `numpy`. Set `TELEOP_ALLOW_SYSTEM_PY=1` only for manual
debugging.

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
