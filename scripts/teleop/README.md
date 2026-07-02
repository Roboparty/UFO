# Teleop Bridge Setup Guide

This README is for a user who wants to run the live teleoperation bridge on their own Ubuntu machine.

If you follow this file from top to bottom, you should end up with:

1. a working Python environment for the teleop bridge,
2. GMR installed for online retargeting,
3. XRoboToolkit PC service installed on the host machine,
4. the XRoboToolkit Python binding installed,
5. the PICO headset configured and connected,
6. the teleop ZMQ bridge running and ready to feed UFO-Deploy's realtime `z` encoder.

## What this bridge does

The runtime in this folder does only three things:

1. read live body and controller data from XRoboToolkit / PICO,
2. retarget the incoming human motion to Unitree G1 with GMR,
3. publish retargeted G1 pose frames over ZMQ so that `scripts/realtime/realtime_z_server.py` can request them and encode realtime latent `z`.

The included files are:

- `xrobot_teleop_to_pose_zmq_server.py`
- `default_mimic_obs.py`
- `teleop_pose_50hz.sh`

## Before you start

Use Ubuntu 22.04 / 24.04 if possible.

You need:

- this repository already checked out on your machine,
- Conda installed,
- a PICO headset with leg trackers,
- XRoboToolkit client installed on the PICO,
- XRoboToolkit PC service installed on the Ubuntu host,
- motion trackers/controllers paired and calibrated on the PICO side.

In the commands below, define one working directory for all external dependencies:

```bash
export TELEOP_WORKSPACE=$HOME/teleop_ws
mkdir -p "$TELEOP_WORKSPACE"
```

## Step 1. Create the teleop Python environment

Create a dedicated Python 3.10 environment for the teleop bridge.

```bash
conda create -n gmr python=3.10 -y
conda activate gmr
```

This matches TWIST2 setup, where online teleoperation and GMR run in a Python 3.10 environment separate from the low-level deployment stack.

## Step 2. Install host system dependencies

Install the build tools and runtime packages needed by GMR and the XRoboToolkit binding.

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-dev \
    python3-pip \
    libgl1 \
    libegl1 \
    libxrender1 \
    libxext6
```

Then install Conda-side helper packages used by the original setup:

```bash
conda activate gmr
conda install -c conda-forge libstdcxx-ng pybind11 -y
```

## Step 3. Install GMR

Clone and install GMR into the same `gmr` environment.

```bash
cd "$TELEOP_WORKSPACE"
git clone https://github.com/YanjieZe/GMR.git
cd GMR
pip install -e .
```

The bridge in this folder also needs `pyzmq`, so install it explicitly:

```bash
pip install pyzmq
```

## Step 4. Install XRoboToolkit PC service on Ubuntu

According to the original TWIST2 instructions, you can either:

- install the Ubuntu `.deb` package from the XRoboToolkit PC service release page
    ```bash
    cd "$TELEOP_WORKSPACE"
    wget https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
    sudo dpkg -i XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
    ```
- build the PC service from [source](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases).

After installation, start the PC service application from the Ubuntu application launcher before you start teleoperation.

## Step 5. Build and install the XRoboToolkit Python binding

The teleop bridge uses `XRobotStreamer`, and `XRobotStreamer` depends on the Python module `xrobotoolkit_sdk`.

### 5.1 Clone the binding repository

```bash
cd "$TELEOP_WORKSPACE"
git clone https://github.com/Axellwppr/XRoboToolkit-PC-Service-Pybind
cd XRoboToolkit-PC-Service-Pybind
```

### 5.2 Build the underlying XRoboToolkit native SDK

```bash
mkdir -p tmp
cd tmp
git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git
cd XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK
bash build.sh
cd ../../../..
```

### 5.3 Copy the built headers and shared library into the pybind repo

Run the following from inside `XRoboToolkit-PC-Service-Pybind`:

```bash
mkdir -p lib
mkdir -p include
cp tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/PXREARobotSDK.h include/
cp -r tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/nlohmann include/nlohmann/
cp tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so lib/
```

### 5.4 Install the Python module

Still inside `XRoboToolkit-PC-Service-Pybind`:

```bash
conda activate gmr
pip uninstall -y xrobotoolkit_sdk
python setup.py install
```

## Step 6. Verify the Python environment

Before touching the headset, verify that the required Python modules import correctly.

```bash
conda activate gmr
python - <<'PY'
import general_motion_retargeting
import xrobotoolkit_sdk
import zmq
print("general_motion_retargeting: OK")
print("xrobotoolkit_sdk: OK")
print("pyzmq: OK")
PY
```

If this fails, do not continue. Fix the environment first.

## Step 7. Install and prepare the PICO side

Install the PICO-side app from:

- `https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/`

Then prepare the headset as follows:

1. Put on the motion trackers.
2. Put the controllers on the wrists.
3. Start VR on the headset.
4. Calibrate the whole-body motion tracking.
5. Open the XRoboToolkit / XRobot app on the headset.
6. Connect the app to the IP address of your Ubuntu host.
7. Start streaming whole-body data.
8. Start streaming controller data.

The PC and the PICO headset must be able to reach each other over the network.

In practice, that means:

- they are on the same LAN, and
- the PICO can route to the Ubuntu host IP you entered in the app.
- Note: ensure the communication link from the PC to the PICO is stable and has minimal packet loss; otherwise motion jitter may occur.
- Note: if you are using `iptables/nftables/ufw` on the Ubuntu host, make sure to allow incoming connections.
- Note: if you are using `VPN/TUN` interface, stop the `VPN/TUN` while teleop, or make sure it is configured to allow the PICO to reach the host IP.
- Note: if `http_proxy` and `https_proxy` environment variables are set on the Ubuntu host, make sure have `127.0.0.1` in the `no_proxy` variable, or unset the proxy variables while teleop.

## Step 8. Verify that XR data is arriving

Once the PC service is running and the PICO is connected, verify that the Python binding can see the stream.

```bash
conda activate gmr
python - <<'PY'
import xrobotoolkit_sdk as xrt

xrt.init()
print("Body data available:", xrt.is_body_data_available())
print("Headset pose:", xrt.get_headset_pose())
print("Left controller pose:", xrt.get_left_controller_pose())
print("Right controller pose:", xrt.get_right_controller_pose())
xrt.close()
PY
```

If body data is not available, the XRoboToolkit binding README suggests checking:

1. the PICO headset is connected,
2. the trackers are connected and calibrated,
3. full body tracking is enabled on the PICO side client.

## Step 9. Run the teleop bridge

Once the environment and XR data stream are ready, start the ZMQ teleop bridge from this repository.

```bash
conda activate gmr
cd <path-to-UFO-Deploy>/scripts/teleop
./teleop_pose_50hz.sh
```

This starts the retargeting server used by UFO-Deploy:

- request socket: `tcp://*:28701`
- reply socket: `tcp://*:28702`
- controller socket: `tcp://*:28703`
- control publish rate: `50 Hz`

Important: During the first few seconds after starting the script, remain in a stable standing posture. The script adjusts the z-axis offset based on foot height; if you are in another pose, the estimated z-offset may affect gait quality.

## Runtime workflow

At runtime, the teleop bridge acts as a low-latency G1 pose supplier for `scripts/realtime/realtime_z_server.py`.

The high-level flow is:

1. `realtime_z_server.py` requests the latest retargeted G1 pose at 50 Hz.
2. This teleop bridge samples the latest XRoboToolkit body stream, retargets it to `unitree_g1` with GMR, and returns one retargeted pose frame.
3. `realtime_z_server.py` runs MuJoCo FK and the exported `backward_encoder.onnx`.
4. `realtime_z_server.py` publishes the resulting 256-D latent `z` on `tcp://*:28711`.
5. The local sim policy or robot policy subscribes to that `z` stream through its tracking task config.

More concretely:

- `realtime_z_server.py` is the active side for pose fetching. It requests the latest retargeted pose each control step.
- The bridge exposes three ZMQ channels:
  - request channel: receives pose requests from `realtime_z_server.py`
  - reply channel: sends retargeted pose frames back
  - control channel: publishes XR controller button state
- The reply payload contains `root_pos`, `root_quat`, and `dof_pos` for each returned frame.
- The realtime `z` server uses the controller channel to switch between follow and freeze modes.
- The bridge keeps a short retarget history and samples with a small lookback window to reduce jitter.
