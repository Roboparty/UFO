# UFO-Deploy

The `deploy` branch is the UFO-Deploy runtime: a deployment-only code path for running
a released BFM-Zero-compatible latent policy on Unitree G1 29-DoF. It is not the training
codebase, and it should not be merged or rebased into `main`.

Release-supported target:

- Unitree G1 29-DoF

Other robot type strings or legacy code paths are not release-supported unless they are
documented and tested in this branch.

Supported deployment flows:

- local MuJoCo sim2sim
- local PICO/XRobot canonical retarget teleop sim2sim
- onboard G1 sim2real
- teleop sim2real, where the workstation retargets PICO motion, encodes realtime latent `z`, and the robot subscribes over ZMQ
- onboard PICO teleop sim2real, where PICO connects directly to the robot

This README is written for a new user cloning the repository from GitHub.

## Not Supported By This Branch

This branch does not train policies.
This branch does not retarget arbitrary robot morphologies.
This branch does not include model artifacts in git.
Only the released Unitree G1 29DoF policy artifact layout documented below is release-supported.

## What You Need

Workstation:

- Linux workstation with Conda
- Python 3.10
- MuJoCo
- Optional CUDA-capable GPU for realtime `z` encoding

Teleop workstation:

- patched `xrobotoolkit_sdk` with callback APIs or polling APIs
- vendored `motion_tracking_retarget` code and G1 assets included in this repo
- `mink`, `mujoco`, `numpy`, `scipy`, `pyyaml`, and `pyzmq`
- PICO/XRobot runtime set up outside this repo
- optional `viser` and `mjviser` for the browser retarget viewer

Robot:

- Unitree G1 29-DoF onboard Jetson
- Python 3.10 venv
- CycloneDDS runtime
- `g1_interface` CPython 3.10 aarch64 binding for real G1 control
- low-level DDS network interface passed explicitly with `G1_INTERFACE` or a local `ROBOT_CONFIG`
- for onboard PICO teleop only: XRoboToolkit headless service, `xrobotoolkit_sdk`,
  canonical retarget dependencies, and PICO headset with trackers/controllers
- optional readonly diagnostics only: `unitree_sdk2py`

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

This is required by both ordinary onboard Sim2Real and onboard teleop Sim2Real.
It must match the UFO Python runtime ABI. The current runtime Python is Python
3.10, so the required onboard extension is:

```text
g1_interface.cpython-310-aarch64-linux-gnu.so
```

Do not use these incompatible bindings:

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
or retargeted qpos. `xrobotoolkit_sdk` is not the policy runtime and is not the
G1 control binding.

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

This optional diagnostic dependency is for readonly low-state subscriber checks:
`q`, `dq`, IMU, and wireless remote state. It is not used by UFO policy control.
Missing `unitree_sdk2py` does not block ordinary Sim2Real or PICO teleop.

Current deploy does not require the external `general_motion_retargeting`/GMR
package. The legacy GMR architecture is no longer used by the deploy runtime:

```text
general_motion_retargeting installed: no
GMR required: no
torch required for teleop retargeting: no
```

Ordinary Sim2Real has no human-pose or retargeting dependency. It runs the
released policy ONNX path directly: observation, backward encoder latent `z`,
UFO policy, and G1 command.

Onboard PICO teleop uses the vendored `scripts/teleop/motion_tracking_retarget/`
package with XRoboToolkit polling and Mink IK. `qpsolvers` and `daqp`, when
present, are Mink solver dependencies rather than legacy GMR dependencies.

See [docs/deployment_dependencies.md](docs/deployment_dependencies.md) for the
deployment dependency matrix.

## Clone And Install

```bash
git clone --branch deploy --single-branch https://github.com/Roboparty/UFO.git UFO-Deploy
cd UFO-Deploy
export UFO_ROOT=$PWD

conda create -n ufo-deploy python=3.10 -y
conda activate ufo-deploy
pip install -r requirements/runtime.txt
```

For PICO teleop, install the teleop set in the teleop Python environment:

```bash
pip install -r requirements/teleop.txt
```

`requirements/teleop.txt` includes `requirements/runtime.txt` because direct
onboard PICO teleop Sim2Real also runs realtime `z`, ONNX inference, policy
code, and G1 runtime. It still excludes torch and the external GMR package.

`requirements.txt` remains as a compatibility superset and includes
training/debug dependencies. For ordinary deployment use
`requirements/runtime.txt`; for PICO teleop deployment use
`requirements/teleop.txt`.

By default, policy inference uses ONNX Runtime `CPUExecutionProvider`. To use CUDA, install
`onnxruntime-gpu` that matches your CUDA setup and set `onnx_providers` in
`config/policy/g1_policy.yaml`.

For CPU-only runs, the `onnxruntime` package from `requirements/runtime.txt` is enough.

Check the base Python dependencies:

```bash
python -c "import mujoco, onnxruntime, zmq, yaml, numpy; print('base deps ok')"
```

Use this `ufo-deploy` environment for MuJoCo, realtime `z`, and policy inference. Use a
separate `ufo-teleop` environment for the PICO/XRobot retarget bridge; for that environment,
PICO/XRobot setup, and canonical retarget checks, follow
[scripts/teleop/README.md](scripts/teleop/README.md).

## Model Files

Released artifact:

```text
HF repo: xuewang/ufo-g1-policy
Runtime repo: Roboparty/UFO
Runtime branch: deploy
Runtime policy: latest deploy HEAD
```

The `deploy` branch tracks the latest supported G1 runtime. The default README workflow
uses the current `deploy` HEAD and the current model artifact from the HF repo. Older
model/runtime pairs should be accessed through explicit Git tags and Hugging Face
revisions, not through the README main flow.

The policy directory expected by the runtime is:

```text
model/g1_policy/
  exported/
    FBcprAuxModel.onnx
    backward_encoder.onnx
  tracking_inference_mjlab/*.pkl
  reward_inference_mjlab/*.pkl
  goal_inference_mjlab/*.pkl
  release_manifest.yaml
```

`model/` is ignored by git because the ONNX model is larger than GitHub's normal file limit. After cloning, put the released model artifact at `model/g1_policy`.
The `ctx_dir` and `ctx_path` values in `config/exp/*/*.yaml` are resolved under this model
root by default. The released artifact layout must match the tree above.

Download the runtime artifact:

```bash
export HF_REPO_ID=xuewang/ufo-g1-policy
# Optional: pin a specific artifact revision if needed.
# export HF_REVISION=<specific_revision>

python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["HF_REPO_ID"],
    repo_type="model",
    revision=os.environ.get("HF_REVISION"),
    local_dir="model",
    allow_patterns=[
        "g1_policy/exported/**",
        "g1_policy/tracking_inference_mjlab/*.pkl",
        "g1_policy/reward_inference_mjlab/*.pkl",
        "g1_policy/goal_inference_mjlab/*.pkl",
        "g1_policy/release_manifest.yaml",
        "g1_policy/README.md",
    ],
)
PY
```

The artifact also contains `g1_policy/tracking_inference_mjlab/tracking_mjlab_*.mp4`
rollout previews. They are useful for inspection but are not required by runtime policy
inference. To download the full artifact including videos, change `allow_patterns` to
`["g1_policy/**"]`.

Additional `tracking_inference_mjlab/zs_*.pkl` files are included for offline comparison
and manual selection.

> **Safety Alert**
> The ONNX policy, backward encoder, context files, and deploy runtime should be
> validated as one release unit. Do not mix an arbitrary old artifact with the latest
> deploy runtime unless it is explicitly marked compatible. The MP4 files are rollout
> previews for inspection only and are not evidence of real-robot safety. Before real
> robot use, complete sim2sim, hoist/support checks, realtime `z` watchdog and R2
> stop-latch checks, and use a physical e-stop.

Verify:

```bash
test -f model/g1_policy/exported/FBcprAuxModel.onnx
test -f model/g1_policy/exported/backward_encoder.onnx
test -f model/g1_policy/tracking_inference_mjlab/zs_7.pkl
test -f model/g1_policy/release_manifest.yaml
```

Verify the required artifact hashes against `release_manifest.yaml`:

```bash
python - <<'PY'
import hashlib
from pathlib import Path
import yaml

root = Path("model/g1_policy")
manifest = yaml.safe_load((root / "release_manifest.yaml").read_text())

files = {
    "policy_onnx": root / "exported" / "FBcprAuxModel.onnx",
    "backward_encoder_onnx": root / "exported" / "backward_encoder.onnx",
    "tracking_context": root / "tracking_inference_mjlab" / "zs_7.pkl",
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

for key, path in files.items():
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = manifest["sha256"][key]
    got = sha256(path)
    if got != expected:
        raise RuntimeError(f"{key} sha256 mismatch: got {got}, expected {expected}")

print("model artifact sha256 ok")
PY
```

For onboard deployment, also run the repo-side manifest checker. This manifest
records required paths and known hashes without committing large model binaries:

```bash
python scripts/onboard/check_deploy_artifacts.py
```

`model/g1_policy/artifact_manifest.yaml` contains authoritative hashes only when
they are known. Missing pkl/metadata hashes are reported as warnings rather than
invented values.

## G1 Onboard Clean-Checkout Setup

From a clean onboard checkout:

```bash
git clone --branch deploy --single-branch https://github.com/Roboparty/UFO.git UFO-Deploy
cd UFO-Deploy
git rev-parse HEAD
```

Then:

Ordinary onboard Sim2Real requires:

- Python 3.10
- runtime dependencies from `requirements/runtime.txt`
- released `model/g1_policy/` artifacts
- `g1_interface.cpython-310-aarch64-linux-gnu.so`
- CycloneDDS runtime
- explicit low-level DDS interface via `G1_INTERFACE` or local `ROBOT_CONFIG`

It does not require `xrobotoolkit_sdk`, XRoboToolkit service, PICO, retargeting,
or `unitree_sdk2py`.

Onboard PICO Teleop Sim2Real requires everything from ordinary onboard Sim2Real,
plus:

- `requirements/teleop.txt`
- `xrobotoolkit_sdk`
- XRoboToolkit headless service
- vendored `motion_tracking_retarget` dependencies
- PICO headset with trackers/controllers

Optional readonly diagnostics additionally require `unitree_sdk2py`.

Set up ordinary onboard Sim2Real in this order:

1. Create a Python 3.10 venv and install ARM64 dependencies from
   `requirements/runtime.txt`.
2. Restore/download `model/g1_policy/` artifacts and run
   `python scripts/onboard/check_deploy_artifacts.py`.
3. Install or expose the CPython 3.10 aarch64 `g1_interface` binding in the
   runtime environment.
4. Choose the low-level G1 DDS NIC explicitly:

   ```bash
   export G1_INTERFACE=<low-level-dds-interface>
   ```

   The launcher validates that this interface exists, is UP, has IPv4, and is
   not the default-route interface unless explicitly allowed. It does not
   silently auto-select a NIC.

5. If the current prebuilt `g1_interface` requires OpenSSL 1.1 on an OpenSSL 3
   system, provide:

   ```bash
   export OPENSSL11_LIB=/path/to/openssl-1.1/lib
   ```

   This is a compatibility requirement of the prebuilt Unitree binding, not a
   UFO Python dependency.

6. Run no-actuation ordinary preflight:

   ```bash
   scripts/onboard/run_preflight_suite.sh --profile ordinary
   ```

For onboard PICO teleop, install the teleop-only dependencies after the ordinary
runtime is healthy:

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

For optional readonly low-state diagnostics, install `unitree_sdk2py` separately
and run:

```bash
scripts/onboard/run_preflight_suite.sh --profile diagnostic
```

The onboard diagnostics live in `scripts/onboard/`. They avoid real actuation by
default and do not substitute for physical safety checks.

## Real G1 Validation

For the fixed d92 deployment, the user confirmed:

- ordinary onboard Sim2Real ran on the real G1;
- onboard PICO teleop Sim2Real ran on the real G1;
- the path from PICO/XRobot motion through retargeting, realtime z, UFO policy,
  and real G1 actuation was validated;
- no obvious functional problem was observed during the user-confirmed real G1
  test.

This does not claim systematic R2 fault injection, physical e-stop fault
injection, PICO disconnect testing, process-kill testing, long-duration free
walking, or quantified impact limits. Those remain separate validation tasks.

## Repository Map

```text
config/policy/g1_policy.yaml
config/robot/g1.yaml
config/robot/g1_real.yaml
config/scene/g1_29dof.yaml
config/exp/tracking/tracking.yaml
config/exp/tracking/teleop.yaml
rl_policy/ufo_policy.py
sim_env/base_sim.py
scripts/realtime/realtime_z_server.py
scripts/realtime/run_realtime_z_server_onboard.sh
scripts/teleop/check_teleop_env.py
scripts/teleop/teleop_pose_50hz.sh
scripts/teleop/teleop_pose_50hz_onboard.sh
scripts/teleop/xrobot_teleop_to_pose_zmq_server.py
run_g1_teleop_policy_onboard.sh
```

The local shell launcher is `scripts/teleop/teleop_pose_50hz.sh`. The `*_onboard.sh` launchers are for direct PICO-to-robot teleop sim2real.

## Recommended Order

Run in this order when bringing up a new machine, model, or teleop setup:

```text
1. local ordinary sim2sim
2. local teleop sim2sim
3. onboard ordinary sim2real with hoist/support
4. onboard PICO teleop sim2real, first observe realtime z and robot state without enabling policy
5. G1 A initializes stable standing -> G1 R1 enables policy action -> G1 B starts tracking -> test X/R2 stop
6. deliberately disconnect PICO/XRobot/ZMQ and confirm the watchdog stops policy action
```

## 1. Ordinary Sim2Sim

Terminal A, start MuJoCo:

```bash
cd "$UFO_ROOT"
conda activate ufo-deploy
python -m sim_env.base_sim \
  --robot_config ./config/robot/g1.yaml \
  --scene_config ./config/scene/g1_29dof.yaml
```

Terminal B, start the policy:

```bash
cd "$UFO_ROOT"
conda activate ufo-deploy
python rl_policy/ufo_policy.py \
  --robot_config config/robot/g1.yaml \
  --policy_config config/policy/g1_policy.yaml \
  --model_path model/g1_policy/exported/FBcprAuxModel.onnx \
  --task config/exp/tracking/tracking.yaml
```

Keyboard controls in the policy terminal:

```text
i   interpolate to default standing pose
]   enable policy action
[   start tracking motion
p   reset tracking motion to stop frame
o   stop policy action and hold current joints
n   next reward/goal z for reward/goal tasks
```

## 2. Teleop Sim2Sim

This runs all processes on the workstation. The policy reads realtime `z` from `tcp://127.0.0.1:28711`.

Terminal A, MuJoCo:

```bash
cd "$UFO_ROOT"
conda activate ufo-deploy
python -m sim_env.base_sim \
  --robot_config ./config/robot/g1.yaml \
  --scene_config ./config/scene/g1_29dof.yaml
```

Terminal B, PICO/XRobot canonical retarget server:

```bash
cd "$UFO_ROOT"
conda activate ufo-teleop
scripts/teleop/teleop_pose_50hz.sh
```

Terminal C, realtime latent `z` encoder:

```bash
cd "$UFO_ROOT"
conda activate ufo-deploy
python scripts/realtime/realtime_z_server.py \
  --teleop_req tcp://127.0.0.1:28701 \
  --teleop_rep tcp://127.0.0.1:28702 \
  --teleop_ctrl tcp://127.0.0.1:28703 \
  --enable-pico-control \
  --z_bind tcp://*:28711 \
  --hz 50 \
  --mujoco_xml data/robots/g1/scene_29dof_freebase.xml \
  --backward_onnx model/g1_policy/exported/backward_encoder.onnx \
  --device cuda \
  --root_height_obs \
  --wall-clock-dt \
  --fix-quat-continuity \
  --angvel-delta-frame world \
  --max-retarget-age-ms 200 \
  --max-z-delta 0.75
```

Use `--device cpu` if CUDA ONNX Runtime is not installed.

Terminal D, policy:

```bash
cd "$UFO_ROOT"
conda activate ufo-deploy
python rl_policy/ufo_policy.py \
  --robot_config config/robot/g1.yaml \
  --policy_config config/policy/g1_policy.yaml \
  --model_path model/g1_policy/exported/FBcprAuxModel.onnx \
  --task config/exp/tracking/teleop.yaml
```

PICO buttons consumed by the realtime `z` server:

```text
right_key_one  follow mode
left_key_one   freeze current z
```

Realtime `z` starts in `freeze` mode by default; press PICO right-hand A after the policy
is ready to begin following the live reference.

## 3. Prepare The Robot

Copy the same repository and model to the robot:

```bash
export ROBOT_HOST=unitree@<ROBOT_IP>
export ROBOT_ROOT=/home/unitree/UFO-Deploy

ssh "$ROBOT_HOST" "mkdir -p $ROBOT_ROOT"
rsync -avP \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  "$UFO_ROOT"/ \
  "$ROBOT_HOST":"$ROBOT_ROOT"/
```

On the robot, activate its runtime environment:

```bash
cd /home/unitree/UFO-Deploy
source /home/unitree/ufo_deploy_venv/bin/activate

export CYCLONEDDS_HOME=/home/unitree/cyclonedds_ws/install/cyclonedds
export LD_LIBRARY_PATH=/home/unitree/unitree_sdk2_bfm/build/lib:/home/unitree/unitree_sdk2_bfm/thirdparty/lib/aarch64:$CYCLONEDDS_HOME/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/unitree/unitree_sdk2_bfm/build/lib:$PYTHONPATH
```

Check robot dependencies:

```bash
cat /sys/devices/system/cpu/online
ip -br addr
python -c "import g1_interface, onnxruntime; print(g1_interface.G1_NUM_MOTOR, onnxruntime.__version__)"
python -c "import onnxruntime as ort; ort.InferenceSession('model/g1_policy/exported/FBcprAuxModel.onnx', providers=['CPUExecutionProvider']); ort.InferenceSession('model/g1_policy/exported/backward_encoder.onnx', providers=['CPUExecutionProvider']); print('onnx ok')"
```

If Jetson CPU online is not `0-7`, fix it before running policy:

```bash
sudo bash -lc 'for c in 4 5 6 7; do echo 1 > /sys/devices/system/cpu/cpu${c}/online; done'
cat /sys/devices/system/cpu/online
```

Set the low-level interface in `config/robot/g1_real.yaml`:

```yaml
INTERFACE: "eth0"
USE_JOYSTICK: True
```

Use the actual interface name reported by `ip -br addr`.

## 4. Ordinary Sim2Real

Run on the robot after the checks above:

```bash
cd /home/unitree/UFO-Deploy
source /home/unitree/ufo_deploy_venv/bin/activate

export CYCLONEDDS_HOME=/home/unitree/cyclonedds_ws/install/cyclonedds
export LD_LIBRARY_PATH=/home/unitree/unitree_sdk2_bfm/build/lib:/home/unitree/unitree_sdk2_bfm/thirdparty/lib/aarch64:$CYCLONEDDS_HOME/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/unitree/unitree_sdk2_bfm/build/lib:$PYTHONPATH
UFO_REAL_ROBOT_OK=1 python rl_policy/ufo_policy.py \
  --robot_config config/robot/g1_real.yaml \
  --policy_config config/policy/g1_policy.yaml \
  --model_path model/g1_policy/exported/FBcprAuxModel.onnx \
  --task config/exp/tracking/tracking.yaml
```

G1 wireless controller sequence:

```text
A    interpolate to default standing pose, about 10 seconds at 50 Hz
R1   enable policy action
B    start tracking motion
X    reset tracking motion to stop frame
R2   stop policy action and hold current joints
Y    next reward/goal z for reward/goal tasks
```

Use the physical e-stop for emergencies.

## Real Robot Safety Checklist

Before enabling policy action on the real robot:

- [ ] Physical e-stop is reachable and tested.
- [ ] Robot is on hoist/support for first run.
- [ ] `cat /sys/devices/system/cpu/online` reports `0-7`.
- [ ] ONNX sessions load on the robot.
- [ ] `config/robot/g1_real.yaml` uses the correct network interface from `ip -br addr`.
- [ ] Ordinary sim2sim passes.
- [ ] Teleop sim2sim passes.
- [ ] Wireless R2 stop latch is tested.
- [ ] Realtime z watchdog is tested by disconnecting PICO/XRobot/ZMQ.
- [ ] `UFO_REAL_ROBOT_OK=1` is set only immediately before real robot control.

## Ports And Network Topology

| Port | Direction | Used by | Notes |
| --- | --- | --- | --- |
| 28701 | realtime z server -> teleop bridge | pose request | Localhost in onboard flow; workstation-local in split flow |
| 28702 | teleop bridge -> realtime z server | pose reply | Localhost in onboard flow; workstation-local in split flow |
| 28703 | teleop bridge -> realtime z server | Pico button/control channel | Used by realtime z server |
| 28704 | teleop bridge -> policy | legacy/debug optional PICO button PUB | Disabled by default; used only when both teleop and policy launchers opt in |
| 28711 | realtime z server -> policy | realtime latent z PUB | `127.0.0.1` for onboard flow; workstation IP for split flow |
| 8080 | browser -> retarget viewer | optional web viewer | Debug only |

In the onboard flow, `ctx_zmq_addr` should be:

```text
tcp://127.0.0.1:28711
```

In the split workstation/robot flow, the robot-side `ctx_zmq_addr` should be:

```text
tcp://<WORKSTATION_IP>:28711
```

## Onboard PICO Teleoperation

Onboard PICO teleop is the direct PICO-to-G1 flow. The teleop host is the Unitree G1
onboard Jetson, not the workstation. The PICO app connects to the robot IP, and the robot
runs the XRoboToolkit headless service, vendored canonical retarget bridge, realtime `z`
server, and UFO policy locally.

Hardware:

- Unitree G1 onboard Jetson
- PICO headset
- trackers/controllers paired and calibrated in the PICO app

Software on the G1 Jetson:

- XRoboToolkit headless service
- `xrobotoolkit_sdk` Python binding
- vendored `motion_tracking_retarget` code and G1 assets included in UFO-Deploy
- `mink`, `mujoco`, `numpy`, `scipy`, `pyyaml`, and `pyzmq` in the teleop Python environment
- UFO-Deploy runtime and released `model/g1_policy` artifact

Flow:

```text
PICO XRoboToolkit client
 |
 WiFi / LAN target IP = <G1_JETSON_IP>
 |
 G1 Jetson XRoboToolkit headless service
 |
 motion_tracking_retarget
 |
 xrobot_teleop_to_pose_zmq_server.py
 |
 realtime_z_server.py
 |
 backward_encoder.onnx
 |
 UFO policy
```

Before startup, follow [scripts/teleop/README.md](scripts/teleop/README.md) to install
the XRoboToolkit headless service, `xrobotoolkit_sdk`, and teleop Python dependencies.
Users only need to clone UFO-Deploy; the canonical retarget code, config, G1 XML, and mesh
assets are vendored under `scripts/teleop/motion_tracking_retarget/`.
`xrobotoolkit_sdk` is only the Python binding; the XRoboToolkit service must be installed
and running separately.
If you override `POLICY_CONFIG`, set the same value as `TELEOP_POLICY_CONFIG` for the
teleop bridge so its joint-order permutation is checked against the policy's
`policy_joint_names`.

Step 1, start XRoboToolkit service:

```bash
bash /opt/apps/roboticsservice/runService.sh
ip -br addr
```

This receives the PICO body, headset, and controller stream on the G1 Jetson.
In the XRoboToolkit PICO app, set the target IP to the G1 onboard computer IP reported by
`ip -br addr`. For onboard mode the PICO connects to the robot, not to the PC.

Step 2, start the teleop pose bridge:

```bash
cd /home/unitree/UFO-Deploy
scripts/teleop/teleop_pose_50hz_onboard.sh
```

This runs `xrobot_teleop_to_pose_zmq_server.py`: PICO/XRoboToolkit data is retargeted
with the vendored canonical retargeter and published as G1 poses on ZMQ ports `28701`,
`28702`, and `28703`. The legacy/debug PICO policy-control PUB port `28704` is disabled
by default. It does not start policy inference and it does not encode latent `z`.
It also does not auto-start the XRoboToolkit service by default; use
`START_XROBOT_SERVICE=1 scripts/teleop/teleop_pose_50hz_onboard.sh` only after the
installed headless service has been verified.

The onboard web viewer is optional and is off by default so port `8080` cannot block the
core PICO -> canonical retarget -> ZMQ path. Enable it only for debugging:

```bash
WEB_VISUALIZE=1 scripts/teleop/teleop_pose_50hz_onboard.sh
```

Step 3, start `scripts/realtime/realtime_z_server.py`:

```bash
cd /home/unitree/UFO-Deploy
Z_PY=/home/unitree/ufo_deploy_venv/bin/python \
  scripts/realtime/run_realtime_z_server_onboard.sh
```

The onboard wrapper launches `scripts/realtime/realtime_z_server.py` with onboard defaults.
It verifies that the selected Python can import `numpy`, `mujoco`, `onnxruntime`, and
`zmq`, requests poses from the teleop bridge, runs `backward_encoder.onnx`, and publishes
realtime latent `z` on port `28711`. It starts in `freeze` mode by default, so it keeps
publishing standing or last valid `z` until PICO right-hand A explicitly switches the live
reference to follow mode.

Step 4, start policy inference:

```bash
cd /home/unitree/UFO-Deploy
source /home/unitree/ufo_deploy_venv/bin/activate
UFO_REAL_ROBOT_OK=1 VENV_PATH=/home/unitree/ufo_deploy_venv/bin/activate \
  ./run_g1_teleop_policy_onboard.sh
```

This subscribes to realtime latent `z` and runs UFO policy inference on the G1. Keep the
robot on support for first bring-up, and test the physical e-stop, wireless R2 stop latch,
and stale-teleop watchdog before free walking.

## 5. Teleop Sim2Real

The recommended release path is 5A direct PICO-to-robot onboard teleop. PICO connects to the robot IP, and the robot runs the retarget server, realtime `z` server, and policy locally. Because the realtime `z` server and policy are both onboard, `config/exp/tracking/teleop.yaml` can keep `ctx_zmq_addr: tcp://127.0.0.1:28711`.

| Flow | PICO connects to | Retarget server | Realtime `z` server | Policy | `ctx_zmq_addr` |
| --- | --- | --- | --- | --- | --- |
| 5A onboard | robot IP | robot | robot | robot | `tcp://127.0.0.1:28711` |
| 5B split | workstation IP | workstation | workstation | robot | `tcp://<WORKSTATION_IP>:28711` |

### 5A. Recommended Onboard PICO-To-Robot Flow

Run all three onboard launchers on the robot. PICO should connect to the robot IP, and `config/exp/tracking/teleop.yaml` can keep:

```yaml
ctx_source: zmq
ctx_zmq_addr: tcp://127.0.0.1:28711
ctx_norm_ref: 16.0
ctx_zmq_timeout_ms: 200
```

Robot terminal A, PICO/XRobot canonical retarget bridge:

```bash
cd /home/unitree/UFO-Deploy
scripts/teleop/teleop_pose_50hz_onboard.sh
```

Optional viewer debug session:

```bash
cd /home/unitree/UFO-Deploy
WEB_VISUALIZE=1 scripts/teleop/teleop_pose_50hz_onboard.sh
```

When viewer debug is enabled, open it from another machine on the same network:

```text
http://<ROBOT_IP>:8080
```

The onboard retarget web viewer loads a temporary MJCF with a checkerboard floor plane for visual debugging.
This floor is viewer-only and does not affect retargeting, realtime `z`, or policy control.

Robot terminal B, realtime `z` publisher:

```bash
cd /home/unitree/UFO-Deploy
Z_PY=/home/unitree/ufo_deploy_venv/bin/python \
  scripts/realtime/run_realtime_z_server_onboard.sh
```

Robot terminal C, real policy controlled by the G1 wireless remote:

```bash
cd /home/unitree/UFO-Deploy
source /home/unitree/ufo_deploy_venv/bin/activate
UFO_REAL_ROBOT_OK=1 VENV_PATH=/home/unitree/ufo_deploy_venv/bin/activate \
  ./run_g1_teleop_policy_onboard.sh
```

G1 wireless remote controls robot and policy state:

```text
G1 A    interpolate to default standing pose
G1 R1   enable policy action
G1 B    start tracking
G1 X    reset tracking/reference
G1 R2   global stop latch
```

PICO buttons control only the live motion reference stream:

```text
PICO right_key_one / right-hand A   follow or resume live reference
PICO left_key_one / left-hand X     freeze current reference/z
```

The realtime `z` server starts frozen in the onboard flow. After using the G1 remote to
enter default stand, enable policy action, and start tracking, press PICO right-hand A to
begin following the live reference. On freeze -> follow resume, realtime `z` resets its
previous-pose velocity history and blends from the frozen `z` to the new live `z`.

PICO buttons do not enable policy, clear R2, enter default pose, reset the real policy
state machine, or bypass the physical e-stop in the default flow. The legacy/debug 28704
PICO policy-control path is off unless both `CTRL_PUB_BIND_ADDR=tcp://*:28704` and
`ENABLE_PICO_POLICY_CONTROL=1` are set explicitly in both launcher shells.

Wireless R2 is a global stop latch: policy action and tracking motion are disabled while
R2 is held, and enable/start inputs cannot directly clear the latch. After R2 is released,
release enable/start inputs first; then re-arm explicitly with wireless R1+B.

The realtime `z` server stops publishing valid `z` when the teleop pose stream is stale or invalid. The policy subscriber rejects invalid realtime `z` packets and stops policy action if no valid 256-dim finite `z` arrives within `ctx_zmq_timeout_ms`. Policy actions and final joint targets are checked for finite values, and final `q_target` commands are slew-rate limited using the configured G1 joint velocity limits.

A physical e-stop is still required.

### 5B. Optional Advanced Split Workstation/Robot Flow

The split workstation/robot flow is still supported for advanced debugging:

On the workstation, find the IP reachable from the robot:

```bash
ip -br addr
```

On the robot copy, set the workstation address in the same teleop task file:

```yaml
# config/exp/tracking/teleop.yaml
ctx_source: zmq
ctx_zmq_addr: tcp://<WORKSTATION_IP>:28711
ctx_norm_ref: 16.0
ctx_zmq_timeout_ms: 200
```

Workstation terminal A, PICO/XRobot canonical retargeting:

```bash
cd "$UFO_ROOT"
conda activate ufo-teleop
scripts/teleop/teleop_pose_50hz.sh
```

Workstation terminal B, realtime `z` publisher:

```bash
cd "$UFO_ROOT"
conda activate ufo-deploy
python scripts/realtime/realtime_z_server.py \
  --teleop_req tcp://127.0.0.1:28701 \
  --teleop_rep tcp://127.0.0.1:28702 \
  --teleop_ctrl tcp://127.0.0.1:28703 \
  --enable-pico-control \
  --z_bind tcp://*:28711 \
  --hz 50 \
  --mujoco_xml data/robots/g1/scene_29dof_freebase.xml \
  --backward_onnx model/g1_policy/exported/backward_encoder.onnx \
  --device cuda \
  --root_height_obs \
  --wall-clock-dt \
  --fix-quat-continuity \
  --angvel-delta-frame world \
  --max-retarget-age-ms 200 \
  --max-z-delta 0.75
```

Robot terminal, policy subscriber:

```bash
cd /home/unitree/UFO-Deploy
source /home/unitree/ufo_deploy_venv/bin/activate

export CYCLONEDDS_HOME=/home/unitree/cyclonedds_ws/install/cyclonedds
export LD_LIBRARY_PATH=/home/unitree/unitree_sdk2_bfm/build/lib:/home/unitree/unitree_sdk2_bfm/thirdparty/lib/aarch64:$CYCLONEDDS_HOME/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/unitree/unitree_sdk2_bfm/build/lib:$PYTHONPATH
UFO_REAL_ROBOT_OK=1 python rl_policy/ufo_policy.py \
  --robot_config config/robot/g1_real.yaml \
  --policy_config config/policy/g1_policy.yaml \
  --model_path model/g1_policy/exported/FBcprAuxModel.onnx \
  --task config/exp/tracking/teleop.yaml
```

Controller sequence:

```text
A -> wait for stable default stand -> R1 -> B
X stops motion, R2 stops policy action.
```

If the robot does not react to teleop:

- the workstation realtime server should print `pose ok`
- for 5B split flow, `ctx_zmq_addr` must use the workstation IP, not `127.0.0.1`
- robot and workstation must be on the same reachable network
- TCP port `28711` must not be blocked

## Quick Validation

Run locally before pushing changes from an environment with the repository dependencies installed:

```bash
conda activate ufo-deploy
# Or, on the robot:
# source /home/unitree/ufo_deploy_venv/bin/activate

python -m py_compile \
  rl_policy/ufo_policy.py \
  rl_policy/observations/ufo_policy.py \
  scripts/realtime/realtime_z_server.py \
  scripts/teleop/check_teleop_env.py \
  scripts/teleop/xrobot_teleop_to_pose_zmq_server.py \
  scripts/teleop/motion_tracking_retarget/*.py \
  sim_env/base_sim.py \
  sim_env/utils/simulation_bridge.py \
  rl_policy/utils/state_processor.py \
  rl_policy/utils/command_sender.py \
  utils/common.py \
  utils/math.py \
  utils/strings.py \
  tests/test_ufo_policy_safety.py \
  tests/test_realtime_z_server_safety.py

bash -n \
  scripts/teleop/teleop_pose_50hz_onboard.sh \
  scripts/realtime/run_realtime_z_server_onboard.sh \
  run_g1_teleop_policy_onboard.sh
python tests/test_motion_tracking_retarget.py
python tests/test_ufo_policy_safety.py
python tests/test_realtime_z_server_safety.py

git diff --check
git diff --cached --check
```
