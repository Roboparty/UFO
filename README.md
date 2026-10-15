# UFO Deploy

Run a UFO policy on **Unitree G1 29-DoF** in MuJoCo or on the robot.

| Task | `sim2sim` | `sim2real` | Reference |
|---|---|---|---|
| `tracking` | Workstation | Onboard | Exported motion z sequence |
| `goal` | Workstation | Onboard | Exported goal z |
| `reward` | Workstation | Onboard | Exported reward z |
| `teleop` | Workstation | Onboard only | PICO → retarget → backward encoder → live z |

All tasks share one observation/history builder, ONNX actor and control loop.
The mode selects MuJoCo communication or the G1 hardware interface. Teleop sim2real
runs retargeting, z encoding and policy inference **on the robot**; PICO connects
to the robot IP. Internal teleop streams use localhost.

## Install

Use Python 3.10: Conda on the workstation, a venv on the G1 onboard computer.

```bash
git clone --branch deploy --single-branch https://github.com/Roboparty/UFO.git UFO-Deploy
cd UFO-Deploy
conda create -n ufo-deploy python=3.10 -y
conda activate ufo-deploy
pip install -r requirements/runtime.txt
```

For teleop, also install `requirements/teleop.txt` and follow the
[PICO / XRoboToolkit setup](scripts/teleop/README.md). For real deployment,
complete the [onboard setup](scripts/onboard/README.md).

| Dependency | Used by |
|---|---|
| `requirements/runtime.txt` | All policy tasks and MuJoCo |
| `requirements/teleop.txt` + `xrobotoolkit_sdk` + XRoboToolkit service | Teleop only |
| `g1_interface` + CycloneDDS | Real G1 control only |
| `unitree_sdk2py` | Optional read-only hardware diagnostics |

Runtime inference uses ONNX Runtime on CPU by default. Neither training nor z
inference from a replay buffer runs in this deployment repository.

## Model and task files

Copy the ONNX and its matching export sidecars together:

```text
model/g1_policy/
├── exported/
│   ├── FBcprAuxModel.onnx
│   ├── FBcprAuxModel.meta.json
│   ├── FBcprAuxModel.deploy.yaml
│   ├── backward_encoder.onnx          # teleop
│   └── backward_encoder.meta.json     # teleop
├── tracking_inference/zs_<motion_id>.pkl
├── goal_inference/goal_reaching.pkl
└── reward_inference/reward_locomotion.pkl
```

The policy config defaults to the chosen ONNX's `.deploy.yaml`. Observation
order, scales, history and model hash are checked against `.meta.json`. ONNX
loading, tensor dimensions and task z files are validated before opening hardware.
Older exports without these files need to be exported again.
`config/policy/g1_policy.yaml` is a reference template; use the model's export.

Edit `config/tasks/<task>.yaml` to select the files and names you actually exported:
tracking's `zs_7.pkl` is an example, not a fixed motion ID. Goal names and reward
z indices must also exist in the selected file. These z files come from the
training repository's inference commands; use the matching policy checkpoint.

`ctx_path` resolves under the task directory above, or may be absolute. The older
`*_inference_mjlab` directory is a fallback when the file is absent from the current
directory. Set `ctx_dir` to override the task directory. Only load trusted files.

## Sim2sim

Run MuJoCo in terminal A:

```bash
python -m sim_env.simulator
```

Run **one** task in terminal B:

```bash
python deploy.py --mode sim2sim --task tracking
python deploy.py --mode sim2sim --task goal
python deploy.py --mode sim2sim --task reward
```

For teleop, first start the pose bridge and z encoder in two additional
terminals, as described in the [teleop guide](scripts/teleop/README.md), then:

```bash
python deploy.py --mode sim2sim --task teleop
```

The simulator starts with a virtual support band. Its viewer keys are `7`
(tighten), `8` (loosen), `9` (toggle support; disabling clears its applied force).

Keyboard controls in the policy terminal:

| Key | Action |
|---|---|
| `i` | Interpolate to the default standing pose |
| `]` | Enable policy action; reward holds the initial pose until `[` |
| `[` | Start tracking / reward policy |
| `p` | Reset tracking to its stop frame, or select the first goal/reward z |
| `n` | Select the next goal/reward z |
| `o` | Stop policy action |
| `r` | Re-arm a latched stop after fresh state returns; does not enable action |

## Sim2real — onboard

After [onboard setup](scripts/onboard/README.md), activate the deployment venv
and select the robot's low-level DDS network interface:

```bash
source /home/unitree/ufo_deploy_venv/bin/activate
export G1_INTERFACE=eth0  # Replace with the actual DDS interface.
export UFO_REAL_ROBOT_OK=1
```

Run **one** task. The launcher configures native libraries and calls
`deploy.py --mode sim2real`:

```bash
scripts/onboard/run_policy.sh --task tracking
scripts/onboard/run_policy.sh --task goal
scripts/onboard/run_policy.sh --task reward
scripts/onboard/run_policy.sh --task teleop
```

Teleop additionally requires the **onboard** pose bridge and z encoder to be
running. See [teleop startup](scripts/teleop/README.md#startup).

G1 remote controls: **A** initializes standing; **R2** stops policy action and
latches the stop. Release **R2**, then press **A** again to initialize once fresh
robot state is available. Holding **R2** takes priority over **A**.

| Task | R1 | B |
|---|---|---|
| Tracking | Enable policy at the configured stop frame | Start the motion sequence |
| Goal | Execute the selected goal | Unused |
| Reward | Arm the policy and hold the initialization pose | Start the reward policy |

Reward uses a fixed z vector. Its waiting pose is the initialization pose;
after pressing **A** again, use **R1 → B** to resume. **Y** selects the next
goal/reward; **X** selects the first one, or resets tracking to its stop frame.
The **R1+B** shortcut directly re-arms and resumes execution after releasing
the stop and enable/start buttons.

For teleop, **R1** enables policy on the latest live reference. PICO right-hand
A follows/resumes the reference stream; left-hand X freezes it.

Keep the robot supported for initial checks. A physical e-stop is required.
State loss, inference errors, invalid actions/targets and stale live z latch a stop.
The control loop skips overdue steps instead of sending catch-up commands; falling
behind by a full control period also latches a stop. After resolving the fault,
use **A → R1**, then **B** for tracking/reward (keyboard `i → ] → [`).
Keyboard `r` also clears the latch without initializing. Receiving fresh data alone
does not resume policy action.

## Choose another export or task configuration

The same options apply to all tasks and to `scripts/onboard/run_policy.sh`:

```bash
python deploy.py --mode sim2sim --task tracking \
  --model /path/to/exported/FBcprAuxModel.onnx \
  --task-config /path/to/tracking.yaml
```

Use `--robot-config` for local robot settings and `--policy-config` only for a
config that still matches the selected export. `--mode` and `--task` are checked
against the loaded configs to prevent accidentally selecting hardware or a
different reference source. `python deploy.py --help` lists all options.

## Code layout

```text
deploy.py                    # Mode/task selection and configuration
config/
├── tasks/                   # tracking, goal, reward, teleop
├── robot/                   # Shared G1 limits + real interface settings
├── policy/                  # Reference config; actual config comes from export
├── scene/                   # MuJoCo scene and timing
└── teleop/                  # Retarget calibration and timing
rl_policy/                   # Shared policy, observations, state/command adapters
sim_env/                     # MuJoCo simulator and PD bridge
scripts/
├── onboard/                 # Native environment, launch and diagnostics
├── teleop/                  # PICO pose bridge and vendored retargeter
├── realtime/                # Pose → backward encoder → z
├── publishers/              # Optional diagnostic stream publishers
└── vis/                     # Optional reference-pose viewer
utils/                       # Shared artifact/config, joint and message helpers
tests/                       # Regression and safety checks
tools/                       # Training/deployment observation comparison
```

`g1_real.yaml` inherits joint limits from `g1.yaml`; copy both together.
Simulation torque caps in `g1.yaml` and the robot XML match the current UFO G1
training caps. They do not set the real robot's motor limits or reproduce its
complete actuator dynamics.
The optional diagnostic publishers/viewer are not needed by the eight main flows.

## Checks

These commands do not actuate the robot:

```bash
python scripts/onboard/check_policy_preflight.py --task config/tasks/tracking.yaml
python -m pip install pytest
python -m pytest -q tests/test_deploy_entry.py tests/test_ufo_policy_safety.py \
  tests/test_realtime_z_server_safety.py tests/test_onboard_deployment_helpers.py \
  tests/test_dependency_layout.py
```

For goal, reward or teleop, give preflight the matching task YAML. For another
export, add `--model-path /path/to/exported/FBcprAuxModel.onnx`.
These tests use mocked hardware; they do not validate physical G1 or PICO behavior.
See the [dependency matrix](docs/deployment_dependencies.md) and
[onboard diagnostics](scripts/onboard/README.md#diagnostics) for native SDK checks.
