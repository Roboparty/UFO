# Unitree G1 Onboard Deployment Status - 2026-07-25

This report records the deployment evidence for Roboparty/UFO `deploy` at fixed
base commit `d92bd38e914f888e90da3a303d6ec2a73ad6e60d`.

## Current Conclusion

- Ordinary onboard Sim2Real has been tested on the real G1.
- Onboard PICO teleop Sim2Real has been tested on the real G1.
- The end-to-end path from PICO/XRobot motion through retargeting, realtime z,
  UFO policy, and real G1 actuation has been validated.
- No obvious functional problem was observed during the user-confirmed
  real-robot test.

This conclusion is based on user-confirmed real-robot operation plus the
automated no-actuation checks recorded below. It does not imply that every
safety fault case has been injected or that long-duration free walking has been
systematically validated.

## Validated Paths

User-confirmed real G1 tests:

- Ordinary onboard Sim2Real policy entrypoint ran on the real G1.
- Onboard PICO teleop Sim2Real ran on the real G1.
- PICO body motion drove the live retarget path, realtime latent z, UFO policy,
  and real G1 actuation.

Automated no-actuation validation:

- Fixed base commit verified against upstream `deploy`.
- Policy ONNX loads with `CPUExecutionProvider`.
- Backward encoder ONNX loads with `CPUExecutionProvider`.
- Policy observation dimension plus 256-D context matches the policy ONNX input.
- Policy action output dimension matches the 29 G1 policy joints.
- MuJoCo runtime XML and canonical teleop G1 XML load.
- Canonical teleop qpos size is 36: root position, root quaternion, and 29 DOF.
- Canonical G1 joint order maps to policy joint order.
- XRoboToolkit SDK imports from an ARM64 CPython 3.10 binding.
- PICO body data gate can be required with `check_xrobot_sdk.py --require-body`.
- Teleop qpos checker distinguishes live retarget qpos from fallback/static qpos.
- Realtime z stream checker validates 256-D finite z packets and stream rate.
- Pure low-state subscriber check received G1 low-state packets, finite 29-motor
  q/dq values, finite IMU values, and decoded wireless remote state without
  creating a command publisher.
- Launcher refuses real control unless `UFO_REAL_ROBOT_OK=1`.

Not system-tested in this pass:

- R2 fault injection.
- Physical e-stop fault injection.
- PICO stream loss during motion.
- Killing the teleop bridge during motion.
- Killing the realtime z server during motion.
- Long-duration free walking.
- Specific action duration or impact-force measurements.

## Dependency Conclusions

This deploy commit does not use the old external GMR stack.

```text
general_motion_retargeting installed: no
GMR required: no
torch required for teleop retargeting: no
```

The online teleop retargeting path uses:

```text
scripts/teleop/motion_tracking_retarget/
XRoboToolkit SDK polling
Mink IK
canonical G1 MuJoCo XML/assets
xrobot_to_g1.json
height alignment
joint-order mapping
```

`qpsolvers` and `daqp`, when installed, are Mink solver dependencies. They are
not old-GMR dependencies.

## Machine-Local Requirements

The following items are intentionally machine-local and should not be committed
as concrete paths or binaries:

- The low-level G1 DDS interface name. Use `G1_INTERFACE=<iface>` or a local
  `ROBOT_CONFIG` overlay.
- The ARM64 XRoboToolkit SDK binary root. Install it with
  `scripts/onboard/install_xrobot_sdk.sh`.
- The OpenSSL 1.1 compatibility library directory required by the current
  prebuilt `g1_interface`. Use `OPENSSL11_LIB=<path>` when needed.
- The CycloneDDS native runtime used by the optional pure Python low-state
  subscriber. Use `CYCLONEDDS_HOME=<path>` when needed.
- The venv `.pth` file created for SDK import.
- Model ONNX binaries and context pkl files.

## Added Diagnostics

The onboard diagnostics are designed to avoid real actuation by default:

- `scripts/onboard/check_g1_onboard_env.py`
- `scripts/onboard/check_deploy_artifacts.py`
- `scripts/onboard/check_policy_preflight.py`
- `scripts/onboard/check_xrobot_sdk.py`
- `scripts/onboard/check_teleop_qpos.py`
- `scripts/onboard/check_z_stream.py`
- `scripts/onboard/check_g1_state_readonly.py`
- `scripts/onboard/run_preflight_suite.sh`

Default no-actuation guarantees:

- No `G1Interface` construction in the offline policy/model checks.
- No `set_control_mode(PR)`.
- No `write_low_command()`.
- No actuator enable.
- No q target publication.
- No real policy launch from `run_preflight_suite.sh`.

`check_g1_state_readonly.py` is the only diagnostic that talks to live G1 DDS
state. It uses the pure Python Unitree `ChannelSubscriber` path and subscribes
to low state only; it does not import or instantiate the command-capable
`g1_interface`.

## Artifact Management

Clean Git checkouts do not contain deployment model binaries. The added manifest:

```text
model/g1_policy/artifact_manifest.yaml
```

tracks required artifact paths and known authoritative hashes. It currently
contains verified hashes for:

```text
model/g1_policy/exported/FBcprAuxModel.onnx
model/g1_policy/exported/backward_encoder.onnx
```

For context pkl and optional metadata files, no authoritative hash was available
during deployment, so the manifest marks hashes as `null` instead of inventing
values. `scripts/onboard/check_deploy_artifacts.py` validates existence, known
SHA256 values, ONNX loadability, ONNX input/output shapes, context shape, and the
policy observation plus context dimension contract.

Large model files remain ignored by Git.

## G1 Interface Configuration

The public `config/robot/g1_real.yaml` must not hard-code a machine-specific DDS
NIC. Real launchers and preflight checks now support:

```bash
G1_INTERFACE=<low-level-dds-interface>
```

The interface validator checks that the named interface exists, is UP, has IPv4,
and is not the default-route interface unless explicitly allowed. It does not
silently auto-select an interface.

## XRoboToolkit SDK Installation

Use:

```bash
scripts/onboard/install_xrobot_sdk.sh \
  --sdk-root /path/to/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64 \
  --venv /path/to/ufo_deploy_venv
```

The script verifies ARM64 ELF output, rejects x86_64 bindings, runs `ldd`,
installs stable symlinks or copies under `external/`, writes an idempotent venv
`.pth`, and verifies `import xrobotoolkit_sdk`.

SDK binaries and symlinks under `external/` remain untracked.

## OpenSSL 1.1 Compatibility

The current prebuilt `g1_interface` may require `libssl.so.1.1` and
`libcrypto.so.1.1` on Ubuntu 22.04 systems that otherwise ship OpenSSL 3.

`run_g1_teleop_policy_onboard.sh` now first tries to import `g1_interface`
without OpenSSL 1.1. If import fails because `libssl.so.1.1` or
`libcrypto.so.1.1` is missing, it checks `OPENSSL11_LIB`, adds it to
`LD_LIBRARY_PATH`, and retries. If the retry fails, it prints `ldd` output and
exits.

The repository may use this conventional local path:

```text
external/openssl-1.1-aarch64
```

The actual OpenSSL library files or symlinks are machine-local and ignored by
Git.

## Scores

- Ordinary onboard Sim2Real: 8.5/10
  - Real G1 operation was user-confirmed.
  - Model/config/no-actuation checks pass.
  - Systematic fault injection and long-duration testing remain separate tasks.
- Onboard teleop Sim2Real: 9/10
  - Real G1 PICO teleop operation was user-confirmed.
  - Live PICO body retarget, realtime z, policy, and G1 actuation path were
    validated.
  - PICO disconnect and process-kill fault cases remain separate tasks.
- Environment installation convenience: 7/10
  - A single deployment venv is sufficient for policy, realtime z, and teleop.
  - Model artifacts, ARM64 XR SDK, OpenSSL 1.1, and G1 interface still require
    explicit setup.
- Documentation completeness: 6/10
  - Onboard setup, no-GMR teleop, artifact checks, SDK install, OpenSSL handling,
    and startup order are now documented.
  - Artifact provenance for every pkl/metadata file still needs upstream release
    documentation.
- Real robot functional readiness: 8/10
  - End-to-end function was confirmed on the real G1.
  - Formal safety fault injection and long-duration validation are not complete.

## Submission Notes

- Core retarget math, XR transforms, Mink IK target logic, height alignment,
  realtime z observation construction, ONNX inference, policy inference, action
  scale, PD gains, safety state machine, R2 latch, stale watchdog, finite checks,
  slew-rate limiting, and ZMQ message formats were not intentionally changed.
- This work should be committed on the diagnostics branch only.
- Do not push directly to upstream `deploy`.
