#!/usr/bin/env python3
"""Read-only onboard deployment preflight for Unitree G1.

This script deliberately avoids creating G1Interface, setting control mode, or
writing low commands. It checks the files, Python ABI surface, model/XML loads,
and joint mapping needed before a real control process is allowed.
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import socket
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class Reporter:
    def __init__(self) -> None:
        self.failures = 0
        self.warnings = 0

    def ok(self, message: str) -> None:
        print(f"[OK] {message}")

    def info(self, message: str) -> None:
        print(f"[INFO] {message}")

    def warn(self, message: str) -> None:
        self.warnings += 1
        print(f"[WARN] {message}")

    def fail(self, message: str) -> None:
        self.failures += 1
        print(f"[FAIL] {message}")

    def skip(self, message: str) -> None:
        print(f"[SKIP] {message}")


PROFILE_CHOICES = ("ordinary", "teleop", "diagnostic", "all")
TARGET_CHOICES = ("workstation", "g1-onboard")
CONTROL_PROFILES = ("ordinary", "teleop", "all")
TELEOP_PROFILES = ("teleop", "all")
DIAGNOSTIC_PROFILES = ("diagnostic", "all")
SUPPORTED_WORKSTATION_ENV = "Workstation: Conda with Python 3.10"
SUPPORTED_ONBOARD_ENV = "G1 onboard: Python 3.10 venv"
RUNTIME_IMPORTS = (
    ("mujoco", "mujoco"),
    ("numpy", "numpy"),
    ("onnxruntime", "onnxruntime"),
    ("scipy", "scipy"),
    ("yaml", "pyyaml"),
    ("zmq", "pyzmq"),
)
G1_INTERFACE_EXT = "g1_interface.cpython-310-aarch64-linux-gnu.so"


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    if not isinstance(data, dict):
        raise TypeError(f"{path} did not load as a mapping")
    return data


def _run_text(args: list[str]) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)


def _check_file(path: Path, label: str, reporter: Reporter) -> bool:
    if path.is_file():
        reporter.ok(f"{label} exists: {path}")
        return True
    reporter.fail(f"{label} missing: {path}")
    return False


def _is_venv_python(
    prefix: str | None = None,
    base_prefix: str | None = None,
    real_prefix: str | None = None,
) -> bool:
    runtime_prefix = prefix if prefix is not None else sys.prefix
    runtime_base_prefix = (
        base_prefix
        if base_prefix is not None
        else getattr(sys, "base_prefix", runtime_prefix)
    )
    runtime_real_prefix = (
        real_prefix
        if real_prefix is not None
        else getattr(sys, "real_prefix", None)
    )
    return runtime_prefix != runtime_base_prefix or runtime_real_prefix is not None


def _is_conda_python(
    prefix: str | None = None,
    executable: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    runtime_prefix = Path(prefix if prefix is not None else sys.prefix)
    runtime_executable = Path(executable if executable is not None else sys.executable)
    runtime_env = os.environ if environ is None else environ

    if (runtime_prefix / "conda-meta").exists():
        return True

    conda_prefix = runtime_env.get("CONDA_PREFIX")
    if conda_prefix:
        try:
            if Path(conda_prefix).resolve(strict=False) == runtime_prefix.resolve(strict=False):
                return True
        except OSError:
            if str(conda_prefix) == str(runtime_prefix):
                return True

    conda_path_markers = {
        "anaconda",
        "anaconda3",
        "conda",
        "miniconda",
        "miniconda3",
        "miniforge",
        "miniforge3",
        "mambaforge",
    }
    path_parts = {part.lower() for part in runtime_executable.parts}
    return bool(path_parts & conda_path_markers)


def _env_problem(reporter: Reporter, allow_nondefault: bool, message: str) -> None:
    if allow_nondefault:
        reporter.warn(f"non-default Python environment override: {message}")
    else:
        reporter.fail(message)


def _check_python_version(reporter: Reporter) -> None:
    version = sys.version_info
    if (version.major, version.minor) == (3, 10):
        reporter.ok("Python version is 3.10")
    else:
        reporter.fail(
            f"Python version is {platform.python_version()}, expected 3.10 for the release runtime ABI"
        )


def _check_python_environment(
    reporter: Reporter,
    target: str,
    allow_nondefault: bool,
) -> None:
    reporter.info(
        "release-supported Python environments: "
        f"{SUPPORTED_WORKSTATION_ENV}; {SUPPORTED_ONBOARD_ENV}"
    )
    reporter.info(f"environment_target: {target}")
    reporter.info(f"python_prefix: {sys.prefix}")
    reporter.info(f"python_base_prefix: {getattr(sys, 'base_prefix', sys.prefix)}")

    _check_python_version(reporter)

    if platform.python_implementation() == "CPython":
        reporter.ok("Python implementation is CPython")
    else:
        _env_problem(
            reporter,
            allow_nondefault,
            f"Python implementation is {platform.python_implementation()}, expected CPython",
        )

    conda_python = _is_conda_python()
    venv_python = _is_venv_python()

    if target == "g1-onboard":
        if venv_python:
            reporter.ok("G1 onboard Python environment is a venv")
        else:
            _env_problem(
                reporter,
                allow_nondefault,
                "validated G1 onboard deployment uses a Python 3.10 venv",
            )

        if conda_python:
            _env_problem(
                reporter,
                allow_nondefault,
                "Conda differs from the validated default G1 onboard environment",
            )
        else:
            reporter.ok("G1 onboard Python environment is not Conda")
        return

    if conda_python:
        reporter.ok("workstation target accepts Conda Python")
    elif venv_python:
        reporter.warn("workstation target is running from venv; Conda is the validated default")
    else:
        reporter.warn("workstation target is not Conda; Conda is the validated default")


def _check_import(module_name: str, label: str, reporter: Reporter) -> object | None:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        reporter.fail(f"{label} missing: {exc.__class__.__name__}: {exc}")
        return None
    origin = getattr(module, "__file__", None)
    if origin:
        reporter.ok(f"{label} import ok: {origin}")
    else:
        reporter.ok(f"{label} import ok")
    return module


def _check_runtime_imports(reporter: Reporter) -> None:
    for module_name, label in RUNTIME_IMPORTS:
        _check_import(module_name, label, reporter)


def _check_onnx(path: Path, reporter: Reporter) -> None:
    import onnxruntime as ort

    try:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        reporter.fail(f"ONNX load failed for {path}: {exc.__class__.__name__}: {exc}")
        return

    reporter.ok(f"ONNX loads with CPUExecutionProvider: {path}")
    reporter.info(f"providers: {sess.get_providers()}")
    reporter.info(f"inputs: {[(x.name, x.shape, x.type) for x in sess.get_inputs()]}")
    reporter.info(f"outputs: {[(x.name, x.shape, x.type) for x in sess.get_outputs()]}")


def _check_xml(path: Path, reporter: Reporter) -> None:
    import mujoco

    try:
        model = mujoco.MjModel.from_xml_path(str(path))
    except Exception as exc:
        reporter.fail(f"MuJoCo XML load failed for {path}: {exc.__class__.__name__}: {exc}")
        return
    reporter.ok(f"MuJoCo XML loads: {path} nq={model.nq} nv={model.nv} njnt={model.njnt}")


def _check_joint_mapping(policy_config: dict[str, Any], reporter: Reporter) -> None:
    sys.path.insert(0, str(ROOT))
    try:
        from utils.strings import unitree_joint_names
    except Exception as exc:
        reporter.fail(f"failed to import unitree_joint_names: {exc.__class__.__name__}: {exc}")
        return

    joint_names = list(policy_config.get("isaac_joint_names", []))
    if len(joint_names) != 29:
        reporter.fail(f"policy isaac_joint_names length is {len(joint_names)}, expected 29")
    else:
        reporter.ok("policy isaac_joint_names length is 29")

    missing = [name for name in joint_names if name not in unitree_joint_names]
    if missing:
        reporter.fail(f"policy joints missing from Unitree order: {missing}")
        return

    permutation = [unitree_joint_names.index(name) for name in joint_names]
    reporter.ok("policy joints all map into Unitree motor order")
    reporter.info(f"policy->unitree permutation: {permutation}")


def _check_g1_import(sdk_lib: Path, openssl11_lib: Path, reporter: Reporter) -> None:
    expected_so = sdk_lib / G1_INTERFACE_EXT
    if sdk_lib.is_dir():
        found = sorted(path.name for path in sdk_lib.glob("g1_interface*.so"))
        if expected_so.is_file():
            reporter.ok(f"g1_interface CPython 3.10 aarch64 extension exists: {expected_so}")
        elif found:
            reporter.fail(
                "g1_interface extension ABI mismatch; expected "
                f"{G1_INTERFACE_EXT}, found {found}"
            )
        else:
            reporter.fail(f"g1_interface extension missing: {expected_so}")
    else:
        reporter.warn(f"UNITREE_SDK_LIB does not exist: {sdk_lib}; trying Python import path")

    code = (
        "import sys; "
        f"sys.path.insert(0, {str(sdk_lib)!r}); "
        "import g1_interface; "
        "print(getattr(g1_interface, 'G1_NUM_MOTOR', None)); "
        "print(getattr(g1_interface, '__file__', ''))"
    )
    env = dict(os.environ)

    def _run_import(extra_ld: str | None = None) -> tuple[int, str]:
        run_env = dict(env)
        if extra_ld:
            run_env["LD_LIBRARY_PATH"] = f"{extra_ld}:{run_env.get('LD_LIBRARY_PATH', '')}"
        proc = subprocess.run(
            [sys.executable, "-c", code],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=run_env,
            check=False,
        )
        return int(proc.returncode), proc.stdout.strip()

    rc, output = _run_import()
    if rc != 0 and ("libssl.so.1.1" in output or "libcrypto.so.1.1" in output) and openssl11_lib.is_dir():
        reporter.warn(f"g1_interface needs OpenSSL 1.1; retrying with {openssl11_lib}")
        rc, output = _run_import(str(openssl11_lib))

    if rc != 0:
        reporter.fail(f"failed to import g1_interface: {output}")
        if expected_so.exists():
            try:
                reporter.info("ldd g1_interface:")
                print(_run_text(["ldd", str(expected_so)]))
            except Exception as exc:
                reporter.warn(f"ldd failed for {expected_so}: {exc}")
        return

    lines = output.splitlines()
    motor_count = lines[0].strip() if lines else ""
    module_file = lines[1].strip() if len(lines) > 1 else ""
    if motor_count == "29":
        reporter.ok("g1_interface imports and G1_NUM_MOTOR=29")
    else:
        reporter.fail(f"g1_interface imported but G1_NUM_MOTOR output={output!r}, expected 29")
    if module_file:
        if module_file.endswith(G1_INTERFACE_EXT):
            reporter.ok(f"g1_interface ABI is CPython 3.10 aarch64: {module_file}")
        else:
            reporter.fail(
                "g1_interface ABI must be CPython 3.10 aarch64; "
                f"imported {module_file}"
            )


def _check_xrobotoolkit_sdk(reporter: Reporter) -> None:
    xrt = _check_import("xrobotoolkit_sdk", "xrobotoolkit_sdk", reporter)
    if xrt is None:
        return

    if hasattr(xrt, "init"):
        reporter.ok("xrobotoolkit_sdk has init()")
    else:
        reporter.fail("xrobotoolkit_sdk missing init()")

    callback_api = all(
        hasattr(xrt, name)
        for name in ("register_frame_callback", "clear_frame_callback", "has_frame_callback")
    )
    polling_api = all(
        hasattr(xrt, name)
        for name in ("is_body_data_available", "get_body_joints_pose", "get_body_timestamp_ns")
    )
    if callback_api:
        reporter.ok("xrobotoolkit_sdk callback API available")
    if polling_api:
        reporter.ok("xrobotoolkit_sdk polling API available")
    if not callback_api and not polling_api:
        reporter.fail("xrobotoolkit_sdk has neither callback nor polling API")


def _check_motion_tracking_retarget(reporter: Reporter) -> None:
    teleop_root = ROOT / "scripts" / "teleop"
    sys.path.insert(0, str(teleop_root))
    try:
        package = importlib.import_module("motion_tracking_retarget")
        joint_mapping = importlib.import_module("motion_tracking_retarget.joint_mapping")
        qpos_size = int(joint_mapping.qpos_size("g1"))
    except Exception as exc:
        reporter.fail(f"motion_tracking_retarget unavailable: {exc.__class__.__name__}: {exc}")
        return
    finally:
        if sys.path and sys.path[0] == str(teleop_root):
            sys.path.pop(0)

    origin = getattr(package, "__file__", "")
    if origin and Path(origin).resolve().is_relative_to(teleop_root / "motion_tracking_retarget"):
        reporter.ok(f"motion_tracking_retarget vendored package import ok: {origin}")
    else:
        reporter.fail(f"motion_tracking_retarget did not resolve to vendored package: {origin}")
    if qpos_size == 36:
        reporter.ok("motion_tracking_retarget G1 qpos_size=36")
    else:
        reporter.fail(f"motion_tracking_retarget G1 qpos_size={qpos_size}, expected 36")


def _check_unitree_sdk2py(reporter: Reporter) -> None:
    _check_import("unitree_sdk2py", "unitree_sdk2py", reporter)
    _check_import("unitree_sdk2py.core.channel", "unitree_sdk2py ChannelSubscriber API", reporter)
    _check_import(
        "unitree_sdk2py.idl.unitree_hg.msg.dds_",
        "unitree_sdk2py G1 low-state IDL",
        reporter,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only G1 onboard deployment preflight")
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default="ordinary")
    parser.add_argument(
        "--target",
        choices=TARGET_CHOICES,
        default=os.environ.get("UFO_DEPLOY_TARGET", "workstation"),
        help=(
            "environment target: workstation allows Conda; g1-onboard validates "
            "the G1 Python 3.10 venv default (default: UFO_DEPLOY_TARGET or workstation)"
        ),
    )
    parser.add_argument("--robot-config", default="config/robot/g1_real.yaml")
    parser.add_argument("--policy-config", default="config/policy/g1_policy.yaml")
    parser.add_argument("--model-path", default="model/g1_policy/exported/FBcprAuxModel.onnx")
    parser.add_argument("--backward-onnx", default="model/g1_policy/exported/backward_encoder.onnx")
    parser.add_argument("--mujoco-xml", default="data/robots/g1/scene_29dof_freebase.xml")
    parser.add_argument("--teleop-xml", default="scripts/teleop/motion_tracking_retarget/assets/g1.xml")
    parser.add_argument(
        "--g1-interface",
        default=os.environ.get("G1_INTERFACE"),
        help="explicit low-level DDS interface override (default: G1_INTERFACE)",
    )
    parser.add_argument(
        "--allow-default-route-interface",
        action="store_true",
        default=os.environ.get("ALLOW_DEFAULT_ROUTE_INTERFACE") == "1",
        help="allow the selected G1 interface to also be the default route (default: ALLOW_DEFAULT_ROUTE_INTERFACE=1)",
    )
    parser.add_argument("--unitree-sdk-lib", default="/home/unitree/unitree_sdk2_bfm/build/lib")
    parser.add_argument("--openssl11-lib", default="external/openssl-1.1-aarch64")
    parser.add_argument("--skip-interface", action="store_true")
    parser.add_argument(
        "--allow-nondefault-python-env",
        action="store_true",
        default=os.environ.get("UFO_ALLOW_NONDEFAULT_ONBOARD_PY") == "1",
        help=(
            "allow non-default onboard Python environments for debugging "
            "(set UFO_ALLOW_NONDEFAULT_ONBOARD_PY=1 to enable)"
        ),
    )
    args = parser.parse_args()

    reporter = Reporter()
    reporter.info("NO-ACTUATION: this script does not create G1Interface, set control mode, or write low commands")
    print(f"Dependency profile: {args.profile}")
    reporter.info(f"repo: {ROOT}")
    reporter.info(f"hostname: {socket.gethostname()}")
    reporter.info(f"python: {sys.executable} {platform.python_version()}")
    reporter.info(f"platform: {platform.platform()} machine={platform.machine()}")
    _check_python_environment(
        reporter,
        target=args.target,
        allow_nondefault=bool(args.allow_nondefault_python_env),
    )

    try:
        reporter.info(f"cpu_online: {Path('/sys/devices/system/cpu/online').read_text().strip()}")
    except Exception as exc:
        reporter.warn(f"could not read cpu_online: {exc}")

    control_profile = args.profile in CONTROL_PROFILES
    teleop_profile = args.profile in TELEOP_PROFILES
    diagnostic_profile = args.profile in DIAGNOSTIC_PROFILES

    if control_profile:
        _check_runtime_imports(reporter)

        robot_config_path = ROOT / args.robot_config
        policy_config_path = ROOT / args.policy_config
        model_path = ROOT / args.model_path
        backward_path = ROOT / args.backward_onnx
        mujoco_xml = ROOT / args.mujoco_xml

        ok_files = [
            _check_file(robot_config_path, "robot config", reporter),
            _check_file(policy_config_path, "policy config", reporter),
            _check_file(model_path, "policy ONNX", reporter),
            _check_file(backward_path, "backward ONNX", reporter),
            _check_file(mujoco_xml, "runtime MuJoCo XML", reporter),
        ]

        robot_config: dict[str, Any] = {}
        policy_config: dict[str, Any] = {}
        if ok_files[0]:
            try:
                robot_config = _read_yaml(robot_config_path)
                reporter.ok(f"robot config loads: ROBOT_TYPE={robot_config.get('ROBOT_TYPE')!r}")
            except Exception as exc:
                reporter.fail(f"robot config load failed: {exc.__class__.__name__}: {exc}")
        if ok_files[1]:
            try:
                policy_config = _read_yaml(policy_config_path)
                reporter.ok("policy config loads")
            except Exception as exc:
                reporter.fail(f"policy config load failed: {exc.__class__.__name__}: {exc}")

        if robot_config:
            if not args.skip_interface:
                try:
                    from scripts.onboard.interface_config import validate_interface

                    validate_interface(
                        str(args.g1_interface or robot_config.get("INTERFACE", "")),
                        reporter,
                        allow_default_route_interface=bool(args.allow_default_route_interface),
                    )
                except Exception as exc:
                    reporter.fail(f"interface validation failed: {exc.__class__.__name__}: {exc}")
            if robot_config.get("ROBOT_TYPE") == "g1_real":
                reporter.ok("ROBOT_TYPE is g1_real")
            else:
                reporter.fail(f"ROBOT_TYPE is {robot_config.get('ROBOT_TYPE')!r}, expected 'g1_real'")

        if policy_config:
            _check_joint_mapping(policy_config, reporter)

        for path in (model_path, backward_path):
            if path.is_file():
                _check_onnx(path, reporter)
        if mujoco_xml.is_file():
            _check_xml(mujoco_xml, reporter)

        openssl11_lib = Path(args.openssl11_lib)
        if not openssl11_lib.is_absolute():
            openssl11_lib = ROOT / openssl11_lib
        _check_g1_import(Path(args.unitree_sdk_lib), openssl11_lib, reporter)
    else:
        reporter.skip("g1_interface (control only)")

    if teleop_profile:
        teleop_xml = ROOT / args.teleop_xml
        if _check_file(teleop_xml, "teleop canonical XML", reporter):
            _check_xml(teleop_xml, reporter)
        _check_xrobotoolkit_sdk(reporter)
        _check_motion_tracking_retarget(reporter)
    else:
        reporter.skip("xrobotoolkit_sdk (teleop only)")
        reporter.skip("motion_tracking_retarget (teleop only)")

    if diagnostic_profile:
        _check_unitree_sdk2py(reporter)
    else:
        reporter.skip("unitree_sdk2py (diagnostic only)")

    if reporter.failures:
        reporter.info(f"summary: {reporter.failures} failure(s), {reporter.warnings} warning(s)")
        return 1
    reporter.info(f"summary: all checks passed, {reporter.warnings} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
