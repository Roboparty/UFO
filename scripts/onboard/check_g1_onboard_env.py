#!/usr/bin/env python3
"""Read-only onboard deployment preflight for Unitree G1.

This script deliberately avoids creating G1Interface, setting control mode, or
writing low commands. It checks the files, Python ABI surface, model/XML loads,
and joint mapping needed before a real control process is allowed.
"""

from __future__ import annotations

import argparse
import os
import platform
import socket
import subprocess
import sys
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
    if not sdk_lib.is_dir():
        reporter.warn(f"UNITREE_SDK_LIB does not exist: {sdk_lib}")
        return

    code = (
        "import sys; "
        f"sys.path.insert(0, {str(sdk_lib)!r}); "
        "import g1_interface; "
        "print(getattr(g1_interface, 'G1_NUM_MOTOR', None))"
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
        so_path = sdk_lib / "g1_interface.cpython-310-aarch64-linux-gnu.so"
        if so_path.exists():
            try:
                reporter.info("ldd g1_interface:")
                print(_run_text(["ldd", str(so_path)]))
            except Exception as exc:
                reporter.warn(f"ldd failed for {so_path}: {exc}")
        return

    if output.splitlines()[-1].strip() == "29":
        reporter.ok("g1_interface imports and G1_NUM_MOTOR=29")
    else:
        reporter.fail(f"g1_interface imported but G1_NUM_MOTOR output={output!r}, expected 29")


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only G1 onboard deployment preflight")
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
    args = parser.parse_args()

    reporter = Reporter()
    reporter.info("NO-ACTUATION: this script does not create G1Interface, set control mode, or write low commands")
    reporter.info(f"repo: {ROOT}")
    reporter.info(f"hostname: {socket.gethostname()}")
    reporter.info(f"python: {sys.executable} {platform.python_version()}")
    reporter.info(f"platform: {platform.platform()} machine={platform.machine()}")

    try:
        reporter.info(f"cpu_online: {Path('/sys/devices/system/cpu/online').read_text().strip()}")
    except Exception as exc:
        reporter.warn(f"could not read cpu_online: {exc}")

    robot_config_path = ROOT / args.robot_config
    policy_config_path = ROOT / args.policy_config
    model_path = ROOT / args.model_path
    backward_path = ROOT / args.backward_onnx
    mujoco_xml = ROOT / args.mujoco_xml
    teleop_xml = ROOT / args.teleop_xml

    ok_files = [
        _check_file(robot_config_path, "robot config", reporter),
        _check_file(policy_config_path, "policy config", reporter),
        _check_file(model_path, "policy ONNX", reporter),
        _check_file(backward_path, "backward ONNX", reporter),
        _check_file(mujoco_xml, "runtime MuJoCo XML", reporter),
        _check_file(teleop_xml, "teleop canonical XML", reporter),
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
    for path in (mujoco_xml, teleop_xml):
        if path.is_file():
            _check_xml(path, reporter)

    openssl11_lib = Path(args.openssl11_lib)
    if not openssl11_lib.is_absolute():
        openssl11_lib = ROOT / openssl11_lib
    _check_g1_import(Path(args.unitree_sdk_lib), openssl11_lib, reporter)

    if reporter.failures:
        reporter.info(f"summary: {reporter.failures} failure(s), {reporter.warnings} warning(s)")
        return 1
    reporter.info(f"summary: all checks passed, {reporter.warnings} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
