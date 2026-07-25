#!/usr/bin/env python3
"""Preflight checks for UFO-Deploy PICO/XRobot teleoperation."""

from __future__ import annotations

import argparse
import errno
import importlib
import os
import socket
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable


TELEOP_PORTS = (28701, 28702, 28703)
REALTIME_PORTS = (28711,)
PORT_PROFILES = {
    "teleop": TELEOP_PORTS,
    "realtime": REALTIME_PORTS,
    "all": TELEOP_PORTS + REALTIME_PORTS,
}
DEFAULT_SERVICE_PATTERNS = (
    "RoboticsServiceProcess",
    "roboticsservice",
    "XRoboToolkit",
    "PXREARobotSDK",
)
CALLBACK_APIS = (
    "register_frame_callback",
    "clear_frame_callback",
    "has_frame_callback",
)
POLLING_APIS = (
    "is_body_data_available",
    "get_body_joints_pose",
    "get_body_timestamp_ns",
    "get_A_button",
    "get_B_button",
    "get_X_button",
    "get_Y_button",
)


class Reporter:
    def __init__(self) -> None:
        self.failures = 0
        self.warnings = 0

    def ok(self, message: str) -> None:
        print(f"[OK] {message}")

    def fail(self, message: str) -> None:
        self.failures += 1
        print(f"[FAIL] {message}")

    def warn(self, message: str) -> None:
        self.warnings += 1
        print(f"[WARN] {message}")

    def info(self, message: str) -> None:
        print(f"[INFO] {message}")


def _module_origin(module: ModuleType) -> str:
    origin = getattr(module, "__file__", None)
    return str(origin) if origin else "built-in"


def _import_module(name: str, label: str, reporter: Reporter) -> ModuleType | None:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        reporter.fail(f"{label} missing: {exc.__class__.__name__}: {exc}")
        return None

    reporter.ok(f"{label} installed")
    reporter.info(f"{name}: {_module_origin(module)}")
    return module


def _check_xrobotoolkit_api(xrt: ModuleType, reporter: Reporter) -> None:
    if not hasattr(xrt, "init"):
        reporter.fail("xrobotoolkit_sdk installed but missing init()")
        return

    has_callback = all(hasattr(xrt, name) for name in CALLBACK_APIS)
    has_polling = all(hasattr(xrt, name) for name in POLLING_APIS)

    if has_callback:
        reporter.ok("xrobotoolkit_sdk callback API available")
    if has_polling:
        reporter.ok("xrobotoolkit_sdk polling API available")
    if not has_callback and not has_polling:
        missing_callback = ", ".join(name for name in CALLBACK_APIS if not hasattr(xrt, name))
        missing_polling = ", ".join(name for name in POLLING_APIS if not hasattr(xrt, name))
        reporter.fail(
            "xrobotoolkit_sdk has neither callback nor polling API "
            f"(callback missing: {missing_callback}; polling missing: {missing_polling})"
        )


def _check_canonical_retarget(reporter: Reporter) -> None:
    try:
        import mujoco as mj
        from motion_tracking_retarget.joint_mapping import (
            build_joint_permutation,
            canonical_joint_names,
            policy_joint_names,
            qpos_size,
        )
        from motion_tracking_retarget.params import XR_BODY_JOINT_NAMES, load_xrobot_ik_config, resolve_robot_xml_path
        from motion_tracking_retarget.robot_config import load_teleop_robot_config
    except Exception as exc:
        reporter.fail(f"motion_tracking_retarget unavailable: {exc.__class__.__name__}: {exc}")
        return

    reporter.ok("motion_tracking_retarget package available")

    if len(XR_BODY_JOINT_NAMES) == 24:
        reporter.ok("XR body joint names: 24")
    else:
        reporter.fail(f"XR body joint names must be 24, got {len(XR_BODY_JOINT_NAMES)}")

    try:
        cfg = load_teleop_robot_config("g1")
    except Exception as exc:
        reporter.fail(f"teleop config invalid: {exc.__class__.__name__}: {exc}")
        return
    reporter.ok("config/teleop/g1.yaml loads")
    if cfg.robot_key == "g1":
        reporter.ok("teleop robot_key: g1")
    else:
        reporter.fail(f"teleop robot_key must be g1, got {cfg.robot_key}")
    if cfg.calibration_button is None:
        reporter.ok("calibration button default: null")
    else:
        reporter.fail(f"calibration button must default to null, got {cfg.calibration_button}")

    try:
        xml_path = resolve_robot_xml_path("g1")
        if xml_path.is_file():
            reporter.ok(f"canonical G1 XML exists: {xml_path}")
        else:
            reporter.fail(f"canonical G1 XML missing: {xml_path}")
            return
        model = mj.MjModel.from_xml_path(str(xml_path))
    except Exception as exc:
        reporter.fail(f"canonical G1 XML failed to load: {exc.__class__.__name__}: {exc}")
        return
    reporter.ok("canonical G1 XML loads with MuJoCo")

    if int(model.nq) == 36:
        reporter.ok("canonical qpos_size: 36")
    else:
        reporter.fail(f"canonical qpos_size must be 36, got {int(model.nq)}")

    body_names = {
        str(mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id))
        for body_id in range(model.nbody)
        if mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id) is not None
    }
    for toe_name in ("left_toe_link", "right_toe_link"):
        if toe_name in body_names:
            reporter.ok(f"toe body exists: {toe_name}")
        else:
            reporter.fail(f"toe body missing: {toe_name}")

    try:
        ik_config = load_xrobot_ik_config("g1")
    except Exception as exc:
        reporter.fail(f"xrobot_to_g1.json invalid: {exc.__class__.__name__}: {exc}")
        return
    required_keys = (
        "human_height_assumption",
        "human_scale_table",
        "ik_match_table1",
        "ik_match_table2",
        "human_root_name",
        "robot_root_name",
        "ground_height",
    )
    missing_keys = [key for key in required_keys if key not in ik_config]
    if missing_keys:
        reporter.fail(f"xrobot_to_g1.json missing keys: {', '.join(missing_keys)}")
    else:
        reporter.ok("xrobot_to_g1.json required fields present")

    try:
        canonical_names = canonical_joint_names("g1")
        output_names = policy_joint_names()
        permutation = build_joint_permutation(canonical_names, output_names)
    except Exception as exc:
        reporter.fail(f"joint permutation invalid: {exc.__class__.__name__}: {exc}")
        return

    if len(canonical_names) == 29:
        reporter.ok("canonical G1 joint names: 29")
    else:
        reporter.fail(f"canonical G1 joint names must be 29, got {len(canonical_names)}")
    if len(output_names) == 29:
        reporter.ok("policy G1 joint names: 29")
    else:
        reporter.fail(f"policy G1 joint names must be 29, got {len(output_names)}")
    if int(qpos_size("g1")) == 36:
        reporter.ok("joint mapping qpos_size: 36")
    else:
        reporter.fail(f"joint mapping qpos_size must be 36, got {int(qpos_size('g1'))}")
    reporter.ok("canonical -> UFO joint permutation valid")
    if list(permutation) == list(range(len(permutation))):
        reporter.info("joint_permutation: identity")
    else:
        reporter.info(f"joint_permutation: {permutation.tolist()}")


def _check_viewer_dependencies(reporter: Reporter) -> None:
    _import_module("viser", "viser", reporter)
    _import_module("mjviser", "mjviser", reporter)


def _read_cmdline(pid_dir: Path) -> str:
    try:
        raw = (pid_dir / "cmdline").read_bytes()
        if raw:
            return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    except OSError:
        pass

    try:
        return (pid_dir / "comm").read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def _find_service_processes(patterns: Iterable[str]) -> list[tuple[int, str]]:
    current_pid = os.getpid()
    lowered = [pattern.lower() for pattern in patterns if pattern]
    matches: list[tuple[int, str]] = []

    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        pid = int(pid_dir.name)
        if pid == current_pid:
            continue
        cmdline = _read_cmdline(pid_dir)
        if not cmdline:
            continue
        haystack = cmdline.lower()
        if any(pattern in haystack for pattern in lowered):
            matches.append((pid, cmdline))

    return matches


def _check_service(patterns: Iterable[str], reporter: Reporter) -> None:
    patterns = tuple(patterns)
    matches = _find_service_processes(patterns)
    if matches:
        pids = ", ".join(str(pid) for pid, _ in matches[:5])
        reporter.ok(f"XRoboToolkit service running (pid {pids})")
        for pid, cmdline in matches[:3]:
            reporter.info(f"service candidate {pid}: {cmdline[:180]}")
        if len(matches) > 3:
            reporter.info(f"{len(matches) - 3} additional service candidates hidden")
        return

    reporter.fail("XRoboToolkit service not running")
    reporter.info("matched process patterns: " + ", ".join(patterns))
    launcher = Path("/opt/apps/roboticsservice/runService.sh")
    if launcher.exists():
        reporter.info(f"headless service launcher exists: {launcher}")


def _port_available(host: str, port: int) -> tuple[bool, str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            return False, "address already in use"
        return False, f"{exc.__class__.__name__}: {exc}"
    finally:
        sock.close()
    return True, "available"


def _check_ports(host: str, ports: Iterable[int], mode: str, reporter: Reporter) -> None:
    for port in ports:
        available, reason = _port_available(host, int(port))
        if mode == "preflight":
            if available:
                reporter.ok(f"port {port} available")
            else:
                reporter.fail(f"port {port} unavailable: {reason}")
        else:
            if available:
                reporter.fail(f"port {port} not occupied")
            else:
                reporter.ok(f"port {port} occupied ({reason})")


def _safe_sdk_call(xrt: ModuleType, name: str, label: str, reporter: Reporter) -> object | None:
    if not hasattr(xrt, name):
        reporter.fail(f"{label} check unavailable: xrobotoolkit_sdk missing {name}()")
        return None

    try:
        value = getattr(xrt, name)()
    except Exception as exc:
        reporter.fail(f"{label} check failed: {exc.__class__.__name__}: {exc}")
        return None

    if value is None:
        reporter.fail(f"{label} unavailable: {name}() returned None")
        return None

    reporter.ok(f"{label} available")
    return value


def _check_xr_data(xrt: ModuleType | None, reporter: Reporter) -> None:
    if xrt is None:
        reporter.fail("XR data check skipped because xrobotoolkit_sdk is missing")
        return

    if not hasattr(xrt, "init"):
        reporter.fail("XR data check unavailable: xrobotoolkit_sdk missing init()")
        return

    initialized = False
    try:
        xrt.init()
        initialized = True
        reporter.ok("xrobotoolkit_sdk init() succeeded")
    except Exception as exc:
        reporter.fail(f"xrobotoolkit_sdk init() failed: {exc.__class__.__name__}: {exc}")
        return
    finally:
        if not initialized:
            close = getattr(xrt, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    try:
        if hasattr(xrt, "is_body_data_available"):
            try:
                body_available = bool(xrt.is_body_data_available())
            except Exception as exc:
                reporter.fail(f"body data check failed: {exc.__class__.__name__}: {exc}")
            else:
                if body_available:
                    reporter.ok("body data available")
                else:
                    reporter.fail("body data unavailable")
        elif hasattr(xrt, "has_frame_callback"):
            reporter.warn("body data availability cannot be polled with this callback-only SDK")
        else:
            reporter.fail("body data check unavailable")

        _safe_sdk_call(xrt, "get_headset_pose", "headset pose", reporter)
        _safe_sdk_call(xrt, "get_left_controller_pose", "left controller pose", reporter)
        _safe_sdk_call(xrt, "get_right_controller_pose", "right controller pose", reporter)
    finally:
        close = getattr(xrt, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                reporter.warn(f"xrobotoolkit_sdk close() failed: {exc.__class__.__name__}: {exc}")


def _parse_patterns(extra_patterns: list[str]) -> list[str]:
    patterns = list(DEFAULT_SERVICE_PATTERNS)
    env_patterns = os.environ.get("XROBOT_SERVICE_PATTERNS", "")
    if env_patterns:
        patterns.extend(part.strip() for part in env_patterns.split(",") if part.strip())
    patterns.extend(extra_patterns)
    return patterns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check UFO-Deploy teleop environment")
    parser.add_argument(
        "--mode",
        choices=("preflight", "running"),
        default="preflight",
        help="preflight expects ports to be free; running expects them to be occupied",
    )
    parser.add_argument(
        "--ports",
        nargs="+",
        type=int,
        default=None,
        help="ZMQ ports to check; overrides --port-profile",
    )
    parser.add_argument(
        "--port-profile",
        choices=tuple(PORT_PROFILES),
        default="teleop",
        help="named ZMQ port set to check when --ports is not provided",
    )
    parser.add_argument("--host", default="0.0.0.0", help="bind host used for port checks")
    parser.add_argument("--skip-service", action="store_true", help="skip XRoboToolkit process check")
    parser.add_argument("--skip-ports", action="store_true", help="skip ZMQ port checks")
    parser.add_argument("--skip-canonical-retarget", action="store_true", help="skip vendored canonical retarget checks")
    parser.add_argument("--web-visualize", action="store_true", help="also check optional web viewer dependencies")
    parser.add_argument("--xr-data", action="store_true", help="also query live XR body/headset/controller data")
    parser.add_argument(
        "--service-pattern",
        action="append",
        default=[],
        help="additional process substring used to detect XRoboToolkit service",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reporter = Reporter()
    ports = args.ports if args.ports is not None else list(PORT_PROFILES[args.port_profile])

    reporter.info(f"python: {sys.executable}")
    _import_module("mink", "mink", reporter)
    _import_module("mujoco", "mujoco", reporter)
    _import_module("numpy", "numpy", reporter)
    _import_module("scipy", "scipy", reporter)
    _import_module("yaml", "pyyaml", reporter)
    _import_module("zmq", "pyzmq", reporter)
    xrt = _import_module("xrobotoolkit_sdk", "xrobotoolkit_sdk", reporter)

    if xrt is not None:
        _check_xrobotoolkit_api(xrt, reporter)

    if not args.skip_canonical_retarget:
        _check_canonical_retarget(reporter)

    if args.web_visualize or os.environ.get("WEB_VISUALIZE", "0") == "1":
        _check_viewer_dependencies(reporter)

    if not args.skip_service:
        _check_service(_parse_patterns(args.service_pattern), reporter)

    if not args.skip_ports:
        reporter.info(f"port_profile: {args.port_profile}")
        _check_ports(args.host, ports, args.mode, reporter)

    if args.xr_data:
        _check_xr_data(xrt, reporter)

    if reporter.failures:
        reporter.info(f"summary: {reporter.failures} failure(s), {reporter.warnings} warning(s)")
        return 1

    reporter.info(f"summary: all checks passed, {reporter.warnings} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
