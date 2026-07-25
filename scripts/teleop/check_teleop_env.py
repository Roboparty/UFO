#!/usr/bin/env python3
"""Preflight checks for UFO-Deploy PICO/GMR teleoperation."""

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


DEFAULT_PORTS = (28701, 28702, 28703, 28711)
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
        if name == "general_motion_retargeting":
            # Jetson environments can hit static TLS issues if libtorch is loaded late.
            try:
                importlib.import_module("torch")
            except Exception:
                pass
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
        default=list(DEFAULT_PORTS),
        help="ZMQ ports to check",
    )
    parser.add_argument("--host", default="0.0.0.0", help="bind host used for port checks")
    parser.add_argument("--skip-service", action="store_true", help="skip XRoboToolkit process check")
    parser.add_argument("--skip-ports", action="store_true", help="skip ZMQ port checks")
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

    reporter.info(f"python: {sys.executable}")
    _import_module("general_motion_retargeting", "GMR", reporter)
    xrt = _import_module("xrobotoolkit_sdk", "xrobotoolkit_sdk", reporter)
    _import_module("zmq", "pyzmq", reporter)

    if xrt is not None:
        _check_xrobotoolkit_api(xrt, reporter)

    if not args.skip_service:
        _check_service(_parse_patterns(args.service_pattern), reporter)

    if not args.skip_ports:
        _check_ports(args.host, args.ports, args.mode, reporter)

    if args.xr_data:
        _check_xr_data(xrt, reporter)

    if reporter.failures:
        reporter.info(f"summary: {reporter.failures} failure(s), {reporter.warnings} warning(s)")
        return 1

    reporter.info(f"summary: all checks passed, {reporter.warnings} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
