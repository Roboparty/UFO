#!/usr/bin/env python3
"""XRoboToolkit SDK and live data diagnostic for onboard teleop."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.onboard.deploy_helpers import ldd_missing_libraries  # noqa: E402


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


def _run(args: list[str]) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)


def _add_sdk_paths() -> None:
    candidates = [
        os.environ.get("XROBOT_PYBIND_ROOT"),
        str(ROOT / "external/XRoboToolkit-PC-Service-Pybind"),
    ]
    for value in candidates:
        if value and Path(value).is_dir():
            sys.path.insert(0, value)


def _check_elf(path: Path, reporter: Reporter) -> None:
    paths = [path]
    resolved = path.resolve(strict=False)
    if resolved != path:
        paths.append(resolved)
    try:
        for item in paths:
            reporter.info(_run(["file", str(item)]).strip())
    except Exception as exc:
        reporter.warn(f"file failed for {path}: {exc}")
    try:
        ldd = _run(["ldd", str(path)])
        not_found = ldd_missing_libraries(ldd)
        if not_found:
            reporter.fail(f"ldd reports missing libraries for {path}: {not_found}")
        else:
            reporter.ok(f"ldd has no missing libraries for {path}")
    except Exception as exc:
        reporter.warn(f"ldd failed for {path}: {exc}")


def _call_bool(module: Any, name: str) -> bool | None:
    if not hasattr(module, name):
        return None
    try:
        return bool(getattr(module, name)())
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Check xrobotoolkit_sdk import, ABI, APIs, and XR data")
    parser.add_argument("--duration", type=float, default=3.0, help="seconds to poll XR data after init")
    parser.add_argument("--period", type=float, default=0.5, help="seconds between SDK polls")
    parser.add_argument("--require-body", action="store_true", help="fail if body data is never available")
    parser.add_argument("--no-init", action="store_true", help="only import and inspect APIs")
    args = parser.parse_args()

    reporter = Reporter()
    reporter.info(f"python: {sys.executable} {platform.python_version()} machine={platform.machine()}")
    _add_sdk_paths()

    try:
        import xrobotoolkit_sdk as xrt  # type: ignore
    except Exception as exc:
        reporter.fail(f"xrobotoolkit_sdk import failed: {exc.__class__.__name__}: {exc}")
        reporter.info(f"summary: {reporter.failures} failure(s), {reporter.warnings} warning(s)")
        return 1

    module_path = Path(getattr(xrt, "__file__", ""))
    reporter.ok(f"xrobotoolkit_sdk import ok: {module_path}")
    if module_path.is_file():
        _check_elf(module_path, reporter)

    lib_path = ROOT / "external/XRoboToolkit-PC-Service-Pybind/lib/aarch64/libPXREARobotSDK.so"
    if lib_path.exists():
        _check_elf(lib_path, reporter)
    else:
        reporter.warn(f"libPXREARobotSDK.so not found at {lib_path}")

    polling_names = [
        "is_body_data_available",
        "get_body_joints_pose",
        "get_body_timestamp_ns",
        "get_A_button",
        "get_B_button",
        "get_X_button",
        "get_Y_button",
    ]
    callback_names = ["register_frame_callback", "clear_frame_callback", "has_frame_callback"]
    if hasattr(xrt, "init"):
        reporter.ok("xrobotoolkit_sdk has init()")
    else:
        reporter.fail("xrobotoolkit_sdk missing init()")
    reporter.info(f"polling_api: {all(hasattr(xrt, name) for name in polling_names)}")
    reporter.info(f"callback_api: {all(hasattr(xrt, name) for name in callback_names)}")

    if args.no_init or reporter.failures:
        reporter.info(f"summary: {reporter.failures} failure(s), {reporter.warnings} warning(s)")
        return 1 if reporter.failures else 0

    try:
        xrt.init()
        reporter.ok("xrobotoolkit_sdk init() succeeded")
    except Exception as exc:
        reporter.fail(f"xrobotoolkit_sdk init() failed: {exc.__class__.__name__}: {exc}")
        reporter.info(f"summary: {reporter.failures} failure(s), {reporter.warnings} warning(s)")
        return 1

    body_seen = False
    last_body_ts: int | None = None
    last_top_ts: int | None = None
    rows: list[tuple[int, bool | None, int | None, int | None, int | None, int | None, str]] = []
    try:
        deadline = time.monotonic() + max(0.0, float(args.duration))
        i = 0
        while True:
            body = _call_bool(xrt, "is_body_data_available")
            body_ts = None
            top_ts = None
            try:
                if hasattr(xrt, "get_body_timestamp_ns"):
                    body_ts = int(xrt.get_body_timestamp_ns())
            except Exception:
                body_ts = None
            try:
                if hasattr(xrt, "get_time_stamp_ns"):
                    top_ts = int(xrt.get_time_stamp_ns())
            except Exception:
                top_ts = None
            if body:
                body_seen = True

            buttons = []
            for getter, label in [
                ("get_A_button", "A"),
                ("get_B_button", "B"),
                ("get_X_button", "X"),
                ("get_Y_button", "Y"),
            ]:
                value = _call_bool(xrt, getter)
                buttons.append(f"{label}={'?' if value is None else int(value)}")

            rows.append(
                (
                    i,
                    body,
                    body_ts,
                    None if last_body_ts is None or body_ts is None else body_ts - last_body_ts,
                    top_ts,
                    None if last_top_ts is None or top_ts is None else top_ts - last_top_ts,
                    " ".join(buttons),
                )
            )
            last_body_ts = body_ts
            last_top_ts = top_ts
            i += 1
            if time.monotonic() >= deadline:
                break
            time.sleep(max(0.01, float(args.period)))
    finally:
        try:
            if hasattr(xrt, "close"):
                xrt.close()
        except Exception as exc:
            reporter.warn(f"xrobotoolkit_sdk close() failed: {exc.__class__.__name__}: {exc}")

    print("i body body_ts d_body_ts top_ts d_top_ts buttons")
    for row in rows:
        print(*row)

    if body_seen:
        reporter.ok("body data became available during polling")
    else:
        message = "body data was not available during polling"
        if args.require_body:
            reporter.fail(message)
        else:
            reporter.warn(message)

    if reporter.failures:
        reporter.info(f"summary: {reporter.failures} failure(s), {reporter.warnings} warning(s)")
        return 1
    reporter.info(f"summary: all checks passed, {reporter.warnings} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
