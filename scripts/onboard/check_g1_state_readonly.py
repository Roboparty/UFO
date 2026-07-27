#!/usr/bin/env python3
"""Strict low-state subscriber check for G1 without command publishing.

This script uses the pure Python Unitree SDK ChannelSubscriber path. It does not
import or instantiate g1_interface, does not create a command publisher, does
not set PR mode, and does not write low commands.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
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
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} did not load as a mapping")
    return data


def _as_array(value: Any) -> np.ndarray:
    try:
        return np.asarray(list(value), dtype=np.float64)
    except TypeError:
        return np.asarray(value, dtype=np.float64)


def _decode_wireless_remote(raw: Any) -> dict[str, Any]:
    data = [int(x) for x in list(raw)]
    if len(data) < 24:
        raise ValueError(f"wireless_remote length is {len(data)}, expected at least 24")
    button1 = [int(bit) for bit in f"{data[2]:08b}"]
    button2 = [int(bit) for bit in f"{data[3]:08b}"]
    return {
        "LT": button1[2],
        "RT": button1[3],
        "back": button1[4],
        "start": button1[5],
        "LB": button1[6],
        "RB": button1[7],
        "left": button2[0],
        "down": button2[1],
        "right": button2[2],
        "up": button2[3],
        "Y": button2[4],
        "X": button2[5],
        "B": button2[6],
        "A": button2[7],
        "lx": struct.unpack("f", bytes(data[4:8]))[0],
        "rx": struct.unpack("f", bytes(data[8:12]))[0],
        "ry": struct.unpack("f", bytes(data[12:16]))[0],
        "ly": struct.unpack("f", bytes(data[20:24]))[0],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read G1 low state via subscriber only")
    parser.add_argument("--robot-config", default="config/robot/g1_real.yaml")
    parser.add_argument("--interface", default=os.environ.get("G1_INTERFACE", ""))
    parser.add_argument("--unitree-sdk-python", default=os.environ.get("UNITREE_SDK_PYTHON", ""))
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--domain-id", type=int, default=None)
    parser.add_argument(
        "--allow-default-route-interface",
        action="store_true",
        default=os.environ.get("ALLOW_DEFAULT_ROUTE_INTERFACE") == "1",
        help="allow selected interface to also be the default route (default: ALLOW_DEFAULT_ROUTE_INTERFACE=1)",
    )
    args = parser.parse_args()

    reporter = Reporter()
    reporter.info("NO-ACTUATION: subscriber only; no G1Interface, no command publisher, no control mode, no low command")
    reporter.info(f"python: {sys.executable}")

    if args.unitree_sdk_python:
        sys.path.insert(0, str(Path(args.unitree_sdk_python).expanduser()))

    try:
        from scripts.onboard.interface_config import validate_interface
    except Exception as exc:
        reporter.fail(f"failed to import interface validator: {exc.__class__.__name__}: {exc}")
        return 1

    config_path = Path(args.robot_config).expanduser()
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    try:
        robot_config = _read_yaml(config_path)
    except Exception as exc:
        reporter.fail(f"failed to read robot config: {exc.__class__.__name__}: {exc}")
        return 1

    iface = str(args.interface or robot_config.get("INTERFACE", "")).strip()
    if not validate_interface(
        iface,
        reporter,
        allow_default_route_interface=bool(args.allow_default_route_interface),
    ):
        return 1
    domain_id = int(args.domain_id if args.domain_id is not None else robot_config.get("DOMAIN_ID", 0))

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    except Exception as exc:
        reporter.fail(f"failed to import pure Python Unitree subscriber API: {exc.__class__.__name__}: {exc}")
        reporter.info("Set UNITREE_SDK_PYTHON to a unitree_sdk2_python checkout if needed.")
        return 1

    packets: list[tuple[float, Any]] = []

    def _handler(msg: Any) -> None:
        packets.append((time.monotonic(), msg))

    try:
        ChannelFactoryInitialize(domain_id, iface)
        sub = ChannelSubscriber("rt/lowstate", LowState_)
        sub.Init(_handler, 10)
    except Exception as exc:
        reporter.fail(f"failed to initialize low-state subscriber: {exc.__class__.__name__}: {exc}")
        return 1

    deadline = time.monotonic() + max(0.0, float(args.duration))
    while time.monotonic() < deadline:
        time.sleep(0.05)

    if not packets:
        reporter.fail("no low-state packets received")
        reporter.info(f"summary: {reporter.failures} failure(s), {reporter.warnings} warning(s)")
        return 1

    reporter.ok(f"received {len(packets)} low-state packet(s)")
    if len(packets) > 1:
        elapsed = max(1e-9, packets[-1][0] - packets[0][0])
        reporter.info(f"receive_rate_hz: {(len(packets) - 1) / elapsed:.2f}")
        ticks = [getattr(item[1], "tick", None) for item in packets]
        if all(tick is not None for tick in ticks):
            tick_values = [int(tick) for tick in ticks]
            non_monotonic = sum(
                1 for before, after in zip(tick_values, tick_values[1:]) if after <= before
            )
            reporter.info(
                f"tick_start={tick_values[0]} tick_end={tick_values[-1]} "
                f"tick_non_monotonic_count={non_monotonic}"
            )

    msg = packets[-1][1]
    motor_state = list(getattr(msg, "motor_state", []))
    if len(motor_state) < 29:
        reporter.fail(f"low-state motor_state has {len(motor_state)} entries, expected at least 29")
    else:
        q = np.asarray([motor_state[i].q for i in range(29)], dtype=np.float64)
        dq = np.asarray([motor_state[i].dq for i in range(29)], dtype=np.float64)
        if np.isfinite(q).all():
            reporter.ok("29 motor q values are finite")
        else:
            reporter.fail("non-finite motor q values observed")
        if np.isfinite(dq).all():
            reporter.ok("29 motor dq values are finite")
        else:
            reporter.fail("non-finite motor dq values observed")

    imu = getattr(msg, "imu_state", None)
    if imu is None:
        reporter.fail("low-state IMU missing")
    else:
        imu_values = []
        for name in ("gyroscope", "rpy", "quaternion", "accelerometer"):
            value = getattr(imu, name, None)
            if value is not None:
                imu_values.append(_as_array(value))
        if imu_values and all(np.isfinite(v).all() for v in imu_values):
            reporter.ok("IMU values are finite")
        else:
            reporter.fail("IMU values missing or non-finite")

    mode_machine = getattr(msg, "mode_machine", None)
    reporter.info(f"mode_machine: {mode_machine}")

    wireless_remote = getattr(msg, "wireless_remote", None)
    if wireless_remote is None:
        reporter.warn("wireless_remote missing from low-state packet")
    else:
        try:
            remote = _decode_wireless_remote(wireless_remote)
            reporter.info(
                "wireless buttons: "
                f"A={remote['A']} B={remote['B']} X={remote['X']} Y={remote['Y']} "
                f"LB={remote['LB']} RB={remote['RB']} LT={remote['LT']} RT={remote['RT']} "
                f"start={remote['start']} back={remote['back']}"
            )
            reporter.info(
                "wireless axes: "
                f"lx={remote['lx']:.3f} ly={remote['ly']:.3f} "
                f"rx={remote['rx']:.3f} ry={remote['ry']:.3f}"
            )
        except Exception as exc:
            reporter.warn(f"failed to decode wireless_remote: {exc.__class__.__name__}: {exc}")

    if reporter.failures:
        reporter.info(f"summary: {reporter.failures} failure(s), {reporter.warnings} warning(s)")
        return 1
    reporter.info(f"summary: all checks passed, {reporter.warnings} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
