#!/usr/bin/env python3
"""Validate and optionally overlay the G1 DDS network interface.

This helper is intentionally limited to file/network-interface checks. It does
not import G1Interface, create DDS publishers, set control mode, or write robot
commands.
"""

from __future__ import annotations

import argparse
import ipaddress
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class InterfaceInfo:
    name: str
    state: str
    addresses: tuple[str, ...]

    @property
    def ipv4_addresses(self) -> tuple[str, ...]:
        out: list[str] = []
        for item in self.addresses:
            try:
                if ipaddress.ip_interface(item).version == 4:
                    out.append(item)
            except ValueError:
                continue
        return tuple(out)


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


def _run_text(args: list[str]) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)


def parse_ip_br_addr(text: str) -> dict[str, InterfaceInfo]:
    interfaces: dict[str, InterfaceInfo] = {}
    for raw in text.splitlines():
        parts = raw.split()
        if len(parts) < 2:
            continue
        name, state = parts[0], parts[1]
        interfaces[name] = InterfaceInfo(name=name, state=state, addresses=tuple(parts[2:]))
    return interfaces


def parse_default_route_interfaces(text: str) -> set[str]:
    ifaces: set[str] = set()
    for raw in text.splitlines():
        parts = raw.split()
        if not parts or parts[0] != "default":
            continue
        if "dev" in parts:
            idx = parts.index("dev")
            if idx + 1 < len(parts):
                ifaces.add(parts[idx + 1])
    return ifaces


def load_interfaces() -> dict[str, InterfaceInfo]:
    return parse_ip_br_addr(_run_text(["ip", "-br", "addr"]))


def load_default_route_interfaces() -> set[str]:
    try:
        return parse_default_route_interfaces(_run_text(["ip", "route", "show", "default"]))
    except Exception:
        return set()


def validate_interface(
    name: str,
    reporter: Reporter,
    *,
    interfaces: dict[str, InterfaceInfo] | None = None,
    default_route_ifaces: set[str] | None = None,
    allow_default_route_interface: bool = False,
) -> bool:
    name = str(name or "").strip()
    if not name:
        reporter.fail("G1 interface is empty; set G1_INTERFACE or provide a local ROBOT_CONFIG")
        return False

    if interfaces is None:
        interfaces = load_interfaces()
    if default_route_ifaces is None:
        default_route_ifaces = load_default_route_interfaces()

    reporter.info("detected non-loopback interfaces:")
    for item in interfaces.values():
        if item.name != "lo":
            reporter.info(f"  {item.name} {item.state} {' '.join(item.addresses)}")

    item = interfaces.get(name)
    if item is None:
        reporter.fail(f"G1 interface {name!r} was not found; set G1_INTERFACE explicitly")
        return False
    if item.name == "lo":
        reporter.fail("G1 interface must not be loopback")
        return False
    if item.state != "UP":
        reporter.fail(f"G1 interface {name!r} is present but not UP: {item.state}")
        return False
    if not item.ipv4_addresses:
        reporter.fail(f"G1 interface {name!r} has no IPv4 address")
        return False
    if item.name in default_route_ifaces and not allow_default_route_interface:
        reporter.fail(
            f"G1 interface {name!r} is the default-route interface; this is usually Wi-Fi/PICO networking. "
            "Use the low-level DDS NIC or pass --allow-default-route-interface only if this is intentional."
        )
        return False

    reporter.ok(f"G1 interface {name!r} is present, UP, and has IPv4 {list(item.ipv4_addresses)}")
    return True


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} did not load as a mapping")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    import yaml

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and optionally overlay the G1 DDS interface")
    parser.add_argument("--robot-config", default=str(ROOT / "config/robot/g1_real.yaml"))
    parser.add_argument("--interface", default=None, help="explicit low-level DDS interface, usually from G1_INTERFACE")
    parser.add_argument("--output", default=None, help="write an overlay robot config with INTERFACE replaced")
    parser.add_argument("--allow-default-route-interface", action="store_true")
    args = parser.parse_args()

    reporter = Reporter()
    config_path = Path(args.robot_config).expanduser()
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    try:
        config = _read_yaml(config_path)
    except Exception as exc:
        reporter.fail(f"failed to load robot config {config_path}: {exc.__class__.__name__}: {exc}")
        return 1

    requested = str(args.interface or config.get("INTERFACE", "")).strip()
    if args.interface:
        reporter.info(f"explicit G1_INTERFACE override: {requested}")
    else:
        reporter.info(f"using INTERFACE from robot config: {requested}")

    if not validate_interface(
        requested,
        reporter,
        allow_default_route_interface=bool(args.allow_default_route_interface),
    ):
        reporter.info(f"summary: {reporter.failures} failure(s), {reporter.warnings} warning(s)")
        return 1

    if args.output:
        out_path = Path(args.output).expanduser()
        config["INTERFACE"] = requested
        _write_yaml(out_path, config)
        reporter.ok(f"wrote robot config overlay: {out_path}")

    reporter.info(f"summary: all checks passed, {reporter.warnings} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
