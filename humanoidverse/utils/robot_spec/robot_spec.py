"""RobotSpec dataclass used by robot-state motion adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RobotSpec:
    name: str
    xml_path: str
    base_body: str
    root_joint: str | None
    free_joint: str | None
    joint_names: list[str]
    control_joint_names: list[str]
    body_names: list[str]
    actuator_names: list[str]
    actuator_joint_names: list[str]
    joint_types: dict[str, str]
    joint_axes: dict[str, list[float]]
    joint_ranges: dict[str, list[float] | None]
    joint_qpos_addr: dict[str, int]
    joint_dof_addr: dict[str, int]
    joint_body_names: dict[str, str]
    body_parent: dict[str, str | None]
    key_bodies: list[str]
    feet: list[str]
    hands: list[str]
    default_dof_pos: dict[str, float]
    root_quat_order: str
    coordinate_system: str
    dof_unit: str
    nq: int
    nv: int
    nu: int

    @property
    def xml_file(self) -> Path:
        return Path(self.xml_path)
