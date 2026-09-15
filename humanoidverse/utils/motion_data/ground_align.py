"""Ground alignment for RobotState motions that were retargeted without a floor constraint.

Some retargeting pipelines match keypoints only and do not enforce that the
target robot's collision geometry stays above the floor.  The result is
reference motion in which feet, hands or the whole body sink below z = 0 --
states a physics-based policy can never reach and that pollute the expert
state distribution.

This module runs MuJoCo forward kinematics on the robot MJCF that the robot
YAML points to, measures the deepest world-frame point of every collision
geom per frame ("ground penetration"), and optionally lifts the root
translation so that penetration beyond a small tolerance is removed.  The
per-frame lift is smoothed with a centered moving average so that a sustained
segment (a robot lying on the floor) is lifted fully while a brief toe-dig
spike only receives a small, smooth correction instead of a vertical bounce.

Clips whose residual penetration after correction still exceeds a threshold
are considered broken retargets and can be dropped at manifest build time via
:func:`filter_clips_by_ground_penetration`.

Nothing here resamples fps, changes joint angles, or touches ``ufo_pkl``
sources: alignment is applied only to ``robot_state_*`` datasets that opt in
through the manifest ``ground_alignment`` block.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import mujoco
import numpy as np
from loguru import logger

from humanoidverse.utils.motion_data.robot_state import RobotStateMotion
from humanoidverse.utils.robot_spec import RobotSpec

GROUND_PENETRATION_FIELD = "ground_penetration"
SUPPORTED_MODES = ("none", "per_frame_lift")


@dataclass(frozen=True)
class GroundAlignmentConfig:
    """Options for :func:`apply_ground_alignment`.

    Attributes:
        mode: ``"none"`` only measures penetration; ``"per_frame_lift"`` also
            lifts the root so penetration beyond ``tolerance_m`` is removed.
        tolerance_m: Penetration depth that is left untouched (retargeting noise
            and collision-hull slack).  Metres.
        smooth_seconds: Width of the centered moving-average window applied to
            the per-frame lift.  ``0`` disables smoothing.
        max_residual_penetration_m: If set, training clips whose residual
            penetration (after the lift) exceeds this value are dropped by
            :func:`filter_clips_by_ground_penetration`.  ``None`` keeps all clips.
    """

    mode: str = "per_frame_lift"
    tolerance_m: float = 0.02
    smooth_seconds: float = 0.5
    max_residual_penetration_m: float | None = 0.10

    def __post_init__(self) -> None:
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(f"ground_alignment.mode must be one of {list(SUPPORTED_MODES)}, got {self.mode!r}")
        if not np.isfinite(self.tolerance_m) or self.tolerance_m < 0.0:
            raise ValueError(f"ground_alignment.tolerance_m must be >= 0, got {self.tolerance_m}")
        if not np.isfinite(self.smooth_seconds) or self.smooth_seconds < 0.0:
            raise ValueError(f"ground_alignment.smooth_seconds must be >= 0, got {self.smooth_seconds}")
        if self.max_residual_penetration_m is not None and (
            not np.isfinite(self.max_residual_penetration_m) or self.max_residual_penetration_m <= 0.0
        ):
            raise ValueError(
                f"ground_alignment.max_residual_penetration_m must be > 0 or null, got {self.max_residual_penetration_m}"
            )

    @classmethod
    def from_manifest(cls, raw: Any) -> "GroundAlignmentConfig | None":
        """Build a config from a manifest ``ground_alignment`` block (``None``/``false`` disables)."""

        if raw is None or raw is False:
            return None
        if raw is True:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError("ground_alignment must be a mapping, true, false, or null")
        allowed = {"mode", "tolerance_m", "smooth_seconds", "max_residual_penetration_m"}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"ground_alignment has unknown fields {unknown}; allowed: {sorted(allowed)}")
        kwargs: dict[str, Any] = {}
        if "mode" in raw:
            kwargs["mode"] = str(raw["mode"])
        if "tolerance_m" in raw:
            kwargs["tolerance_m"] = float(raw["tolerance_m"])
        if "smooth_seconds" in raw:
            kwargs["smooth_seconds"] = float(raw["smooth_seconds"])
        if "max_residual_penetration_m" in raw:
            value = raw["max_residual_penetration_m"]
            kwargs["max_residual_penetration_m"] = None if value is None else float(value)
        return cls(**kwargs)


@lru_cache(maxsize=8)
def _load_model(xml_path: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(xml_path)


def _collision_geom_ids(model: mujoco.MjModel) -> list[int]:
    skip = {int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_HFIELD), int(mujoco.mjtGeom.mjGEOM_SDF)}
    return [
        gid
        for gid in range(model.ngeom)
        if (int(model.geom_contype[gid]) != 0 or int(model.geom_conaffinity[gid]) != 0) and int(model.geom_type[gid]) not in skip
    ]


def _mesh_vertices(model: mujoco.MjModel, gid: int) -> np.ndarray:
    mesh_id = int(model.geom_dataid[gid])
    start = int(model.mesh_vertadr[mesh_id])
    count = int(model.mesh_vertnum[mesh_id])
    return np.asarray(model.mesh_vert[start : start + count], dtype=np.float64)


def _geom_lowest_z(model: mujoco.MjModel, data: mujoco.MjData, gid: int, mesh_cache: dict[int, np.ndarray]) -> float:
    """World-frame z of the lowest point of one collision geom (exact for mesh/sphere/capsule/box)."""

    geom_type = int(model.geom_type[gid])
    pos_z = float(data.geom_xpos[gid][2])
    rot = data.geom_xmat[gid].reshape(3, 3)
    size = model.geom_size[gid]
    if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
        verts = mesh_cache.get(gid)
        if verts is None:
            verts = mesh_cache[gid] = _mesh_vertices(model, gid)
        return float((verts @ rot[2]).min() + pos_z)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return pos_z - float(size[0])
    if geom_type in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
        half_axis_z = float(size[1]) * abs(float(rot[2, 2]))
        return pos_z - half_axis_z - float(size[0])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        return pos_z - float(np.abs(rot[2]) @ np.asarray(size[:3], dtype=np.float64))
    if geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        return pos_z - float(np.max(size[:3]))
    return pos_z - float(model.geom_rbound[gid])


def compute_ground_penetration(
    robot_spec: RobotSpec,
    root_pos: np.ndarray,
    root_quat_xyzw: np.ndarray,
    dof_pos: np.ndarray,
) -> np.ndarray:
    """Per-frame ground penetration depth in metres (>= 0) of the robot's collision geometry.

    ``dof_pos`` must follow ``robot_spec.control_joint_names``; ``root_quat_xyzw``
    is the floating-base orientation in xyzw order.  Frames where every
    collision geom is above the floor return 0.
    """

    model = _load_model(str(robot_spec.xml_path))
    data = mujoco.MjData(model)
    if robot_spec.free_joint is None:
        raise ValueError(f"Robot {robot_spec.name} has no free joint; ground alignment needs a floating base")
    root_adr = int(robot_spec.joint_qpos_addr[robot_spec.free_joint])
    dof_adrs = np.asarray([int(robot_spec.joint_qpos_addr[name]) for name in robot_spec.control_joint_names], dtype=np.int64)

    root_pos = np.asarray(root_pos, dtype=np.float64)
    quat = np.asarray(root_quat_xyzw, dtype=np.float64)
    dof = np.asarray(dof_pos, dtype=np.float64)
    if root_pos.shape != (dof.shape[0], 3) or quat.shape != (dof.shape[0], 4) or dof.shape[1] != len(dof_adrs):
        raise ValueError(
            f"compute_ground_penetration shape mismatch: root_pos={root_pos.shape}, root_quat={quat.shape}, dof_pos={dof.shape}"
        )
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)

    geom_ids = _collision_geom_ids(model)
    if not geom_ids:
        raise ValueError(f"Robot XML {robot_spec.xml_path} has no collision geoms; cannot measure ground penetration")
    mesh_cache: dict[int, np.ndarray] = {}
    penetration = np.zeros(dof.shape[0], dtype=np.float32)
    qpos = np.array(model.qpos0, dtype=np.float64)
    for t in range(dof.shape[0]):
        qpos[root_adr : root_adr + 3] = root_pos[t]
        qpos[root_adr + 3 : root_adr + 7] = (quat[t, 3], quat[t, 0], quat[t, 1], quat[t, 2])
        qpos[dof_adrs] = dof[t]
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        lowest = min(_geom_lowest_z(model, data, gid, mesh_cache) for gid in geom_ids)
        penetration[t] = max(0.0, -lowest)
    return penetration


def _centered_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.shape[0] <= 1:
        return np.asarray(values, dtype=np.float64).copy()
    window = int(window) | 1  # force odd so the window is centered
    pad = window // 2
    padded = np.pad(np.asarray(values, dtype=np.float64), (pad, pad), mode="edge")
    kernel = np.full(window, 1.0 / window)
    return np.convolve(padded, kernel, mode="valid")


def apply_ground_alignment(
    motion: RobotStateMotion,
    robot_spec: RobotSpec,
    config: GroundAlignmentConfig,
) -> tuple[RobotStateMotion, np.ndarray, dict[str, Any]]:
    """Measure penetration and (optionally) lift the root of one RobotState motion.

    Returns the (possibly modified) motion, the per-frame residual penetration
    ``[T]`` and a stats dict suitable for motion metadata.
    """

    penetration = compute_ground_penetration(robot_spec, motion.root_pos, motion.root_quat, motion.dof_pos)
    lift = np.zeros_like(penetration, dtype=np.float64)
    if config.mode == "per_frame_lift":
        raw_lift = np.maximum(penetration.astype(np.float64) - config.tolerance_m, 0.0)
        window = int(round(config.smooth_seconds * float(motion.fps)))
        lift = _centered_moving_average(raw_lift, window)
    residual = np.maximum(penetration.astype(np.float64) - lift, 0.0).astype(np.float32)

    stats: dict[str, Any] = {
        "mode": config.mode,
        "tolerance_m": float(config.tolerance_m),
        "smooth_seconds": float(config.smooth_seconds),
        "frames": int(penetration.shape[0]),
        "frames_penetrating": int(np.count_nonzero(penetration > config.tolerance_m)),
        "max_penetration_m": float(penetration.max()) if penetration.size else 0.0,
        "max_lift_m": float(lift.max()) if lift.size else 0.0,
        "mean_lift_m": float(lift.mean()) if lift.size else 0.0,
        "max_residual_penetration_m": float(residual.max()) if residual.size else 0.0,
    }
    if config.mode == "none" or not np.any(lift > 0.0):
        return motion, residual, stats

    root_pos = np.asarray(motion.root_pos, dtype=np.float32).copy()
    root_pos[:, 2] += lift.astype(np.float32)
    aligned = RobotStateMotion(
        motion_key=motion.motion_key,
        root_pos=root_pos,
        root_quat=motion.root_quat,
        dof_pos=motion.dof_pos,
        fps=motion.fps,
        joint_names=motion.joint_names,
        source=motion.source,
        metadata=dict(motion.metadata or {}),
    )
    return aligned, residual, stats


def filter_clips_by_ground_penetration(
    data: dict[str, dict[str, Any]],
    max_penetration_m: float,
    *,
    source_name: str,
) -> tuple[dict[str, Any], list[str]]:
    """Drop clips whose recorded residual ground penetration exceeds ``max_penetration_m``.

    Clips without a ``ground_penetration`` field are kept.  Returns the kept
    dict and the list of dropped motion keys.
    """

    kept: dict[str, Any] = {}
    dropped: list[str] = []
    for motion_key, record in data.items():
        series = record.get(GROUND_PENETRATION_FIELD)
        if series is not None and np.asarray(series).size and float(np.max(series)) > max_penetration_m:
            dropped.append(str(motion_key))
            continue
        kept[motion_key] = record
    if not kept:
        raise ValueError(
            f"ground_alignment.max_residual_penetration_m={max_penetration_m} dropped every clip of source={source_name!r}; "
            "raise the threshold or fix the retargeted motion data"
        )
    if dropped:
        logger.warning(
            f"[ground-align] source={source_name} dropped {len(dropped)}/{len(data)} clips with residual penetration "
            f"> {max_penetration_m:.3f} m (e.g. {dropped[:5]})"
        )
    return kept, dropped


def strip_ground_penetration_field(data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Return a copy of ``data`` without the per-frame penetration series (metadata stats are kept)."""

    stripped: dict[str, Any] = {}
    for motion_key, record in data.items():
        record = dict(record)
        record.pop(GROUND_PENETRATION_FIELD, None)
        stripped[motion_key] = record
    return stripped
