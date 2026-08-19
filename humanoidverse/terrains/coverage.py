"""Preflight checks for keeping complete motion clips inside one terrain patch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np


@dataclass(frozen=True)
class TerrainCoverageReport:
    max_excursion: float
    sensor_radius: float
    policy_margin: float
    patch_safe_radius: float
    motion_key: str

    @property
    def required_radius(self) -> float:
        return self.max_excursion + self.sensor_radius + self.policy_margin


def validate_motion_terrain_coverage(
    data_paths: str | Path | list[str] | tuple[str, ...],
    *,
    patch_size: tuple[float, float],
    sensor_radius: float,
    policy_margin: float,
) -> TerrainCoverageReport:
    """Assert that every reference clip plus sensing margin fits one patch."""
    paths = [data_paths] if isinstance(data_paths, (str, Path)) else list(data_paths)
    if not paths:
        raise ValueError("at least one motion data path is required for terrain coverage validation")

    max_excursion = -1.0
    max_key = ""
    for path in paths:
        motions = joblib.load(Path(path).expanduser())
        if not isinstance(motions, dict):
            raise ValueError(f"motion data must be a dict for terrain coverage validation: {path}")
        for key, motion in motions.items():
            root = np.asarray(motion["root_trans_offset"], dtype=np.float64)
            if root.ndim != 2 or root.shape[1] < 2 or root.shape[0] == 0:
                raise ValueError(f"invalid root_trans_offset for motion {key!r}: {root.shape}")
            excursion = float(np.linalg.norm(root[:, :2] - root[0, :2], axis=1).max())
            if excursion > max_excursion:
                max_excursion = excursion
                max_key = str(key)

    report = TerrainCoverageReport(
        max_excursion=max_excursion,
        sensor_radius=float(sensor_radius),
        policy_margin=float(policy_margin),
        patch_safe_radius=min(float(patch_size[0]), float(patch_size[1])) / 2.0,
        motion_key=max_key,
    )
    if report.required_radius >= report.patch_safe_radius:
        raise RuntimeError(
            "terrain patch coverage invariant failed: "
            f"max_excursion={report.max_excursion:.3f}m motion={report.motion_key!r}, "
            f"sensor_radius={report.sensor_radius:.3f}m, policy_margin={report.policy_margin:.3f}m, "
            f"required={report.required_radius:.3f}m >= patch_safe_radius={report.patch_safe_radius:.3f}m"
        )
    return report
