"""PBFM terrain-height sensor integration helpers."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import torch
import warp as wp
from mjlab.entity import Entity
from mjlab.sensor.raycast_sensor import RayCastSensor, RayCastSensorCfg
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor, TerrainHeightSensorCfg


HEIGHT_SCAN_MIN = -5.0
HEIGHT_SCAN_MAX = 5.0


def clipped_height_clearance(
    sample_height: torch.Tensor,
    hit_height: torch.Tensor,
    distance: torch.Tensor,
    *,
    offset: float = 0.0,
) -> torch.Tensor:
    """Return finite sample-to-terrain clearance, filling invalid rays with the clip maximum."""

    clearance = sample_height - hit_height - float(offset)
    valid = (distance >= 0.0) & torch.isfinite(distance) & torch.isfinite(clearance)
    clearance = torch.where(valid, clearance, HEIGHT_SCAN_MAX)
    return clearance.clamp(HEIGHT_SCAN_MIN, HEIGHT_SCAN_MAX)


@dataclass
class WorldUpTerrainHeightSensorCfg(RayCastSensorCfg):
    """Body-attached sample points with world-down terrain rays."""

    ray_start_height: float = 20.0
    local_center: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def build(self) -> "WorldUpTerrainHeightSensor":
        return WorldUpTerrainHeightSensor(self)


class WorldUpTerrainHeightSensor(RayCastSensor):
    cfg: WorldUpTerrainHeightSensorCfg

    def initialize(self, mj_model, model, data, device: str) -> None:
        super().initialize(mj_model, model, data, device)
        if self._local_offsets is None:
            raise RuntimeError("world-up terrain ray pattern was not initialized")
        self._local_offsets = self._local_offsets.clone()
        self._local_offsets += self._local_offsets.new_tensor(self.cfg.local_center)
        self._sample_points_w: torch.Tensor | None = None

    @property
    def sample_points_w(self) -> torch.Tensor:
        """World-space locations whose vertical terrain clearance is measured."""

        if self._sample_points_w is None:
            raise RuntimeError("terrain sample points are not available before sensing")
        return self._sample_points_w

    def prepare_rays(self) -> None:
        super().prepare_rays()
        if (
            self._cached_world_origins is None
            or self._cached_world_rays is None
            or self._ray_pnt is None
            or self._ray_vec is None
        ):
            raise RuntimeError("world-up terrain ray origins were not prepared")

        self._sample_points_w = self._cached_world_origins.clone()
        self._cached_world_origins[..., 2] += self.cfg.ray_start_height
        self._cached_world_rays.zero_()
        self._cached_world_rays[..., 2] = -1.0
        wp.to_torch(self._ray_pnt).view_as(self._cached_world_origins).copy_(self._cached_world_origins)
        wp.to_torch(self._ray_vec).view_as(self._cached_world_rays).copy_(self._cached_world_rays)


def terrain_padding_regions(
    scene_spec: mujoco.MjSpec,
    *,
    expected_count: int = 4,
) -> tuple[tuple[float, float, float, float, float], ...]:
    """Return the XY rectangles and top heights of generator border boxes."""
    terrain_body = scene_spec.body("terrain")
    if terrain_body is None:
        raise RuntimeError("terrain body is missing while configuring height rays")
    padding_geoms = [geom for geom in terrain_body.geoms if int(geom.group) == 0]
    if len(padding_geoms) != expected_count:
        raise RuntimeError(
            "expected exactly four group-0 global terrain padding geoms, "
            f"found {len(padding_geoms)}"
        )
    return tuple(
        (
            float(geom.pos[0] - geom.size[0]),
            float(geom.pos[0] + geom.size[0]),
            float(geom.pos[1] - geom.size[1]),
            float(geom.pos[1] + geom.size[1]),
            float(geom.pos[2] + geom.size[2]),
        )
        for geom in padding_geoms
    )


def mark_terrain_padding_ray_group(
    scene_spec: mujoco.MjSpec,
    *,
    ray_group: int = 5,
    expected_count: int = 4,
) -> int:
    """Move only generator-owned global border boxes into the terrain ray group."""
    terrain_body = scene_spec.body("terrain")
    if terrain_body is None:
        raise RuntimeError("terrain body is missing while configuring height rays")
    padding_geoms = [geom for geom in terrain_body.geoms if int(geom.group) == 0]
    if len(padding_geoms) != expected_count:
        raise RuntimeError(
            "expected exactly four group-0 global terrain padding geoms, "
            f"found {len(padding_geoms)}"
        )
    for geom in padding_geoms:
        geom.group = ray_group
    return len(padding_geoms)


def repair_padding_ray_misses(
    distances: torch.Tensor,
    hit_positions: torch.Tensor,
    normals: torch.Tensor,
    origins: torch.Tensor,
    padding_regions: tuple[tuple[float, float, float, float, float], ...],
    *,
    max_distance: float,
) -> None:
    """Repair only ray misses whose XY lies on known flat generator padding."""
    misses = distances < 0.0
    for x_min, x_max, y_min, y_max, ground_z in padding_regions:
        in_region = (
            (origins[..., 0] >= x_min)
            & (origins[..., 0] <= x_max)
            & (origins[..., 1] >= y_min)
            & (origins[..., 1] <= y_max)
        )
        clearance = (origins[..., 2] - ground_z).clamp(min=0.0)
        repair = misses & in_region & (clearance <= max_distance)
        distances[repair] = clearance[repair]
        hit_positions[repair] = origins[repair]
        hit_positions[..., 2][repair] = origins[..., 2][repair] - clearance[repair]
        normals[..., 0][repair] = 0.0
        normals[..., 1][repair] = 0.0
        normals[..., 2][repair] = 1.0


@dataclass
class PbfmTerrainHeightSensorCfg(TerrainHeightSensorCfg):
    """Terrain sensor that makes MJLab generator padding visible to its rays."""

    padding_ray_group: int = 5

    def build(self) -> PbfmTerrainHeightSensor:
        return PbfmTerrainHeightSensor(self)


class PbfmTerrainHeightSensor(TerrainHeightSensor):
    cfg: PbfmTerrainHeightSensorCfg

    def __init__(self, cfg: PbfmTerrainHeightSensorCfg) -> None:
        super().__init__(cfg)
        self._padding_regions: tuple[tuple[float, float, float, float, float], ...] = ()

    def edit_spec(
        self,
        scene_spec: mujoco.MjSpec,
        entities: dict[str, Entity],
    ) -> None:
        terrain = entities.get("terrain")
        generator_cfg = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
        # MJLab creates generator padding directly in geom group 0.  The
        # legacy G1 terrains move all physical terrain into a dedicated group
        # and therefore need the padding moved and miss-repaired as well.
        # RP1 terrain intentionally stays in source group 0, so its padding is
        # already visible and must not be rediscovered by scanning every
        # group-0 terrain geom.
        needs_padding_group_remap = int(self.cfg.padding_ray_group) != 0
        if (
            generator_cfg is not None
            and float(generator_cfg.border_width) > 0.0
            and needs_padding_group_remap
        ):
            self._padding_regions = terrain_padding_regions(scene_spec)
            mark_terrain_padding_ray_group(
                scene_spec,
                ray_group=self.cfg.padding_ray_group,
            )
        super().edit_spec(scene_spec, entities)

    def postprocess_rays(self) -> None:
        super().postprocess_rays()
        if not self._padding_regions:
            return
        assert self._cached_world_origins is not None
        assert self._distances is not None
        assert self._hit_pos_w is not None
        assert self._normals_w is not None

        repair_padding_ray_misses(
            self._distances,
            self._hit_pos_w,
            self._normals_w,
            self._cached_world_origins,
            self._padding_regions,
            max_distance=self.cfg.max_distance,
        )
