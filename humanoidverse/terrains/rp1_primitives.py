"""RP1 terrain primitives that are not provided by MJLab."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.terrains import (
    BoxFlatTerrainCfg,
    BoxInvertedPyramidStairsTerrainCfg,
    FlatPatchSamplingCfg,
    HfPerlinNoiseTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    SubTerrainCfg,
)
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput


def _add_box(
    body: mujoco.MjsBody,
    *,
    center_xy: tuple[float, float],
    size_xy: tuple[float, float],
    bottom: float,
    top: float,
    color: tuple[float, float, float, float],
) -> TerrainGeometry:
    height = max(top - bottom, 1e-6)
    geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(max(size_xy[0] / 2, 1e-6), max(size_xy[1] / 2, 1e-6), height / 2),
        pos=(center_xy[0], center_xy[1], bottom + height / 2),
    )
    return TerrainGeometry(geom=geom, color=color)


def _remove_height_material(output: TerrainOutput) -> TerrainOutput:
    for geometry in output.geometries:
        if geometry.geom is not None:
            geometry.geom.material = ""
    return output


def _mark_as_terrain(output: TerrainOutput) -> TerrainOutput:
    """Put collision terrain in the geom group queried by height rays."""
    for geometry in output.geometries:
        if geometry.geom is not None:
            geometry.geom.group = 5
    return output


@dataclass(kw_only=True)
class TerrainBoxFlatCfg(BoxFlatTerrainCfg):
    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        return _mark_as_terrain(super().function(difficulty, spec, rng))


@dataclass(kw_only=True)
class TerrainInvertedPyramidStairsCfg(BoxInvertedPyramidStairsTerrainCfg):
    """Stairs with a safe center platform and ascent in every heading."""

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        return _mark_as_terrain(super().function(difficulty, spec, rng))


@dataclass(kw_only=True)
class TerrainBoundedStairsCfg(SubTerrainCfg):
    """Concentric stairs that ascend for a fixed count and then plateau."""

    step_height_range: tuple[float, float] = (0.08, 0.11)
    step_width: float = 0.30
    platform_width: float = 1.5
    num_steps: int = 4
    border_width: float = 0.5
    base_thickness: float = 0.25

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        del rng
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        available_width = min(self.size) - 2 * self.border_width
        ascent_width = self.platform_width + 2 * self.num_steps * self.step_width
        if ascent_width >= available_width:
            raise ValueError("terrain patch is too small for bounded stairs and plateau")

        step_height = self.step_height_range[0] + difficulty * (
            self.step_height_range[1] - self.step_height_range[0]
        )
        center_x = self.size[0] / 2
        center_y = self.size[1] / 2
        body = spec.body("terrain")
        geometries = [
            _add_box(
                body,
                center_xy=(center_x, center_y),
                size_xy=self.size,
                bottom=-self.base_thickness,
                top=0.0,
                color=(0.32, 0.32, 0.32, 1.0),
            )
        ]

        for step in range(1, self.num_steps + 1):
            inner_width = self.platform_width + 2 * (step - 1) * self.step_width
            outer_width = (
                available_width
                if step == self.num_steps
                else self.platform_width + 2 * step * self.step_width
            )
            band_width = (outer_width - inner_width) / 2
            offset = inner_width / 2 + band_width / 2
            top = step * step_height
            color = (0.30 + 0.04 * step, 0.34 + 0.03 * step, 0.38 + 0.02 * step, 1.0)
            for y in (center_y - offset, center_y + offset):
                geometries.append(
                    _add_box(
                        body,
                        center_xy=(center_x, y),
                        size_xy=(outer_width, band_width),
                        bottom=0.0,
                        top=top,
                        color=color,
                    )
                )
            for x in (center_x - offset, center_x + offset):
                geometries.append(
                    _add_box(
                        body,
                        center_xy=(x, center_y),
                        size_xy=(band_width, inner_width),
                        bottom=0.0,
                        top=top,
                        color=color,
                    )
                )

        return _mark_as_terrain(
            TerrainOutput(
                origin=np.array([center_x, center_y, 0.0]),
                geometries=geometries,
            )
        )


@dataclass(kw_only=True)
class TerrainTraversalCourseCfg(SubTerrainCfg):
    """A heading-agnostic radial route with a fixed obstacle sequence."""

    flat_run: float = 1.80
    step_height: float = 0.12
    step_depth: float = 0.30
    num_steps: int = 5
    top_platform_length: float = 0.80
    connector_length: float = 1.00
    ramp_length: float = 2.50
    ramp_angle_deg: float = 8.0
    border_width: float = 0.50
    base_thickness: float = 0.25
    horizontal_scale: float = 0.10

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        del difficulty, rng
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if min(self.step_height, self.step_depth, self.ramp_length, self.horizontal_scale) <= 0.0:
            raise ValueError("course step, ramp, and heightfield dimensions must be positive")

        stairs_start = self.flat_run
        stairs_up_end = stairs_start + self.num_steps * self.step_depth
        stairs_down_start = stairs_up_end + self.top_platform_length
        stairs_down_end = stairs_down_start + self.num_steps * self.step_depth
        ramp_start = stairs_down_end + self.connector_length
        ramp_end = ramp_start + self.ramp_length
        safe_radius = min(self.size) / 2.0 - self.border_width
        if ramp_end >= safe_radius:
            raise ValueError("course does not fit inside the terrain patch")

        num_cols = int(round(self.size[0] / self.horizontal_scale)) + 1
        num_rows = int(round(self.size[1] / self.horizontal_scale)) + 1
        x_coords = np.linspace(0.0, self.size[0], num_cols)
        y_coords = np.linspace(0.0, self.size[1], num_rows)
        xx, yy = np.meshgrid(x_coords, y_coords)
        center_x = self.size[0] / 2.0
        center_y = self.size[1] / 2.0
        radius = np.hypot(xx - center_x, yy - center_y)
        heights = np.zeros_like(radius)

        up = (radius >= stairs_start) & (radius < stairs_up_end)
        up_levels = np.floor((radius[up] - stairs_start) / self.step_depth).astype(np.int32) + 1
        heights[up] = up_levels * self.step_height

        stair_peak = self.num_steps * self.step_height
        top = (radius >= stairs_up_end) & (radius < stairs_down_start)
        heights[top] = stair_peak

        down = (radius >= stairs_down_start) & (radius < stairs_down_end)
        down_levels = np.floor((radius[down] - stairs_down_start) / self.step_depth).astype(np.int32) + 1
        heights[down] = np.maximum(stair_peak - down_levels * self.step_height, 0.0)

        ramp = (radius >= ramp_start) & (radius < ramp_end)
        ramp_slope = math.tan(math.radians(self.ramp_angle_deg))
        heights[ramp] = (radius[ramp] - ramp_start) * ramp_slope
        ramp_rise = self.ramp_length * ramp_slope
        heights[radius >= ramp_end] = ramp_rise

        max_height = float(np.max(heights))
        normalized = heights / max_height
        field = spec.add_hfield(
            name=f"hfield_course_{uuid.uuid4().hex}",
            size=(self.size[0] / 2.0, self.size[1] / 2.0, max_height, self.base_thickness),
            nrow=num_rows,
            ncol=num_cols,
            userdata=normalized.astype(np.float32).ravel().tolist(),
        )
        geom = spec.body("terrain").add_geom(
            type=mujoco.mjtGeom.mjGEOM_HFIELD,
            hfieldname=field.name,
            pos=(center_x, center_y, 0.0),
        )
        return _mark_as_terrain(
            TerrainOutput(
                origin=np.array([center_x, center_y, 0.0]),
                geometries=[TerrainGeometry(geom=geom, hfield=field, color=(0.38, 0.44, 0.40, 1.0))],
            )
        )


@dataclass(kw_only=True)
class NeutralHfPerlinNoiseTerrainCfg(HfPerlinNoiseTerrainCfg):
    """Perlin terrain without its height-colored texture material."""

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        output = _mark_as_terrain(_remove_height_material(super().function(difficulty, spec, rng)))
        if output.flat_patches is not None and len(output.flat_patches.get("spawn", ())) > 0:
            output.origin = output.flat_patches["spawn"][0].copy()
        return output


@dataclass(kw_only=True)
class NeutralHfPyramidSlopedTerrainCfg(HfPyramidSlopedTerrainCfg):
    """Pyramid slope terrain without its height-colored texture material."""

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        return _mark_as_terrain(_remove_height_material(super().function(difficulty, spec, rng)))


def spawn_patch_sampling(
    *, patch_radius: float, patch_center_range: float, patch_size: tuple[float, float]
) -> dict[str, FlatPatchSamplingCfg]:
    """Constrain rough-terrain reset patches to the central safe region."""
    return {
        "spawn": FlatPatchSamplingCfg(
            num_patches=8,
            patch_radius=patch_radius,
            max_height_diff=0.025,
            x_range=(patch_size[0] / 2 - patch_center_range, patch_size[0] / 2 + patch_center_range),
            y_range=(patch_size[1] / 2 - patch_center_range, patch_size[1] / 2 + patch_center_range),
        )
    }


@dataclass(kw_only=True)
class BoxPlatformsTerrainCfg(SubTerrainCfg):
    """Alternating raised and flat concentric square platform bands."""

    platform_height_range: tuple[float, float] = (0.05, 0.15)
    band_width: float = 0.6
    center_width: float = 2.0
    border_width: float = 0.5
    base_thickness: float = 0.2

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        del rng
        available_width = min(self.size) - 2 * self.border_width
        if self.center_width >= available_width:
            raise ValueError("Terrain patch is too small for platform bands.")

        platform_height = self.platform_height_range[0] + difficulty * (self.platform_height_range[1] - self.platform_height_range[0])
        center_x = self.size[0] / 2
        center_y = self.size[1] / 2
        body = spec.body("terrain")
        green = (0.25, 0.80, 0.45, 1.0)
        geometries = [
            _add_box(
                body,
                center_xy=(center_x, center_y),
                size_xy=self.size,
                bottom=-self.base_thickness,
                top=0.0,
                color=(0.18, 0.38, 0.24, 1.0),
            )
        ]

        band_index = 0
        inner_width = self.center_width
        while inner_width < available_width:
            outer_width = min(inner_width + 2 * self.band_width, available_width)
            actual_band_width = (outer_width - inner_width) / 2
            if band_index % 2 == 0 and actual_band_width > 1e-6:
                offset = inner_width / 2 + actual_band_width / 2
                for y in (center_y - offset, center_y + offset):
                    geometries.append(
                        _add_box(
                            body,
                            center_xy=(center_x, y),
                            size_xy=(outer_width, actual_band_width),
                            bottom=0.0,
                            top=platform_height,
                            color=green,
                        )
                    )
                for x in (center_x - offset, center_x + offset):
                    geometries.append(
                        _add_box(
                            body,
                            center_xy=(x, center_y),
                            size_xy=(actual_band_width, inner_width),
                            bottom=0.0,
                            top=platform_height,
                            color=green,
                        )
                    )
            inner_width = outer_width
            band_index += 1

        return TerrainOutput(
            origin=np.array([center_x, center_y, 0.0]),
            geometries=geometries,
        )
