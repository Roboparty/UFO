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


def _add_square_ring(
    body: mujoco.MjsBody,
    *,
    center_xy: tuple[float, float],
    inner_width: float,
    outer_width: float,
    bottom: float,
    top: float,
    color: tuple[float, float, float, float],
) -> list[TerrainGeometry]:
    """Build a square annulus from four non-overlapping boxes."""
    band_width = (outer_width - inner_width) / 2
    if band_width <= 1e-6:
        return []
    center_x, center_y = center_xy
    offset = inner_width / 2 + band_width / 2
    geometries = []
    for y in (center_y - offset, center_y + offset):
        geometries.append(
            _add_box(
                body,
                center_xy=(center_x, y),
                size_xy=(outer_width, band_width),
                bottom=bottom,
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
                bottom=bottom,
                top=top,
                color=color,
            )
        )
    return geometries


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
    """Repeat bounded up-platform-down stair cycles across the patch."""

    step_height_range: tuple[float, float] = (0.10, 0.18)
    step_width: float = 0.30
    platform_width: float = 1.5
    num_steps: int = 6
    plateau_width: float = 0.8
    border_width: float = 0.5
    base_thickness: float = 0.25

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        del rng
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if min(self.step_width, self.platform_width, self.plateau_width) <= 0.0:
            raise ValueError("stair widths must be positive")
        available_width = min(self.size) - 2 * self.border_width
        cycle_radius = 2 * self.num_steps * self.step_width + 2 * self.plateau_width
        available_radius = (available_width - self.platform_width) / 2
        num_cycles = int(available_radius // cycle_radius)
        if num_cycles <= 0:
            raise ValueError("terrain patch is too small for one complete up-platform-down cycle")

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

        center_xy = (center_x, center_y)
        peak_height = self.num_steps * step_height
        current_width = self.platform_width
        for cycle in range(num_cycles):
            for step in range(1, self.num_steps + 1):
                outer_width = current_width + 2 * self.step_width
                top = step * step_height
                color = (0.30 + 0.04 * step, 0.34 + 0.03 * step, 0.38 + 0.02 * step, 1.0)
                geometries.extend(
                    _add_square_ring(
                        body,
                        center_xy=center_xy,
                        inner_width=current_width,
                        outer_width=outer_width,
                        bottom=0.0,
                        top=top,
                        color=color,
                    )
                )
                current_width = outer_width

            plateau_outer_width = current_width + 2 * self.plateau_width
            geometries.extend(
                _add_square_ring(
                    body,
                    center_xy=center_xy,
                    inner_width=current_width,
                    outer_width=plateau_outer_width,
                    bottom=0.0,
                    top=peak_height,
                    color=(0.52, 0.52, 0.50, 1.0),
                )
            )
            current_width = plateau_outer_width

            for step in range(1, self.num_steps + 1):
                outer_width = current_width + 2 * self.step_width
                top = (self.num_steps - step) * step_height
                if top > 0.0:
                    geometries.extend(
                        _add_square_ring(
                            body,
                            center_xy=center_xy,
                            inner_width=current_width,
                            outer_width=outer_width,
                            bottom=0.0,
                            top=top,
                            color=(0.42, 0.43, 0.43, 1.0),
                        )
                    )
                current_width = outer_width

            # The base box supplies the low platform before the next cycle.
            current_width += 2 * self.plateau_width

        return _mark_as_terrain(
            TerrainOutput(
                origin=np.array([center_x, center_y, 0.0]),
                geometries=geometries,
            )
        )


@dataclass(kw_only=True)
class TerrainSeparatedPyramidStairsCfg(SubTerrainCfg):
    """A fixed-step square staircase with a center-only transition start."""

    direction: str
    step_height_range: tuple[float, float] = (0.10, 0.18)
    step_width: float = 0.30
    platform_width: float = 0.80
    num_steps: int = 10
    border_width: float = 0.50
    base_thickness: float = 0.25

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        del rng
        if self.direction not in {"up", "down"}:
            raise ValueError("stair direction must be 'up' or 'down'")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if min(self.step_width, self.platform_width, self.base_thickness) <= 0.0:
            raise ValueError("stair dimensions must be positive")

        stair_outer_width = self.platform_width + 2 * self.num_steps * self.step_width
        available_width = min(self.size) - 2 * self.border_width
        if stair_outer_width >= available_width:
            raise ValueError("separated stairs need a flat outer seam inside the terrain patch")

        step_height = self.step_height_range[0] + difficulty * (
            self.step_height_range[1] - self.step_height_range[0]
        )
        peak_height = self.num_steps * step_height
        center_xy = (self.size[0] / 2.0, self.size[1] / 2.0)
        body = spec.body("terrain")
        base_top = -peak_height if self.direction == "up" else 0.0
        base_bottom = base_top - self.base_thickness
        base_size = (
            (self.platform_width, self.platform_width)
            if self.direction == "up"
            else self.size
        )
        geometries = [
            _add_box(
                body,
                center_xy=center_xy,
                size_xy=base_size,
                bottom=base_bottom,
                top=base_top,
                color=(0.30, 0.32, 0.34, 1.0),
            )
        ]

        if self.direction == "down":
            geometries.append(
                _add_box(
                    body,
                    center_xy=center_xy,
                    size_xy=(self.platform_width, self.platform_width),
                    bottom=0.0,
                    top=peak_height,
                    color=(0.54, 0.54, 0.50, 1.0),
                )
            )

        current_width = self.platform_width
        for step in range(1, self.num_steps + 1):
            outer_width = current_width + 2 * self.step_width
            if self.direction == "up":
                top = -(self.num_steps - step) * step_height
                bottom = base_top
            else:
                top = (self.num_steps - step) * step_height
                bottom = 0.0
            if top > bottom + 1.0e-6:
                geometries.extend(
                    _add_square_ring(
                        body,
                        center_xy=center_xy,
                        inner_width=current_width,
                        outer_width=outer_width,
                        bottom=bottom,
                        top=top,
                        color=(0.34 + 0.025 * step, 0.37 + 0.02 * step, 0.40, 1.0),
                    )
                )
            current_width = outer_width

        if self.direction == "up":
            # Fill the remainder at z=0 so every tile boundary shares the
            # connected map's common collision and ray-cast height.
            geometries.extend(
                _add_square_ring(
                    body,
                    center_xy=center_xy,
                    inner_width=current_width,
                    # A tiny overlap removes exact-edge ray gaps between
                    # independently compiled neighboring tiles.
                    outer_width=min(self.size) + 2.0e-3,
                    bottom=base_top,
                    top=0.0,
                    color=(0.42, 0.43, 0.43, 1.0),
                )
            )

        center_height = -peak_height if self.direction == "up" else peak_height
        return _mark_as_terrain(
            TerrainOutput(
                origin=np.array([center_xy[0], center_xy[1], center_height]),
                geometries=geometries,
            )
        )


@dataclass(kw_only=True)
class TerrainTraversalCourseCfg(SubTerrainCfg):
    """A straight +x route with stair edges spanning the full tile width."""

    flat_run: float = 1.80
    step_height: float = 0.12
    step_depth: float = 0.30
    num_steps: int = 5
    top_platform_length: float = 0.80
    connector_length: float = 1.00
    include_ramp: bool = True
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
            raise ValueError("course step and terrain dimensions must be positive")
        if self.include_ramp:
            raise ValueError("box-based traversal course does not support include_ramp=True")

        stairs_start = self.flat_run
        stairs_up_end = stairs_start + self.num_steps * self.step_depth
        stairs_down_start = stairs_up_end + self.top_platform_length
        stairs_down_end = stairs_down_start + self.num_steps * self.step_depth
        ramp_start = stairs_down_end + self.connector_length
        ramp_end = ramp_start + self.ramp_length
        safe_forward_extent = self.size[0] / 2.0 - self.border_width
        course_end = ramp_end if self.include_ramp else stairs_down_end
        if course_end >= safe_forward_extent:
            raise ValueError("course does not fit inside the terrain patch")

        center_x = self.size[0] / 2.0
        center_y = self.size[1] / 2.0
        terrain_body = spec.body("terrain")
        color = (0.38, 0.44, 0.40, 1.0)
        geometries = [
            _add_box(
                terrain_body,
                center_xy=(center_x, center_y),
                size_xy=self.size,
                bottom=-self.base_thickness,
                top=0.0,
                color=color,
            )
        ]

        def add_band(start: float, end: float, height: float) -> None:
            if height <= 0.0 or end <= start:
                return
            geometries.append(
                _add_box(
                    terrain_body,
                    center_xy=(center_x + (start + end) / 2.0, center_y),
                    size_xy=(end - start, self.size[1]),
                    bottom=0.0,
                    top=height,
                    color=color,
                )
            )

        for level in range(1, self.num_steps + 1):
            start = stairs_start + (level - 1) * self.step_depth
            add_band(start, start + self.step_depth, level * self.step_height)
        stair_peak = self.num_steps * self.step_height
        add_band(stairs_up_end, stairs_down_start, stair_peak)
        for level in range(1, self.num_steps + 1):
            start = stairs_down_start + (level - 1) * self.step_depth
            add_band(start, start + self.step_depth, (self.num_steps - level) * self.step_height)

        return _mark_as_terrain(
            TerrainOutput(
                origin=np.array([center_x, center_y, 0.0]),
                geometries=geometries,
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
        if min(self.band_width, self.center_width) <= 0.0:
            raise ValueError("Platform band and center widths must be positive.")
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
                geometries.extend(
                    _add_square_ring(
                        body,
                        center_xy=(center_x, center_y),
                        inner_width=inner_width,
                        outer_width=outer_width,
                        bottom=0.0,
                        top=platform_height,
                        color=green,
                    )
                )
            inner_width = outer_width
            band_index += 1

        return _mark_as_terrain(
            TerrainOutput(
                origin=np.array([center_x, center_y, 0.0]),
                geometries=geometries,
            )
        )
