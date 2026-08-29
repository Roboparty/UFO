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
    BoxPyramidStairsTerrainCfg,
    BoxRandomGridTerrainCfg,
    FlatPatchSamplingCfg,
    HfDiscreteObstaclesTerrainCfg,
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


def _mark_as_terrain(output: TerrainOutput, *, group: int = 5) -> TerrainOutput:
    """Put collision terrain in the requested ray-query geom group."""
    for geometry in output.geometries:
        if geometry.geom is not None:
            geometry.geom.group = group
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
class TerrainPyramidStairsCfg(BoxPyramidStairsTerrainCfg):
    """RP1 ascending stairs marked for the dedicated terrain ray group."""

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        return _mark_as_terrain(super().function(difficulty, spec, rng))


@dataclass(kw_only=True)
class TerrainRandomGridCfg(BoxRandomGridTerrainCfg):
    """RP1 low platforms marked for the dedicated terrain ray group."""

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
    """A square-symmetric center-out route: stairs up, platform, stairs down."""

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
        safe_forward_extent = min(self.size) / 2.0 - self.border_width
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

        inner_width = 2.0 * self.flat_run
        for level in range(1, self.num_steps + 1):
            outer_width = inner_width + 2.0 * self.step_depth
            geometries.extend(
                _add_square_ring(
                    terrain_body,
                    center_xy=(center_x, center_y),
                    inner_width=inner_width,
                    outer_width=outer_width,
                    bottom=0.0,
                    top=level * self.step_height,
                    color=color,
                )
            )
            inner_width = outer_width

        stair_peak = self.num_steps * self.step_height
        outer_width = inner_width + 2.0 * self.top_platform_length
        geometries.extend(
            _add_square_ring(
                terrain_body,
                center_xy=(center_x, center_y),
                inner_width=inner_width,
                outer_width=outer_width,
                bottom=0.0,
                top=stair_peak,
                color=color,
            )
        )
        inner_width = outer_width

        for level in range(1, self.num_steps + 1):
            outer_width = inner_width + 2.0 * self.step_depth
            height = (self.num_steps - level) * self.step_height
            if height > 0.0:
                geometries.extend(
                    _add_square_ring(
                        terrain_body,
                        center_xy=(center_x, center_y),
                        inner_width=inner_width,
                        outer_width=outer_width,
                        bottom=0.0,
                        top=height,
                        color=color,
                    )
                )
            inner_width = outer_width

        return _mark_as_terrain(
            TerrainOutput(
                origin=np.array([center_x, center_y, 0.0]),
                geometries=geometries,
            )
        )


@dataclass(kw_only=True)
class NeutralHfPerlinNoiseTerrainCfg(HfPerlinNoiseTerrainCfg):
    """Perlin terrain without its height-colored texture material."""

    geom_group: int = 5

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        output = _mark_as_terrain(
            _remove_height_material(super().function(difficulty, spec, rng)),
            group=self.geom_group,
        )
        if output.flat_patches is not None and len(output.flat_patches.get("spawn", ())) > 0:
            output.origin = output.flat_patches["spawn"][0].copy()
        return output


@dataclass(kw_only=True)
class NeutralHfPyramidSlopedTerrainCfg(HfPyramidSlopedTerrainCfg):
    """Pyramid slope terrain without its height-colored texture material."""

    geom_group: int = 5

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        return _mark_as_terrain(
            _remove_height_material(super().function(difficulty, spec, rng)),
            group=self.geom_group,
        )


@dataclass(kw_only=True)
class NeutralHfDiscreteObstaclesTerrainCfg(HfDiscreteObstaclesTerrainCfg):
    """RP1 discrete obstacles without height material, visible to terrain rays."""

    geom_group: int = 5

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        return _mark_as_terrain(
            _remove_height_material(super().function(difficulty, spec, rng)),
            group=self.geom_group,
        )


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
class ScatteredBoxPlatformsTerrainCfg(SubTerrainCfg):
    """Irregular non-overlapping rectangular platforms on a connected base."""

    easy_height_range: tuple[float, float] = (0.05, 0.10)
    hard_height_range: tuple[float, float] = (0.10, 0.18)
    center_width: float = 1.2
    border_width: float = 0.5
    gap_width: float = 0.25
    min_platform_width: float = 0.60
    raised_fraction: float = 1.0
    row_count_range: tuple[int, int] = (5, 7)
    column_count_range: tuple[int, int] = (4, 7)
    spawn_patch_margin: float = 0.25
    num_spawn_patches: int = 12
    base_thickness: float = 0.2

    @staticmethod
    def _partition(length: float, count: int, rng: np.random.Generator) -> np.ndarray:
        weights = rng.uniform(0.85, 1.15, size=count)
        return length * weights / weights.sum()

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        if min(self.size) <= 2 * self.border_width:
            raise ValueError("Terrain patch is too small for the requested flat border.")
        if self.center_width <= 0.0 or self.center_width >= min(self.size) - 2 * self.border_width:
            raise ValueError("center_width must fit inside the platform field.")
        if self.gap_width < 0.0 or self.min_platform_width <= 0.0 or not 0.0 <= self.raised_fraction <= 1.0:
            raise ValueError("gap_width and raised_fraction are invalid.")
        if self.num_spawn_patches <= 0 or self.spawn_patch_margin < 0.0:
            raise ValueError("Platform spawn-patch parameters must be positive.")

        easy_low, easy_high = self.easy_height_range
        hard_low, hard_high = self.hard_height_range
        if not 0.0 <= easy_low <= easy_high or not 0.0 <= hard_low <= hard_high:
            raise ValueError("Platform height ranges must be ordered and non-negative.")
        difficulty = float(np.clip(difficulty, 0.0, 1.0))
        height_low = easy_low + difficulty * (hard_low - easy_low)
        height_high = easy_high + difficulty * (hard_high - easy_high)

        center_x = self.size[0] / 2
        center_y = self.size[1] / 2
        interior_x = self.size[0] - 2 * self.border_width
        interior_y = self.size[1] - 2 * self.border_width
        body = spec.body("terrain")
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

        min_rows, max_rows = self.row_count_range
        min_cols, max_cols = self.column_count_range
        if min_rows <= 0 or min_cols <= 0 or min_rows > max_rows or min_cols > max_cols:
            raise ValueError("Platform row/column count ranges are invalid.")
        row_count = int(rng.integers(min_rows, max_rows + 1))
        rows_below = (row_count - 1) // 2
        rows_above = row_count - 1 - rows_below
        side_y = (interior_y - self.center_width) / 2
        row_heights = np.concatenate(
            (
                self._partition(side_y, rows_below, rng),
                np.array([self.center_width]),
                self._partition(side_y, rows_above, rng),
            )
        )
        spawn_candidates: list[np.ndarray] = []
        y_cursor = self.border_width

        for row_index, row_height in enumerate(row_heights):
            column_count = int(rng.integers(min_cols, max_cols + 1))
            columns_left = (column_count - 1) // 2
            columns_right = column_count - 1 - columns_left
            side_x = (interior_x - self.center_width) / 2
            column_widths = np.concatenate(
                (
                    self._partition(side_x, columns_left, rng),
                    np.array([self.center_width]),
                    self._partition(side_x, columns_right, rng),
                )
            )
            x_cursor = self.border_width
            for column_index, column_width in enumerate(column_widths):
                rect_x0 = x_cursor + self.gap_width / 2
                rect_x1 = x_cursor + column_width - self.gap_width / 2
                rect_y0 = y_cursor + self.gap_width / 2
                rect_y1 = y_cursor + row_height - self.gap_width / 2
                x_cursor += column_width
                is_center = row_index == rows_below and column_index == columns_left
                if rect_x1 <= rect_x0 or rect_y1 <= rect_y0:
                    continue
                if is_center or rng.random() >= self.raised_fraction:
                    continue

                height = float(rng.uniform(height_low, height_high))
                height_alpha = 0.0 if height_high <= height_low else (height - height_low) / (height_high - height_low)
                color = (0.22 + 0.10 * height_alpha, 0.56 + 0.18 * height_alpha, 0.34, 1.0)
                rectangles = [(rect_x0, rect_x1, rect_y0, rect_y1)]
                for part_x0, part_x1, part_y0, part_y1 in rectangles:
                    rect_size = (part_x1 - part_x0, part_y1 - part_y0)
                    if min(rect_size) < self.min_platform_width:
                        continue
                    rect_center = ((part_x0 + part_x1) / 2, (part_y0 + part_y1) / 2)
                    geometries.append(
                        _add_box(
                            body,
                            center_xy=rect_center,
                            size_xy=rect_size,
                            bottom=0.0,
                            top=height,
                            color=color,
                        )
                    )
                    if min(rect_size) >= 2 * self.spawn_patch_margin:
                        spawn_candidates.append(np.array([rect_center[0], rect_center[1], height], dtype=np.float64))
            y_cursor += row_height

        if not spawn_candidates:
            raise RuntimeError("Scattered platform generation produced no safe raised reset patch.")
        selected = [spawn_candidates[index % len(spawn_candidates)] for index in range(self.num_spawn_patches)]
        return _mark_as_terrain(
            TerrainOutput(
                origin=np.array([center_x, center_y, 0.0]),
                geometries=geometries,
                flat_patches={"platform_spawn": np.stack(selected)},
            )
        )
