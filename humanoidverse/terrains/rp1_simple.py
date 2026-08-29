"""Physical terrain presets for the opt-in UFO terrain feasibility experiment."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal

import mujoco
import numpy as np
from mjlab.terrains import (
    BoxFlatTerrainCfg,
    BoxInvertedPyramidStairsTerrainCfg,
    BoxPyramidStairsTerrainCfg,
    BoxRandomGridTerrainCfg,
    FlatPatchSamplingCfg,
    TerrainEntityCfg,
    TerrainGeneratorCfg,
)

from humanoidverse.terrains.rp1_primitives import (
    NeutralHfDiscreteObstaclesTerrainCfg,
    NeutralHfPerlinNoiseTerrainCfg,
    NeutralHfPyramidSlopedTerrainCfg,
    ScatteredBoxPlatformsTerrainCfg,
    TerrainBoundedStairsCfg,
    TerrainBoxFlatCfg,
    TerrainInvertedPyramidStairsCfg,
    TerrainSeparatedPyramidStairsCfg,
    TerrainTraversalCourseCfg,
    spawn_patch_sampling,
)

TerrainMode = Literal[
    "plane", "flat", "slope", "stairs", "stairs_up", "stairs_down", "rough", "platforms",
    "mixed", "rp1_simple", "course",
]
SUPPORTED_TERRAINS: tuple[TerrainMode, ...] = (
    "plane",
    "flat",
    "slope",
    "stairs",
    "stairs_up",
    "stairs_down",
    "rough",
    "platforms",
    "mixed",
    "rp1_simple",
    "course",
)
TERRAIN_COMPONENT_NAMES = ("flat", "slope", "stairs_up", "stairs_down", "rough", "platforms")
RP1_TERRAIN_COMPONENT_NAMES = (
    "flat",
    "perlin_rough",
    "low_stairs_up",
    "low_stairs_down",
    "low_platforms",
    "hf_pyramid_slope_inv",
    "boxes",
)
RP1_TERRAIN_PROPORTIONS = {
    "flat": 0.10,
    "perlin_rough": 0.10,
    "low_stairs_up": 0.25,
    "low_stairs_down": 0.25,
    "low_platforms": 0.10,
    "hf_pyramid_slope_inv": 0.10,
    "boxes": 0.10,
}
RP1_TERRAIN_REFERENCE_PROJECT = "UFO-rp1"
RP1_TERRAIN_REFERENCE_COMMIT = "8c364e1001734097aac58e5033a1b5076925d3c5"
RP1_PATCH_SIZE = 5.0
RP1_CENTER_PLATFORM_WIDTH = 1.2
RP1_NONFLAT_GUARD_WIDTH = 10.0
RP1_FLAT_SAFETY_WIDTH = 2.0
RP1_TERRAIN_BORDER_WIDTH = RP1_NONFLAT_GUARD_WIDTH + RP1_FLAT_SAFETY_WIDTH
RP1_GUARD_TILE_RINGS = int(RP1_NONFLAT_GUARD_WIDTH / RP1_PATCH_SIZE)
RP1_GUARD_HEIGHT_LIFT = 0.05
RP1_GUARD_FAMILIES = ("perlin_rough", "boxes")
RP1_STAIR_LEVELS = 5
RP1_STAIR_BORDER_WIDTH = 0.5
RP1_STAIR_STEP_WIDTH = 0.3
RP1_STAIR_PLATFORM_WIDTH = RP1_CENTER_PLATFORM_WIDTH
RP1_OUTER_WALL_HEIGHT = 5.0
RP1_OUTER_WALL_THICKNESS = 0.05
RP1_OUTER_WALL_COLOR = (0.5, 0.5, 0.5, 1.0)


def _get(config: Any, path: str, default: Any) -> Any:
    value = config
    for part in path.split("."):
        if value is None:
            return default
        value = value.get(part, default) if isinstance(value, Mapping) else getattr(value, part, default)
        if value is default:
            return default
    return value


def terrain_component_names(mode: TerrainMode) -> tuple[str, ...]:
    if mode == "rp1_simple":
        return RP1_TERRAIN_COMPONENT_NAMES
    if mode == "mixed":
        return TERRAIN_COMPONENT_NAMES
    if mode == "plane":
        return ("flat",)
    if mode == "course":
        return ("course",)
    return (mode,)


def _terrain_mix(config: Any) -> dict[str, float]:
    raw = _get(config, "terrain_mix", None)
    default_weight = 1.0 / len(TERRAIN_COMPONENT_NAMES)
    weights = {name: float(_get(raw, name, default_weight)) for name in TERRAIN_COMPONENT_NAMES}
    if any(value < 0.0 for value in weights.values()) or sum(weights.values()) <= 0.0:
        raise ValueError(f"terrain_mix must be non-negative and have positive total weight: {weights}")
    total = sum(weights.values())
    return {name: value / total for name, value in weights.items()}


def make_ufo_v0_generator_cfg(mode: TerrainMode, config: Any) -> TerrainGeneratorCfg:
    """Create collision terrain and spawn origins from one MJLab generator."""
    if mode == "rp1_simple":
        return make_rp1_simple_generator_cfg()
    selected = terrain_component_names(mode)
    weights = _terrain_mix(config)
    if mode != "mixed":
        weights = {name: float(name == mode) for name in TERRAIN_COMPONENT_NAMES}

    if mode == "course":
        size_path = "course.patch_size"
    elif mode == "stairs":
        size_path = "stairs.legacy_patch_size"
    else:
        size_path = "patch_size"
    size = tuple(float(x) for x in _get(config, size_path, (8.0, 8.0)))
    slope_min_deg = float(_get(config, "slope.min_angle_deg", 5.0))
    slope_max_deg = float(_get(config, "slope.max_angle_deg", 12.0))
    step_height = tuple(float(x) for x in _get(config, "stairs.step_height_range", (0.10, 0.18)))
    horizontal_scale = float(_get(config, "heightfield.horizontal_scale", 0.1))
    spawn_center_range = float(_get(config, "spawn.center_range", 1.25))

    all_sub_terrains = {
        "flat": TerrainBoxFlatCfg(proportion=weights["flat"]),
        # A safe center platform becomes higher away from the origin. This
        # keeps arbitrary LaFAN headings valid while providing uphill terrain.
        "slope": NeutralHfPyramidSlopedTerrainCfg(
            proportion=weights["slope"],
            slope_range=(math.tan(math.radians(slope_min_deg)), math.tan(math.radians(slope_max_deg))),
            platform_width=float(_get(config, "slope.platform_width", 1.5)),
            inverted=True,
            border_width=float(_get(config, "slope.border_width", 0.5)),
            horizontal_scale=horizontal_scale,
            vertical_scale=float(_get(config, "heightfield.vertical_scale", 0.005)),
        ),
        # Retain the old combined staircase as an explicit legacy inference
        # mode; mixed training uses separate ascent and descent families.
        "stairs": TerrainBoundedStairsCfg(
            proportion=1.0,
            step_height_range=step_height,
            step_width=float(_get(config, "stairs.step_depth", 0.30)),
            platform_width=float(_get(config, "stairs.legacy_platform_width", 1.0)),
            num_steps=int(_get(config, "stairs.legacy_num_steps", 6)),
            plateau_width=float(_get(config, "stairs.plateau_width", 0.8)),
            border_width=float(_get(config, "stairs.border_width", 0.5)),
        ),
        "stairs_up": TerrainSeparatedPyramidStairsCfg(
            proportion=weights["stairs_up"],
            direction="up",
            step_height_range=step_height,
            step_width=float(_get(config, "stairs.step_depth", 0.30)),
            platform_width=float(_get(config, "stairs.platform_width", 0.80)),
            num_steps=int(_get(config, "stairs.num_steps", 10)),
            border_width=float(_get(config, "stairs.border_width", 0.5)),
        ),
        "stairs_down": TerrainSeparatedPyramidStairsCfg(
            proportion=weights["stairs_down"],
            direction="down",
            step_height_range=step_height,
            step_width=float(_get(config, "stairs.step_depth", 0.30)),
            platform_width=float(_get(config, "stairs.platform_width", 0.80)),
            num_steps=int(_get(config, "stairs.num_steps", 10)),
            border_width=float(_get(config, "stairs.border_width", 0.5)),
        ),
        "rough": NeutralHfPerlinNoiseTerrainCfg(
            proportion=weights["rough"],
            height_range=tuple(float(x) for x in _get(config, "rough.amplitude_range", (0.03, 0.08))),
            octaves=int(_get(config, "rough.octaves", 2)),
            persistence=float(_get(config, "rough.persistence", 0.25)),
            lacunarity=float(_get(config, "rough.lacunarity", 2.0)),
            scale=float(_get(config, "rough.spatial_scale", 20.0)),
            horizontal_scale=horizontal_scale,
            resolution=horizontal_scale,
            border_width=float(_get(config, "rough.border_width", 0.5)),
            flat_patch_sampling=spawn_patch_sampling(
                patch_radius=float(_get(config, "spawn.patch_radius", 0.45)),
                patch_center_range=spawn_center_range,
                patch_size=size,
            ),
        ),
        "platforms": ScatteredBoxPlatformsTerrainCfg(
            proportion=weights["platforms"],
            easy_height_range=tuple(float(x) for x in _get(config, "platforms.easy_height_range", (0.05, 0.10))),
            hard_height_range=tuple(float(x) for x in _get(config, "platforms.hard_height_range", (0.10, 0.18))),
            center_width=float(_get(config, "platforms.center_width", 1.2)),
            border_width=float(_get(config, "platforms.border_width", 0.5)),
            gap_width=float(_get(config, "platforms.gap_width", 0.25)),
            min_platform_width=float(_get(config, "platforms.min_platform_width", 0.60)),
            raised_fraction=float(_get(config, "platforms.raised_fraction", 1.0)),
            row_count_range=tuple(int(x) for x in _get(config, "platforms.row_count_range", (5, 7))),
            column_count_range=tuple(int(x) for x in _get(config, "platforms.column_count_range", (4, 7))),
            spawn_patch_margin=float(_get(config, "reset.platforms_edge_margin", 0.25)),
            num_spawn_patches=int(_get(config, "platforms.num_spawn_patches", 12)),
            flat_patch_sampling={
                "platform_spawn": FlatPatchSamplingCfg(
                    num_patches=int(_get(config, "platforms.num_spawn_patches", 12)),
                    patch_radius=float(_get(config, "reset.platforms_edge_margin", 0.25)),
                )
            },
        ),
        "course": TerrainTraversalCourseCfg(
            proportion=1.0,
            flat_run=float(_get(config, "course.flat_run", 1.80)),
            step_height=float(_get(config, "course.step_height", 0.12)),
            step_depth=float(_get(config, "course.step_depth", 0.30)),
            num_steps=int(_get(config, "course.num_steps", 5)),
            top_platform_length=float(_get(config, "course.top_platform_length", 0.80)),
            connector_length=float(_get(config, "course.connector_length", 1.00)),
            include_ramp=bool(_get(config, "course.include_ramp", True)),
            ramp_length=float(_get(config, "course.ramp_length", 2.50)),
            ramp_angle_deg=float(_get(config, "course.ramp_angle_deg", 8.0)),
            border_width=float(_get(config, "course.border_width", 0.50)),
            horizontal_scale=float(_get(config, "heightfield.horizontal_scale", 0.10)),
        ),
    }
    sub_terrains = {name: all_sub_terrains[name] for name in selected}
    return TerrainGeneratorCfg(
        seed=int(_get(config, "seed", 0)),
        size=size,
        border_width=float(_get(config, "border_width", 1.0)),
        num_rows=int(_get(config, "num_rows", 10)),
        num_cols=len(sub_terrains),
        curriculum=True,
        difficulty_range=tuple(float(x) for x in _get(config, "difficulty_range", (0.0, 1.0))),
        color_scheme="none",
        sub_terrains=sub_terrains,
        add_lights=False,
    )


def make_rp1_simple_generator_cfg() -> TerrainGeneratorCfg:
    """Build the UFO-rp1 core with G1-depth perimeter and seam adaptations."""
    return TerrainGeneratorCfg(
        seed=0,
        size=(RP1_PATCH_SIZE, RP1_PATCH_SIZE),
        border_width=RP1_TERRAIN_BORDER_WIDTH,
        num_rows=10,
        num_cols=7,
        curriculum=True,
        difficulty_range=(0.0, 1.0),
        color_scheme="none",
        sub_terrains={
            "flat": BoxFlatTerrainCfg(proportion=RP1_TERRAIN_PROPORTIONS["flat"]),
            "perlin_rough": NeutralHfPerlinNoiseTerrainCfg(
                proportion=RP1_TERRAIN_PROPORTIONS["perlin_rough"],
                height_range=(0.01, 0.03),
                octaves=2,
                persistence=0.25,
                lacunarity=2.0,
                scale=20.0,
                horizontal_scale=0.2,
                resolution=0.2,
                border_width=0.0,
                geom_group=0,
            ),
            "low_stairs_up": BoxPyramidStairsTerrainCfg(
                proportion=RP1_TERRAIN_PROPORTIONS["low_stairs_up"],
                step_height_range=(0.05, 0.15),
                step_width=RP1_STAIR_STEP_WIDTH,
                platform_width=RP1_STAIR_PLATFORM_WIDTH,
                border_width=0.0,
            ),
            "low_stairs_down": BoxInvertedPyramidStairsTerrainCfg(
                proportion=RP1_TERRAIN_PROPORTIONS["low_stairs_down"],
                step_height_range=(0.05, 0.15),
                step_width=RP1_STAIR_STEP_WIDTH,
                platform_width=RP1_STAIR_PLATFORM_WIDTH,
                border_width=0.0,
            ),
            "low_platforms": BoxRandomGridTerrainCfg(
                proportion=RP1_TERRAIN_PROPORTIONS["low_platforms"],
                grid_width=0.8,
                grid_height_range=(0.03, 0.10),
                platform_width=RP1_CENTER_PLATFORM_WIDTH,
                merge_similar_heights=True,
                height_merge_threshold=0.025,
                max_merge_distance=3,
                border_width=0.0,
            ),
            "hf_pyramid_slope_inv": NeutralHfPyramidSlopedTerrainCfg(
                proportion=RP1_TERRAIN_PROPORTIONS["hf_pyramid_slope_inv"],
                slope_range=(0.0, 0.2),
                platform_width=RP1_CENTER_PLATFORM_WIDTH,
                inverted=True,
                border_width=0.0,
                horizontal_scale=0.2,
                geom_group=0,
            ),
            "boxes": NeutralHfDiscreteObstaclesTerrainCfg(
                proportion=RP1_TERRAIN_PROPORTIONS["boxes"],
                num_obstacles=50,
                obstacle_height_mode="choice",
                obstacle_width_range=(0.3, 1.0),
                obstacle_height_range=(0.05, 0.10),
                platform_width=RP1_CENTER_PLATFORM_WIDTH,
                border_width=0.0,
                horizontal_scale=0.2,
                vertical_scale=0.005,
                geom_group=0,
            ),
        },
        add_lights=False,
    )


def _make_rp1_guard_sub_terrains() -> dict[str, Any]:
    """Create efficient non-flat heightfields for the non-spawn guard ring."""
    return {
        "perlin_rough": NeutralHfPerlinNoiseTerrainCfg(
            proportion=0.5,
            height_range=(0.01, 0.03),
            octaves=2,
            persistence=0.25,
            lacunarity=2.0,
            scale=20.0,
            horizontal_scale=0.2,
            resolution=0.2,
            border_width=0.0,
            geom_group=0,
        ),
        "boxes": NeutralHfDiscreteObstaclesTerrainCfg(
            proportion=0.5,
            num_obstacles=50,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.3, 1.0),
            obstacle_height_range=(0.05, 0.10),
            platform_width=RP1_CENTER_PLATFORM_WIDTH,
            border_width=0.0,
            horizontal_scale=0.2,
            vertical_scale=0.005,
            geom_group=0,
        ),
    }


def add_rp1_nonflat_guard_ring(
    spec: mujoco.MjSpec,
    generator: TerrainGeneratorCfg,
) -> tuple[int, int]:
    """Add two non-spawn 5 m tile rings over the inner 10 m of the border.

    The generator-owned border remains as a continuous collision underlay.
    Guard heightfields are lifted slightly above it, while the outermost 2 m
    remains exposed flat terrain before the collision walls.
    """
    terrain_body = spec.body("terrain")
    if terrain_body is None:
        raise ValueError("RP1 guard ring requires a compiled terrain body")
    tile_x, tile_y = (float(value) for value in generator.size)
    rings_x = int(round(RP1_NONFLAT_GUARD_WIDTH / tile_x))
    rings_y = int(round(RP1_NONFLAT_GUARD_WIDTH / tile_y))
    if not math.isclose(rings_x * tile_x, RP1_NONFLAT_GUARD_WIDTH) or not math.isclose(
        rings_y * tile_y, RP1_NONFLAT_GUARD_WIDTH
    ):
        raise ValueError("RP1 non-flat guard width must be an integer number of terrain tiles")
    if not math.isclose(float(generator.border_width), RP1_TERRAIN_BORDER_WIDTH):
        raise ValueError(
            "RP1 generator border must equal the non-flat guard plus flat safety width"
        )

    num_rows = int(generator.num_rows)
    num_cols = len(generator.sub_terrains) if generator.curriculum else int(generator.num_cols)
    core_x_min = -num_rows * tile_x / 2.0
    core_y_min = -num_cols * tile_y / 2.0
    guard_cfgs = _make_rp1_guard_sub_terrains()
    for guard_cfg in guard_cfgs.values():
        guard_cfg.size = (tile_x, tile_y)
    rng = np.random.default_rng(int(generator.seed or 0) + 104_729)
    lower, upper = (float(value) for value in generator.difficulty_range)
    tile_count = 0
    geom_count = 0
    for row in range(-rings_x, num_rows + rings_x):
        for col in range(-rings_y, num_cols + rings_y):
            if 0 <= row < num_rows and 0 <= col < num_cols:
                continue
            family = RP1_GUARD_FAMILIES[tile_count % len(RP1_GUARD_FAMILIES)]
            guard_cfg = guard_cfgs[family]
            phase = ((tile_count % 10) + 0.5) / 10.0
            difficulty = lower + phase * (upper - lower)
            output = guard_cfg.function(difficulty, spec, rng)
            world_offset = np.array(
                (
                    core_x_min + row * tile_x,
                    core_y_min + col * tile_y,
                    RP1_GUARD_HEIGHT_LIFT,
                )
            )
            for local_geom_index, geometry in enumerate(output.geometries):
                geom = geometry.geom
                if geom is None:
                    continue
                geom.pos = np.asarray(geom.pos) + world_offset
                geom.name = f"g1depth_guard_tile_{tile_count:03d}_geom_{local_geom_index:03d}"
                geom.group = 0
                geom.mass = 0.0
                geom.rgba = (0.5, 0.5, 0.5, 1.0)
                geom_count += 1
            tile_count += 1
    return tile_count, geom_count


def add_rp1_outer_walls(spec: mujoco.MjSpec, generator: TerrainGeneratorCfg) -> None:
    """Add four collision walls outside the 10 m guard and 2 m flat safety ring."""
    terrain_body = spec.body("terrain")
    if terrain_body is None:
        raise ValueError("RP1 outer walls require a compiled terrain body")
    num_cols = len(generator.sub_terrains) if generator.curriculum else generator.num_cols
    outer_x = generator.num_rows * generator.size[0] + 2.0 * generator.border_width
    outer_y = num_cols * generator.size[1] + 2.0 * generator.border_width
    half_height = RP1_OUTER_WALL_HEIGHT / 2.0
    half_thickness = RP1_OUTER_WALL_THICKNESS / 2.0
    walls = (
        (
            "rp1_outer_wall_x_negative",
            (-outer_x / 2.0 - half_thickness, 0.0, half_height),
            (half_thickness, outer_y / 2.0 + RP1_OUTER_WALL_THICKNESS, half_height),
        ),
        (
            "rp1_outer_wall_x_positive",
            (outer_x / 2.0 + half_thickness, 0.0, half_height),
            (half_thickness, outer_y / 2.0 + RP1_OUTER_WALL_THICKNESS, half_height),
        ),
        (
            "rp1_outer_wall_y_negative",
            (0.0, -outer_y / 2.0 - half_thickness, half_height),
            (outer_x / 2.0, half_thickness, half_height),
        ),
        (
            "rp1_outer_wall_y_positive",
            (0.0, outer_y / 2.0 + half_thickness, half_height),
            (outer_x / 2.0, half_thickness, half_height),
        ),
    )
    for name, pos, size in walls:
        wall = terrain_body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size)
        wall.rgba = RP1_OUTER_WALL_COLOR
        wall.group = 0
        wall.contype = 1
        wall.conaffinity = 1


def make_terrain_entity_cfg(
    mode: TerrainMode,
    *,
    env_spacing: float,
    config: Any = None,
    inference_all_stairs: bool = False,
    inference_all_flat: bool = False,
) -> TerrainEntityCfg:
    if mode == "plane":
        return TerrainEntityCfg(terrain_type="plane", env_spacing=env_spacing)
    if mode not in SUPPORTED_TERRAINS:
        raise ValueError(f"Unsupported terrain mode: {mode!r}. Expected one of {SUPPORTED_TERRAINS}")
    generator = make_ufo_v0_generator_cfg(mode, config)
    if mode == "rp1_simple":
        if inference_all_stairs and inference_all_flat:
            raise ValueError("inference_all_stairs and inference_all_flat are mutually exclusive")
        if inference_all_stairs:
            generator.sub_terrains = {
                name: BoxPyramidStairsTerrainCfg(
                    proportion=RP1_TERRAIN_PROPORTIONS[name],
                    step_height_range=(0.15, 0.15),
                    step_width=RP1_STAIR_STEP_WIDTH,
                    platform_width=RP1_STAIR_PLATFORM_WIDTH,
                    border_width=0.0,
                )
                for name in generator.sub_terrains
            }
        elif inference_all_flat:
            generator.sub_terrains = {
                name: BoxFlatTerrainCfg(proportion=RP1_TERRAIN_PROPORTIONS[name])
                for name in generator.sub_terrains
            }
    return TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=generator,
        env_spacing=None,
        max_init_terrain_level=generator.num_rows - 1,
    )
