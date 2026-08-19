"""Physical terrain presets for the opt-in UFO terrain feasibility experiment."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal

from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg

from humanoidverse.terrains.rp1_primitives import (
    NeutralHfPerlinNoiseTerrainCfg,
    NeutralHfPyramidSlopedTerrainCfg,
    TerrainBoxFlatCfg,
    TerrainBoundedStairsCfg,
    spawn_patch_sampling,
)

TerrainMode = Literal["plane", "flat", "slope", "stairs", "rough", "mixed", "rp1_simple"]
SUPPORTED_TERRAINS: tuple[TerrainMode, ...] = (
    "plane",
    "flat",
    "slope",
    "stairs",
    "rough",
    "mixed",
    "rp1_simple",
)
TERRAIN_COMPONENT_NAMES = ("flat", "slope", "stairs", "rough")


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
    if mode in {"mixed", "rp1_simple"}:
        return TERRAIN_COMPONENT_NAMES
    if mode == "plane":
        return ("flat",)
    return (mode,)


def _terrain_mix(config: Any) -> dict[str, float]:
    raw = _get(config, "terrain_mix", None)
    weights = {name: float(_get(raw, name, 0.25)) for name in TERRAIN_COMPONENT_NAMES}
    if any(value < 0.0 for value in weights.values()) or sum(weights.values()) <= 0.0:
        raise ValueError(f"terrain_mix must be non-negative and have positive total weight: {weights}")
    total = sum(weights.values())
    return {name: value / total for name, value in weights.items()}


def make_ufo_v0_generator_cfg(mode: TerrainMode, config: Any) -> TerrainGeneratorCfg:
    """Create collision terrain and spawn origins from one MJLab generator."""
    if mode == "rp1_simple":
        mode = "mixed"
    selected = terrain_component_names(mode)
    weights = _terrain_mix(config)
    if mode != "mixed":
        weights = {name: float(name == mode) for name in TERRAIN_COMPONENT_NAMES}

    size = tuple(float(x) for x in _get(config, "patch_size", (8.0, 8.0)))
    slope_min_deg = float(_get(config, "slope.min_angle_deg", 5.0))
    slope_max_deg = float(_get(config, "slope.max_angle_deg", 12.0))
    step_height = tuple(float(x) for x in _get(config, "stairs.step_height_range", (0.08, 0.15)))
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
        "stairs": TerrainBoundedStairsCfg(
            proportion=weights["stairs"],
            step_height_range=step_height,
            step_width=float(_get(config, "stairs.step_depth", 0.30)),
            platform_width=float(_get(config, "stairs.platform_width", 1.5)),
            num_steps=int(_get(config, "stairs.num_steps", 4)),
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
    """Backward-compatible name for the default mixed V0 generator."""
    return make_ufo_v0_generator_cfg("mixed", None)


def make_terrain_entity_cfg(mode: TerrainMode, *, env_spacing: float, config: Any = None) -> TerrainEntityCfg:
    if mode == "plane":
        return TerrainEntityCfg(terrain_type="plane", env_spacing=env_spacing)
    if mode not in SUPPORTED_TERRAINS:
        raise ValueError(f"Unsupported terrain mode: {mode!r}. Expected one of {SUPPORTED_TERRAINS}")
    generator = make_ufo_v0_generator_cfg(mode, config)
    return TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=generator,
        env_spacing=None,
        max_init_terrain_level=generator.num_rows - 1,
    )
