from __future__ import annotations

import mujoco
import numpy as np
from mjlab.terrains import (
    BoxFlatTerrainCfg,
    BoxInvertedPyramidStairsTerrainCfg,
    BoxPyramidStairsTerrainCfg,
    BoxRandomGridTerrainCfg,
    TerrainGenerator,
)

from humanoidverse.terrains import make_terrain_entity_cfg, terrain_component_names
from humanoidverse.terrains.rp1_primitives import (
    NeutralHfDiscreteObstaclesTerrainCfg,
    NeutralHfPerlinNoiseTerrainCfg,
    NeutralHfPyramidSlopedTerrainCfg,
)
from humanoidverse.terrains.rp1_simple import (
    RP1_FLAT_SAFETY_WIDTH,
    RP1_GUARD_TILE_RINGS,
    RP1_NONFLAT_GUARD_WIDTH,
    RP1_PATCH_SIZE,
    RP1_TERRAIN_BORDER_WIDTH,
    RP1_TERRAIN_COMPONENT_NAMES,
    RP1_TERRAIN_PROPORTIONS,
    RP1_TERRAIN_REFERENCE_COMMIT,
    RP1_TERRAIN_REFERENCE_PROJECT,
    add_rp1_nonflat_guard_ring,
    add_rp1_outer_walls,
    make_rp1_simple_generator_cfg,
)
from humanoidverse.agents.utils import AnchoredEveryNStepsChecker
from humanoidverse.train import build_ufo_mjlab_config


def test_rp1_generator_matches_pinned_source_core_with_g1_perimeter() -> None:
    cfg = make_rp1_simple_generator_cfg()
    assert RP1_TERRAIN_REFERENCE_PROJECT == "UFO-rp1"
    assert RP1_TERRAIN_REFERENCE_COMMIT == "8c364e1001734097aac58e5033a1b5076925d3c5"
    assert cfg.seed == 0
    assert cfg.size == (5.0, 5.0)
    assert cfg.border_width == 12.0
    assert cfg.border_width == RP1_NONFLAT_GUARD_WIDTH + RP1_FLAT_SAFETY_WIDTH
    assert RP1_PATCH_SIZE / 2.0 + cfg.border_width == 14.5
    assert RP1_GUARD_TILE_RINGS == 2
    assert cfg.num_rows == 10
    assert cfg.num_cols == 7
    assert cfg.curriculum
    assert cfg.difficulty_range == (0.0, 1.0)
    assert tuple(cfg.sub_terrains) == RP1_TERRAIN_COMPONENT_NAMES
    assert terrain_component_names("rp1_simple") == RP1_TERRAIN_COMPONENT_NAMES
    assert {name: sub.proportion for name, sub in cfg.sub_terrains.items()} == RP1_TERRAIN_PROPORTIONS

    expected_types = {
        "flat": BoxFlatTerrainCfg,
        "perlin_rough": NeutralHfPerlinNoiseTerrainCfg,
        "low_stairs_up": BoxPyramidStairsTerrainCfg,
        "low_stairs_down": BoxInvertedPyramidStairsTerrainCfg,
        "low_platforms": BoxRandomGridTerrainCfg,
        "hf_pyramid_slope_inv": NeutralHfPyramidSlopedTerrainCfg,
        "boxes": NeutralHfDiscreteObstaclesTerrainCfg,
    }
    assert {name: type(sub) for name, sub in cfg.sub_terrains.items()} == expected_types
    assert cfg.sub_terrains["perlin_rough"].height_range == (0.01, 0.03)
    assert cfg.sub_terrains["low_stairs_up"].step_height_range == (0.05, 0.15)
    assert cfg.sub_terrains["low_stairs_down"].step_height_range == (0.05, 0.15)
    assert cfg.sub_terrains["low_platforms"].grid_height_range == (0.03, 0.10)
    assert cfg.sub_terrains["hf_pyramid_slope_inv"].slope_range == (0.0, 0.2)
    assert cfg.sub_terrains["boxes"].num_obstacles == 50
    assert cfg.sub_terrains["perlin_rough"].geom_group == 0
    assert cfg.sub_terrains["hf_pyramid_slope_inv"].geom_group == 0
    assert cfg.sub_terrains["boxes"].geom_group == 0
    for name in (
        "perlin_rough",
        "low_stairs_up",
        "low_stairs_down",
        "low_platforms",
        "hf_pyramid_slope_inv",
        "boxes",
    ):
        assert cfg.sub_terrains[name].border_width == 0.0


def test_rp1_entity_and_outer_walls_match_grid_extent() -> None:
    entity = make_terrain_entity_cfg("rp1_simple", env_spacing=2.0)
    generator = entity.terrain_generator
    assert generator is not None
    assert generator.size == (RP1_PATCH_SIZE, RP1_PATCH_SIZE)
    assert generator.border_width == RP1_TERRAIN_BORDER_WIDTH
    assert entity.max_init_terrain_level == generator.num_rows - 1

    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    add_rp1_outer_walls(spec, generator)
    walls = [geom for geom in spec.geoms if str(geom.name).startswith("rp1_outer_wall_")]
    assert len(walls) == 4
    assert all(geom.group == 0 and geom.contype == 1 and geom.conaffinity == 1 for geom in walls)


def test_rp1_nonflat_guard_has_two_tile_rings_and_is_not_spawnable() -> None:
    generator = make_rp1_simple_generator_cfg()
    spec = mujoco.MjSpec()
    TerrainGenerator(generator).compile(spec)
    tile_count, geom_count = add_rp1_nonflat_guard_ring(spec, generator)
    assert tile_count == (10 + 2 * RP1_GUARD_TILE_RINGS) * (7 + 2 * RP1_GUARD_TILE_RINGS) - 10 * 7
    assert tile_count == 84
    assert geom_count == tile_count
    guard_geoms = [geom for geom in spec.geoms if str(geom.name).startswith("g1depth_guard_tile_")]
    assert len(guard_geoms) == geom_count
    assert {int(geom.group) for geom in guard_geoms} == {0}
    assert max(abs(float(geom.pos[0])) for geom in guard_geoms) == 32.5
    assert max(abs(float(geom.pos[1])) for geom in guard_geoms) == 25.0

    model = spec.compile()
    guard_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if (model.geom(geom_id).name or "").startswith("g1depth_guard_tile_")
    ]
    assert len(guard_geom_ids) == geom_count
    guard_hfield_ids = model.geom_dataid[guard_geom_ids]
    assert np.all(guard_hfield_ids >= 0)
    for hfield_id in guard_hfield_ids:
        data_start = int(model.hfield_adr[hfield_id])
        data_size = int(model.hfield_nrow[hfield_id] * model.hfield_ncol[hfield_id])
        assert np.ptp(model.hfield_data[data_start : data_start + data_size]) > 0.0

    # Only the original 10x7 core contributes reset origins. Guard tiles are
    # scene geometry and cannot be selected by TerrainEntity reset sampling.
    assert generator.num_rows == 10
    assert len(generator.sub_terrains) == 7


def test_every_rp1_terrain_family_uses_the_source_geom_group() -> None:
    generator = make_rp1_simple_generator_cfg()
    for name, sub_terrain in generator.sub_terrains.items():
        sub_terrain.size = generator.size
        spec = mujoco.MjSpec()
        spec.worldbody.add_body(name="terrain")
        output = sub_terrain.function(0.8, spec, np.random.default_rng(3))
        groups = {
            int(geometry.geom.group)
            for geometry in output.geometries
            if geometry.geom is not None
        }
        assert groups == {0}, f"{name} produced geom groups {groups}"


def test_fb_depth_defaults_to_rp1_simple_without_changing_fb_terrain_default() -> None:
    common = dict(
        device="cpu",
        work_dir="/tmp/ufo-rp1-terrain-config-test",
        num_envs=2,
        num_env_steps=2048,
        seed=7,
        use_wandb=False,
        wandb_run_name=None,
        disable_eval_prioritization=True,
        smoke=True,
    )
    depth = build_ufo_mjlab_config(agent="fb_depth", terrain_mode=None, **common)
    terrain = build_ufo_mjlab_config(agent="fb_terrain", terrain_mode=None, **common)
    assert "terrain.terrain_type=rp1_simple" in depth.env.hydra_overrides
    assert "terrain.terrain_type=mixed" in terrain.env.hydra_overrides


def test_fb_depth_uses_separate_checkpoint_tracking_and_same_z_cadences() -> None:
    cfg = build_ufo_mjlab_config(
        device="cpu",
        work_dir="/tmp/ufo-rp1-eval-cadence-test",
        num_envs=2,
        num_env_steps=2048,
        seed=7,
        use_wandb=False,
        wandb_run_name=None,
        disable_eval_prioritization=False,
        smoke=False,
        agent="fb_depth",
        terrain_mode="rp1_simple",
    )
    evaluations = {evaluation.name_in_logs: evaluation for evaluation in cfg.evaluations}
    assert cfg.checkpoint_every_steps == 9_600_000
    assert evaluations["humanoidverse_tracking_eval"].every_steps == 3_200_000
    assert evaluations["same_z_terrain_eval"].every_steps == 9_600_000


def test_tracking_third_trigger_aligns_with_same_z_and_checkpoint() -> None:
    tracking = AnchoredEveryNStepsChecker(0, 3_200_000)
    same_z = AnchoredEveryNStepsChecker(0, 9_600_000)
    checkpoint = AnchoredEveryNStepsChecker(0, 9_600_000)
    tracking_triggers = []
    shared_triggers = []
    for step in range(0, 9_700_000, 8192):
        if tracking.check(step):
            tracking_triggers.append(step)
            tracking.update_last_step(step)
        if same_z.check(step):
            shared_triggers.append(step)
            same_z.update_last_step(step)
        if checkpoint.check(step):
            assert shared_triggers[-1] == step
            checkpoint.update_last_step(step)
    assert tracking_triggers == [0, 3_203_072, 6_406_144, 9_601_024]
    assert shared_triggers == [0, 9_601_024]
