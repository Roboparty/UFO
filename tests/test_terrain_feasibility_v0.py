from __future__ import annotations

import inspect
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gymnasium
import mujoco
import numpy as np
import torch
from mjlab.terrains import TerrainGenerator
from mjlab.terrains.terrain_entity import _proportional_counts
from omegaconf import OmegaConf

from humanoidverse.agents.buffers.transition import DictBuffer
from humanoidverse.agents.envs.humanoidverse_mjlab import (
    HumanoidVerseMjlabConfig,
    HumanoidVerseMjlabCore,
    body_contact_severity,
    peak_contact_force,
    tangential_contact_speed,
)
from humanoidverse.agents.evaluations.humanoidverse_mjlab import _calc_metrics, emd_numpy
from humanoidverse.agents.presets import build_agent_preset
from humanoidverse.envs.motion_observations import compute_humanoid_observations_max
from humanoidverse.mjlab_inference_utils import (
    _camera_lookat_from_root,
    _inference_scene_option,
    _limit_absolute_near_clip,
    _style_untextured_terrain,
)
from humanoidverse.terrain_transfer import clone_same_z_for_terrains, tensor_checksum
from humanoidverse.terrain_transfer_inference import _course_completion_radius, _stairs_step_center_offset
from humanoidverse.terrains import make_terrain_entity_cfg, terrain_component_names
from humanoidverse.terrains.coverage import validate_motion_terrain_coverage
from humanoidverse.terrains.rp1_simple import make_ufo_v0_generator_cfg
from humanoidverse.terrains.terrain_observation import (
    RobotCentricGridPatternCfg,
    flat_zero_observations,
    observations_from_clearances,
    reference_ray_index,
)
from humanoidverse.tracking_inference import _center_target_states_on_terrain
from humanoidverse.training.workspace import Workspace, make_flat_terrain_priority_eval_config


def _preset(name: str):
    return build_agent_preset(
        agent=name,
        device="cpu",
        compile=False,
        update_z_every_step=100,
        lr_scale=1.0,
        clip_grad_norm=0.0,
        cartwheel_aux_safe=False,
        wandb_project="test",
    )["agent_cfg"]


def test_tracking_target_reset_is_centered_only_for_terrain() -> None:
    root_states = torch.tensor([[8.0, -7.0, 0.82, 0.0, 0.0, 0.0, 1.0, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0]])
    target_states = {"root_states": root_states, "dof_states": torch.zeros(1, 29, 2)}
    terrain_env = SimpleNamespace(terrain_enabled=True, env_origins=torch.tensor([[1.5, -2.5, 0.0]]))

    centered = _center_target_states_on_terrain(target_states, terrain_env)

    torch.testing.assert_close(centered["root_states"][:, :2], terrain_env.env_origins[:, :2])
    torch.testing.assert_close(centered["root_states"][:, 2:], root_states[:, 2:])
    torch.testing.assert_close(target_states["root_states"], root_states)
    plain_env = SimpleNamespace(terrain_enabled=False, env_origins=torch.zeros(1, 3))
    assert _center_target_states_on_terrain(target_states, plain_env) is target_states


def test_stairs_inference_start_uses_requested_step_center() -> None:
    assert _stairs_step_center_offset(0, platform_width=1.5, step_depth=0.3) == 0.0
    assert math.isclose(
        _stairs_step_center_offset(5, platform_width=1.5, step_depth=0.3),
        2.10,
    )
    with np.testing.assert_raises(ValueError):
        _stairs_step_center_offset(-1, platform_width=1.5, step_depth=0.3)


def test_course_completion_radius_is_inside_final_flat() -> None:
    course = OmegaConf.load("humanoidverse/config/terrain/terrain_ufo_v0.yaml").terrain.course
    assert math.isclose(_course_completion_radius(course), 9.4)


def _physics_height_profile(mode: str) -> np.ndarray:
    config = OmegaConf.load("humanoidverse/config/terrain/terrain_ufo_v0.yaml").terrain
    config = OmegaConf.merge(config, {"num_rows": 1, "difficulty_range": [1.0, 1.0]})
    generator = TerrainGenerator(make_ufo_v0_generator_cfg(mode, config))
    spec = mujoco.MjSpec()
    generator.compile(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    origin = generator.terrain_origins[0, 0]
    heights = []
    terrain_group = np.array([0, 0, 0, 0, 0, 1], dtype=np.uint8)
    for x_offset in np.linspace(0.0, 1.6, 17):
        point = np.array([origin[0] + x_offset, origin[1], origin[2] + 3.0])
        distance = mujoco.mj_ray(
            model,
            data,
            point,
            np.array([0.0, 0.0, -1.0]),
            terrain_group,
            1,
            -1,
            np.array([-1], dtype=np.int32),
        )
        if distance < 0:
            raise AssertionError(f"ray missed {mode} collision geometry at x={x_offset}")
        heights.append(point[2] - distance)
    values = np.asarray(heights)
    return values - values[0]


def _physics_heights_at_offsets(mode: str, offsets: list[float]) -> np.ndarray:
    return _physics_heights_at_xy_offsets(mode, [(offset, 0.0) for offset in offsets])


def _physics_heights_at_xy_offsets(mode: str, offsets: list[tuple[float, float]]) -> np.ndarray:
    config = OmegaConf.load("humanoidverse/config/terrain/terrain_ufo_v0.yaml").terrain
    config = OmegaConf.merge(config, {"num_rows": 1, "difficulty_range": [1.0, 1.0]})
    generator = TerrainGenerator(make_ufo_v0_generator_cfg(mode, config))
    spec = mujoco.MjSpec()
    generator.compile(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    origin = generator.terrain_origins[0, 0]
    heights = []
    terrain_group = np.array([0, 0, 0, 0, 0, 1], dtype=np.uint8)
    for x_offset, y_offset in offsets:
        point = np.array([origin[0] + x_offset, origin[1] + y_offset, origin[2] + 3.0])
        distance = mujoco.mj_ray(
            model,
            data,
            point,
            np.array([0.0, 0.0, -1.0]),
            terrain_group,
            1,
            -1,
            np.array([-1], dtype=np.int32),
        )
        if distance < 0:
            raise AssertionError(f"ray missed {mode} collision geometry at ({x_offset}, {y_offset})")
        heights.append(point[2] - distance)
    return np.asarray(heights)


def _body_state(root_height: float) -> tuple[torch.Tensor, ...]:
    body_pos = torch.tensor([[[0.0, 0.0, root_height], [0.2, 0.0, root_height - 0.3]]])
    body_rot = torch.tensor([[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]])
    body_vel = torch.tensor([[[0.4, 0.0, 0.0], [0.4, 0.0, 0.0]]])
    body_ang_vel = torch.tensor([[[0.0, 0.1, 0.0], [0.0, 0.1, 0.0]]])
    return body_pos, body_rot, body_vel, body_ang_vel


def _canonical_privileged_state(root_height: float, ground_height: float) -> torch.Tensor:
    obs = compute_humanoid_observations_max(*_body_state(root_height), local_root_obs=True, root_height_obs=True)
    clearances = torch.full((1, 273), root_height - ground_height)
    root_clearance, _terrain_actor, _terrain_priv = observations_from_clearances(clearances, 58, clip=0.5)
    obs["root_height"] = root_clearance
    return torch.cat(list(obs.values()), dim=-1)


class TerrainPhysicsObservationTest(unittest.TestCase):
    def test_peak_contact_force_selects_strongest_physics_substep(self) -> None:
        history = torch.tensor([[[[0.0, 0.0, 10.0], [0.0, 0.0, 80.0], [0.0, 0.0, 20.0]]]])

        torch.testing.assert_close(peak_contact_force(history), torch.tensor([[[0.0, 0.0, 80.0]]]))

    def test_body_contact_severity_distinguishes_impact_direction_and_speed(self) -> None:
        # MJLab's primary-to-secondary force points downward for a body-ground contact.
        force = torch.tensor([[[0.0, 0.0, -100.0]]])
        weight = torch.tensor([500.0])
        falling = body_contact_severity(force, torch.tensor([[[0.0, 0.0, -1.5]]]), weight)
        resting = body_contact_severity(force, torch.zeros_like(force), weight)
        rising = body_contact_severity(force, torch.tensor([[[0.0, 0.0, 1.5]]]), weight)
        light = body_contact_severity(torch.tensor([[[0.0, 0.0, -20.0]]]), torch.zeros_like(force), weight)

        self.assertGreater(falling.item(), resting.item())
        torch.testing.assert_close(rising, resting)
        torch.testing.assert_close(light, torch.zeros_like(light))

    def test_slippage_uses_only_loaded_tangential_velocity(self) -> None:
        force = torch.tensor([[[0.0, 0.0, -100.0]]])
        weight = torch.tensor([500.0])

        normal = tangential_contact_speed(force, torch.tensor([[[0.0, 0.0, 2.0]]]), weight)
        tangent = tangential_contact_speed(force, torch.tensor([[[0.6, 0.8, 2.0]]]), weight)

        torch.testing.assert_close(normal, torch.zeros_like(normal), atol=1.0e-6, rtol=0.0)
        torch.testing.assert_close(tangent, torch.ones_like(tangent), atol=1.0e-6, rtol=0.0)

    def test_inference_camera_tracks_negative_world_height_on_sloped_terrain(self) -> None:
        root = np.array([-135.0, 0.0, -0.39])

        np.testing.assert_allclose(_camera_lookat_from_root(root), root)

    def test_inference_renderer_shows_mjlab_terrain_geom_group(self) -> None:
        option = _inference_scene_option()

        np.testing.assert_array_equal(option.geomgroup, np.array([1, 1, 1, 0, 0, 1], dtype=np.uint8))
        self.assertEqual(int(option.geomgroup[5]), 1)

    def test_inference_renderer_only_recolors_untextured_terrain(self) -> None:
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><asset><material name="terrain_material" rgba="1 1 1 1"/></asset><worldbody>'
            '<geom name="terrain" group="5" type="plane" size="1 1 .1" rgba="1 1 1 1" material="terrain_material"/>'
            '<geom name="colored" group="5" type="box" size=".1 .1 .1" rgba=".2 .3 .4 1"/>'
            '<geom name="robot" group="0" type="sphere" size=".1" rgba="1 1 1 1"/>'
            '</worldbody></mujoco>'
        )

        _style_untextured_terrain(model)

        np.testing.assert_allclose(model.geom_rgba[0], [0.24, 0.30, 0.27, 1.0])
        np.testing.assert_allclose(model.geom_rgba[1], [0.2, 0.3, 0.4, 1.0])
        np.testing.assert_allclose(model.geom_rgba[2], [1.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_equal(model.geom_matid, [-1, -1, -1])

    def test_large_terrain_scene_keeps_renderer_near_clip_in_front_of_robot(self) -> None:
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><statistic extent="333"/><worldbody><geom type="plane" size="1 1 .1"/></worldbody></mujoco>'
        )
        self.assertGreater(float(model.vis.map.znear) * float(model.stat.extent), 3.0)

        absolute_near = _limit_absolute_near_clip(model)

        self.assertAlmostEqual(absolute_near, 0.015, places=6)
        self.assertLess(float(model.vis.map.znear), 0.0001)

    def test_boundary_violation_is_counted_and_returned_for_reset(self) -> None:
        core = HumanoidVerseMjlabCore.__new__(HumanoidVerseMjlabCore)
        core.num_envs = 3
        core.device = "cpu"
        core._terrain_boundary_required = 2.0
        core._terrain_boundary_min = torch.tensor(float("inf"))
        core._terrain_boundary_violation_count = torch.zeros((), dtype=torch.long)
        core._terrain_fail_on_boundary = False
        core._terrain_boundary_margin = Mock(return_value=torch.tensor([3.0, 1.5, 2.0]))

        violations = core._check_terrain_boundary()

        torch.testing.assert_close(violations, torch.tensor([False, True, False]))
        self.assertEqual(core._terrain_boundary_violation_count.item(), 1)
        self.assertEqual(core._terrain_boundary_min.item(), 1.5)
        self.assertIn("boundary_resets", inspect.getsource(HumanoidVerseMjlabCore.step))

    def test_boundary_debug_fail_fast_uses_python_exception(self) -> None:
        core = HumanoidVerseMjlabCore.__new__(HumanoidVerseMjlabCore)
        core.num_envs = 1
        core.device = "cpu"
        core._terrain_boundary_required = 2.0
        core._terrain_boundary_min = torch.tensor(float("inf"))
        core._terrain_boundary_violation_count = torch.zeros((), dtype=torch.long)
        core._terrain_fail_on_boundary = True
        core._terrain_boundary_margin = Mock(return_value=torch.tensor([1.25]))

        with self.assertRaisesRegex(RuntimeError, "minimum margin=1.250000m"):
            core._check_terrain_boundary()

    def test_grid_is_robot_forward_asymmetric_and_fixed_dim(self) -> None:
        pattern = RobotCentricGridPatternCfg()
        offsets, directions = pattern.generate_rays(None, "cpu")
        self.assertEqual(pattern.shape, (21, 13))
        self.assertEqual(pattern.dimension, 273)
        self.assertEqual(tuple(offsets.shape), (273, 3))
        self.assertAlmostEqual(float(offsets[:, 0].min()), -0.4, places=5)
        self.assertAlmostEqual(float(offsets[:, 0].max()), 1.6, places=5)
        self.assertTrue(torch.all(directions[:, 2] == -1.0))

    def test_reference_ray_is_unique_robot_origin_not_array_midpoint(self) -> None:
        offsets, _directions = RobotCentricGridPatternCfg().generate_rays(None, "cpu")
        index = reference_ray_index(offsets)
        self.assertEqual(index, 58)
        torch.testing.assert_close(offsets[index], torch.zeros(3))
        at_origin = torch.all(torch.isclose(offsets[:, :2], torch.zeros_like(offsets[:, :2]), atol=1e-6, rtol=0.0), dim=-1)
        self.assertEqual(int(torch.count_nonzero(at_origin)), 1)
        self.assertNotEqual(index, offsets.shape[0] // 2)

    def test_clearance_builds_canonical_height_and_zero_center_terrain(self) -> None:
        clearances = torch.linspace(0.2, 1.2, 273).unsqueeze(0)
        root_height, terrain_actor, terrain_priv = observations_from_clearances(clearances, 58, clip=0.5)
        torch.testing.assert_close(root_height, clearances[:, 58:59])
        torch.testing.assert_close(terrain_actor, clearances)
        torch.testing.assert_close(terrain_priv[:, 58], torch.zeros(1))

    def test_actor_clearances_make_vertical_translation_observable(self) -> None:
        low = torch.full((1, 273), 0.55)
        high = torch.full((1, 273), 0.82)
        _low_root, low_actor, low_priv = observations_from_clearances(low, 58, clip=0.5)
        _high_root, high_actor, high_priv = observations_from_clearances(high, 58, clip=0.5)
        self.assertFalse(torch.equal(low_actor, high_actor))
        torch.testing.assert_close(low_priv, high_priv, atol=0.0, rtol=0.0)

    def test_fast_flat_is_numerically_equivalent_to_flat_raycast(self) -> None:
        pelvis_world_z = torch.tensor([[0.82], [0.55], [1.10]])
        flat_clearances = pelvis_world_z.expand(-1, 273).clone()
        ray_root_height, ray_terrain_actor, ray_terrain_priv = observations_from_clearances(flat_clearances, 58, clip=0.5)
        fast_root_height, fast_terrain_actor, fast_terrain_priv = flat_zero_observations(pelvis_world_z, 273)

        torch.testing.assert_close(fast_root_height, ray_root_height, atol=0.0, rtol=0.0)
        torch.testing.assert_close(fast_terrain_actor, ray_terrain_actor, atol=0.0, rtol=0.0)
        torch.testing.assert_close(fast_terrain_priv, ray_terrain_priv, atol=0.0, rtol=0.0)
        torch.testing.assert_close(fast_terrain_priv, torch.zeros_like(fast_terrain_priv), atol=0.0, rtol=0.0)

    def test_full_privileged_state_is_vertical_translation_invariant(self) -> None:
        ground_state = _canonical_privileged_state(root_height=0.82, ground_height=0.0)
        elevated_state = _canonical_privileged_state(root_height=1.82, ground_height=1.0)
        torch.testing.assert_close(ground_state, elevated_state, atol=1e-6, rtol=0.0)

    def test_stair_levels_preserve_canonical_body_state(self) -> None:
        lower_step = _canonical_privileged_state(root_height=0.82, ground_height=0.0)
        upper_step = _canonical_privileged_state(root_height=1.42, ground_height=0.6)
        torch.testing.assert_close(lower_step, upper_step, atol=1e-6, rtol=0.0)

    def test_original_observation_helper_retains_world_root_height(self) -> None:
        low = compute_humanoid_observations_max(*_body_state(0.82), local_root_obs=True, root_height_obs=True)
        high = compute_humanoid_observations_max(*_body_state(1.82), local_root_obs=True, root_height_obs=True)
        torch.testing.assert_close(low["root_height"], torch.tensor([[0.82]]))
        torch.testing.assert_close(high["root_height"], torch.tensor([[1.82]]))

    def test_flat_collision_heights_are_zero(self) -> None:
        np.testing.assert_allclose(_physics_height_profile("flat"), 0.0, atol=1e-6)

    def test_plane_with_terrain_config_remains_infinite_plane(self) -> None:
        terrain = OmegaConf.load("humanoidverse/config/terrain/terrain_ufo_v0.yaml").terrain
        entity = make_terrain_entity_cfg("plane", env_spacing=4.0, config=terrain)
        self.assertEqual(entity.terrain_type, "plane")
        self.assertIsNone(entity.terrain_generator)

    def test_slope_collision_heights_increase_forward(self) -> None:
        values = _physics_height_profile("slope")
        self.assertGreater(values[-1], values[0] + 0.02)
        self.assertGreaterEqual(float(np.diff(values).min()), -1e-5)

    def test_stairs_have_bounded_up_platform_down_profile(self) -> None:
        offsets = [
            0.0,
            0.65,
            0.95,
            1.25,
            1.55,
            1.85,
            2.15,
            2.7,
            3.65,
            3.95,
            4.25,
            4.55,
            4.85,
            5.15,
            5.6,
        ]
        values = _physics_heights_at_offsets("stairs", offsets)
        expected = [
            0.0,
            0.18,
            0.36,
            0.54,
            0.72,
            0.90,
            1.08,
            1.08,
            0.90,
            0.72,
            0.54,
            0.36,
            0.18,
            0.0,
            0.0,
        ]
        np.testing.assert_allclose(values, expected, atol=1e-5)

    def test_stairs_repeat_the_complete_cycle_across_the_patch(self) -> None:
        offsets = [5.8, 6.65, 8.15, 8.7, 9.65, 11.15, 12.0, 13.5]
        values = _physics_heights_at_offsets("stairs", offsets)
        expected = [0.0, 0.18, 1.08, 1.08, 0.90, 0.0, 0.0, 0.0]
        np.testing.assert_allclose(values, expected, atol=1e-5)

    def test_stairs_have_the_same_profile_in_all_four_directions(self) -> None:
        for radius, expected_height in (
            (0.65, 0.18),
            (2.15, 1.08),
            (2.7, 1.08),
            (3.65, 0.90),
            (5.6, 0.0),
            (6.65, 0.18),
            (8.15, 1.08),
            (9.65, 0.90),
            (11.5, 0.0),
        ):
            offsets = [(radius, 0.0), (-radius, 0.0), (0.0, radius), (0.0, -radius)]
            np.testing.assert_allclose(
                _physics_heights_at_xy_offsets("stairs", offsets),
                expected_height,
                atol=1e-5,
            )

    def test_platform_bands_alternate_between_raised_and_flat(self) -> None:
        offsets = [0.0, 1.0, 1.8, 2.6, 3.4, 4.2]
        values = _physics_heights_at_offsets("platforms", offsets)
        np.testing.assert_allclose(values, [0.0, 0.15, 0.0, 0.15, 0.0, 0.15], atol=1e-5)

    def test_traversal_course_has_ordered_flat_stairs_ramp_flat_profile(self) -> None:
        offsets = [0.0, 1.9, 3.4, 4.2, 5.8, 6.7, 8.8, 9.3]
        values = _physics_heights_at_offsets("course", offsets)
        self.assertAlmostEqual(values[0], 0.0, places=5)
        self.assertAlmostEqual(values[1], 0.12, places=4)
        self.assertAlmostEqual(values[2], 0.60, places=4)
        self.assertAlmostEqual(values[3], 0.48, places=4)
        self.assertAlmostEqual(values[4], 0.0, places=4)
        self.assertGreater(values[6], values[5])
        self.assertAlmostEqual(values[7], 2.5 * math.tan(math.radians(8.0)), places=4)

    def test_rough_collision_heights_vary_and_are_finite(self) -> None:
        values = _physics_height_profile("rough")
        self.assertTrue(np.isfinite(values).all())
        self.assertGreater(float(np.var(values)), 1e-5)


class TerrainNetworkRoutingTest(unittest.TestCase):
    def test_flat_raycast_and_fast_flat_match_through_actor_and_backward(self) -> None:
        batch_size = 3
        pelvis_world_z = torch.tensor([[0.82], [0.55], [1.10]])
        ray_root, ray_terrain_actor, ray_terrain = observations_from_clearances(pelvis_world_z.expand(-1, 273), 58, clip=0.5)
        fast_root, fast_terrain_actor, fast_terrain = flat_zero_observations(pelvis_world_z, 273)
        privileged_base = torch.zeros(batch_size, 7)
        ray_privileged = torch.cat((ray_root, privileged_base), dim=-1)
        fast_privileged = torch.cat((fast_root, privileged_base), dim=-1)
        obs_space = gymnasium.spaces.Dict(
            {
                "state": gymnasium.spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float32),
                "privileged_state": gymnasium.spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32),
                "last_action": gymnasium.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
                "history_actor": gymnasium.spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float32),
                "terrain_actor": gymnasium.spaces.Box(-np.inf, np.inf, shape=(273,), dtype=np.float32),
                "terrain_priv": gymnasium.spaces.Box(-np.inf, np.inf, shape=(273,), dtype=np.float32),
            }
        )
        common = {
            "state": torch.randn(batch_size, 12),
            "last_action": torch.randn(batch_size, 4),
            "history_actor": torch.randn(batch_size, 16),
        }
        ray_obs = {
            **common,
            "privileged_state": ray_privileged,
            "terrain_actor": ray_terrain_actor,
            "terrain_priv": ray_terrain,
        }
        fast_obs = {
            **common,
            "privileged_state": fast_privileged,
            "terrain_actor": fast_terrain_actor,
            "terrain_priv": fast_terrain,
        }
        archi = _preset("fb_terrain").model.archi
        actor = archi.actor.build(obs_space, archi.z_dim, action_dim=4).eval()
        backward = archi.b.build(obs_space, archi.z_dim).eval()
        z = torch.randn(batch_size, archi.z_dim)

        with torch.no_grad():
            ray_action = actor(ray_obs, z, std=0.1).mean
            fast_action = actor(fast_obs, z, std=0.1).mean
            ray_z = backward(ray_obs)
            fast_z = backward(fast_obs)
        torch.testing.assert_close(fast_action, ray_action, atol=0.0, rtol=0.0)
        torch.testing.assert_close(fast_z, ray_z, atol=0.0, rtol=0.0)

    def test_canonical_states_are_identical_through_real_backward_path(self) -> None:
        state = torch.zeros(1, 4)
        lower_privileged = _canonical_privileged_state(root_height=0.82, ground_height=0.0)
        upper_privileged = _canonical_privileged_state(root_height=1.42, ground_height=0.6)
        obs_space = gymnasium.spaces.Dict(
            {
                "state": gymnasium.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
                "privileged_state": gymnasium.spaces.Box(
                    -np.inf, np.inf, shape=(lower_privileged.shape[-1],), dtype=np.float32
                ),
            }
        )
        cfg = _preset("fb_terrain").model.archi
        backward = cfg.b.build(obs_space, cfg.z_dim).eval()
        with torch.no_grad():
            lower_z = backward({"state": state, "privileged_state": lower_privileged})
            upper_z = backward({"state": state, "privileged_state": upper_privileged})
        torch.testing.assert_close(lower_z, upper_z, atol=1e-6, rtol=0.0)

    def test_terrain_routing_and_terrain_agnostic_prior(self) -> None:
        archi = _preset("fb_terrain").model.archi
        self.assertEqual(archi.b.input_filter.key, ["state", "privileged_state"])
        self.assertEqual(archi.discriminator.input_filter.key, ["state", "privileged_state"])
        self.assertIn("terrain_actor", archi.actor.input_filter.key)
        self.assertNotIn("terrain_priv", archi.actor.input_filter.key)
        for name in ("f", "critic", "aux_critic"):
            self.assertIn("terrain_priv", getattr(archi, name).input_filter.key)
            self.assertNotIn("terrain_actor", getattr(archi, name).input_filter.key)

    def test_default_fb_path_is_unchanged(self) -> None:
        cfg = _preset("fb")
        archi = cfg.model.archi
        self.assertEqual(archi.actor.input_filter.key, ["state", "last_action", "history_actor"])
        self.assertEqual(archi.f.input_filter.key, ["state", "privileged_state", "last_action", "history_actor"])
        self.assertEqual(archi.critic.input_filter.key, ["state", "privileged_state", "last_action", "history_actor"])
        self.assertNotIn("terrain_priv", cfg.model.obs_normalizer.normalizers)
        self.assertNotIn("terrain_actor", cfg.model.obs_normalizer.normalizers)
        self.assertIn("penalty_undesired_contact", cfg.aux_rewards)
        self.assertIn("penalty_feet_ori", cfg.aux_rewards)
        self.assertNotIn("penalty_body_impact", cfg.aux_rewards)

    def test_terrain_auxiliary_rewards_use_continuous_impact(self) -> None:
        cfg = _preset("fb_terrain")
        self.assertEqual(
            cfg.aux_rewards,
            [
                "penalty_action_rate",
                "limits_dof_pos",
                "penalty_body_impact",
                "penalty_slippage",
                "penalty_ankle_roll",
            ],
        )
        self.assertEqual(
            cfg.aux_rewards_scaling,
            {
                "penalty_action_rate": -0.1,
                "limits_dof_pos": -10.0,
                "penalty_body_impact": -1.0,
                "penalty_slippage": -1.0,
                "penalty_ankle_roll": -1.0,
            },
        )

    def test_terrain_normalizer_is_opt_in(self) -> None:
        self.assertIn("terrain_priv", _preset("fb_terrain").model.obs_normalizer.normalizers)
        self.assertIn("terrain_actor", _preset("fb_terrain").model.obs_normalizer.normalizers)
        self.assertNotIn("terrain_priv", _preset("fb").model.obs_normalizer.normalizers)
        self.assertNotIn("terrain_actor", _preset("fb").model.obs_normalizer.normalizers)

    def test_phase2_uses_fixed_harder_terrain_distribution(self) -> None:
        terrain = OmegaConf.load("humanoidverse/config/terrain/terrain_ufo_v0.yaml").terrain
        self.assertEqual(
            dict(terrain.terrain_mix),
            {"flat": 0.15, "slope": 0.15, "stairs": 0.3, "rough": 0.2, "platforms": 0.2},
        )
        self.assertEqual(list(terrain.stairs.step_height_range), [0.10, 0.18])
        self.assertEqual(list(terrain.rough.amplitude_range), [0.03, 0.05])
        self.assertEqual(float(terrain.slope.max_angle_deg), 8.0)
        self.assertEqual(list(terrain.patch_size), [30.0, 30.0])
        self.assertEqual(int(terrain.stairs.num_steps), 6)
        self.assertEqual(float(terrain.stairs.plateau_width), 1.2)
        cycle_radius = (
            2 * int(terrain.stairs.num_steps) * float(terrain.stairs.step_depth)
            + 2 * float(terrain.stairs.plateau_width)
        )
        available_width = min(float(value) for value in terrain.patch_size) - 2 * float(
            terrain.stairs.border_width
        )
        available_radius = (available_width - float(terrain.stairs.platform_width)) / 2
        num_cycles = int(available_radius // cycle_radius)
        stair_transition_width = float(terrain.stairs.platform_width) + 2 * num_cycles * cycle_radius
        self.assertEqual(num_cycles, 2)
        self.assertAlmostEqual(stair_transition_width, 25.0)
        self.assertLess(stair_transition_width, available_width)
        np.testing.assert_array_equal(
            _proportional_counts(1024, np.asarray(list(terrain.terrain_mix.values()))),
            [154, 154, 306, 205, 205],
        )
        np.testing.assert_array_equal(
            _proportional_counts(8192, np.asarray(list(terrain.terrain_mix.values()))),
            [1229, 1229, 2457, 1639, 1638],
        )
        self.assertFalse(bool(terrain.coverage.fail_on_boundary_violation))

    def test_all_lafan_clips_fit_patch_with_sensor_and_policy_margin(self) -> None:
        report = validate_motion_terrain_coverage(
            "humanoidverse/data/lafan_29dof_10s-clipped.pkl",
            patch_size=(30.0, 30.0),
            sensor_radius=math.hypot(1.6, 0.6),
            policy_margin=2.0,
        )
        self.assertLess(report.required_radius, report.patch_safe_radius)
        self.assertEqual(report.motion_key, "sprint1_subject4_clip1")
        with self.assertRaisesRegex(RuntimeError, "coverage invariant failed"):
            validate_motion_terrain_coverage(
                "humanoidverse/data/lafan_29dof_10s-clipped.pkl",
                patch_size=(20.0, 20.0),
                sensor_radius=math.hypot(1.6, 0.6),
                policy_margin=2.0,
            )


class TerrainReplayAndTransferTest(unittest.TestCase):
    def test_current_and_next_terrain_are_preserved_in_replay(self) -> None:
        current = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        next_value = current + 1.0
        data = {
            "observation": {"terrain_actor": current + 10.0, "terrain_priv": current},
            "next": {"observation": {"terrain_actor": next_value + 10.0, "terrain_priv": next_value}},
            "action": torch.zeros(3, 2),
        }
        buffer = DictBuffer(capacity=8, device="cpu")
        buffer.extend(data)
        stored = buffer.get_full_buffer()
        torch.testing.assert_close(stored["observation"]["terrain_priv"], current)
        torch.testing.assert_close(stored["next"]["observation"]["terrain_priv"], next_value)
        torch.testing.assert_close(stored["observation"]["terrain_actor"], current + 10.0)
        torch.testing.assert_close(stored["next"]["observation"]["terrain_actor"], next_value + 10.0)
        self.assertTrue(torch.isfinite(stored["next"]["observation"]["terrain_priv"]).all())

    def test_exact_same_z_is_reused_for_every_terrain(self) -> None:
        z = torch.arange(3 * 256, dtype=torch.float32).reshape(3, 256)
        terrains = ["flat", "slope", "stairs", "rough", "platforms"]
        clones = clone_same_z_for_terrains(z, terrains)
        checksums = {tensor_checksum(value) for value in clones.values()}
        self.assertEqual(checksums, {tensor_checksum(z)})
        self.assertTrue(all(value.data_ptr() != z.data_ptr() for value in clones.values()))


class TerrainPriorityEvaluationTest(unittest.TestCase):
    def _mixed_env_cfg(self) -> HumanoidVerseMjlabConfig:
        return HumanoidVerseMjlabConfig(
            lafan_tail_path="motions.pkl",
            hydra_overrides=[
                "robot=g1/g1_29dof",
                "terrain=terrain_ufo_v0",
                "terrain.terrain_type=mixed",
                "terrain.seed=7",
            ],
        )

    def test_priority_eval_config_selects_plane_and_explicit_fast_observation(self) -> None:
        mixed = self._mixed_env_cfg()
        flat = make_flat_terrain_priority_eval_config(mixed)
        self.assertIn("terrain.terrain_type=mixed", mixed.hydra_overrides)
        self.assertIn("terrain.terrain_type=plane", flat.hydra_overrides)
        self.assertIn("terrain.terrain_priv.mode=flat_zero", flat.hydra_overrides)
        self.assertIn("terrain=terrain_ufo_v0", flat.hydra_overrides)
        self.assertIn("terrain.seed=7", flat.hydra_overrides)
        changed = [(before, after) for before, after in zip(mixed.hydra_overrides, flat.hydra_overrides) if before != after]
        self.assertEqual(changed, [("terrain.terrain_type=mixed", "terrain.terrain_type=plane")])
        self.assertEqual(flat.hydra_overrides[-1], "terrain.terrain_priv.mode=flat_zero")
        self.assertNotIn("terrain.terrain_priv.mode=flat_zero", mixed.hydra_overrides)

        clearances = torch.full((3, 273), 0.82)
        _root_height, terrain_actor, terrain_priv = observations_from_clearances(clearances, 58, clip=0.5)
        torch.testing.assert_close(terrain_actor, clearances)
        self.assertEqual(tuple(terrain_priv.shape), (3, 273))
        torch.testing.assert_close(terrain_priv, torch.zeros_like(terrain_priv))

    def test_mixed_train_and_flat_priority_component_sets(self) -> None:
        self.assertEqual(
            terrain_component_names("mixed"),
            ("flat", "slope", "stairs", "rough", "platforms"),
        )
        self.assertEqual(terrain_component_names("flat"), ("flat",))

    def test_priority_eval_env_is_lazy_reused_and_closed(self) -> None:
        workspace = Workspace.__new__(Workspace)
        workspace.cfg = SimpleNamespace(
            env=self._mixed_env_cfg(),
            online_parallel_envs=16,
            distributed_sync=False,
        )
        workspace.distributed_rank = 0
        workspace._priority_eval_env = None
        built_env = Mock()
        with patch.object(HumanoidVerseMjlabConfig, "build", return_value=(built_env, {})) as build:
            self.assertIs(workspace._get_priority_eval_env(), built_env)
            self.assertIs(workspace._get_priority_eval_env(), built_env)
        build.assert_called_once_with(num_envs=16)
        workspace._close_priority_eval_env()
        built_env.close.assert_called_once_with()
        self.assertIsNone(workspace._priority_eval_env)

    def test_only_priority_evaluator_uses_fixed_flat_env(self) -> None:
        workspace = Workspace.__new__(Workspace)
        workspace.cfg = SimpleNamespace(prioritization=True, tags={"agent": "fb_terrain"})
        workspace.priorization_eval_name = "priority"
        self.assertTrue(workspace._uses_fixed_flat_priority_eval("priority"))
        self.assertFalse(workspace._uses_fixed_flat_priority_eval("monitoring"))
        workspace.cfg.tags = {"agent": "fb"}
        self.assertFalse(workspace._uses_fixed_flat_priority_eval("priority"))

    def test_eval_routes_only_priority_evaluator_to_flat_env(self) -> None:
        workspace = Workspace.__new__(Workspace)
        workspace.cfg = SimpleNamespace(
            env=self._mixed_env_cfg(),
            prioritization=True,
            tags={"agent": "fb_terrain"},
            use_wandb=False,
        )
        workspace.priorization_eval_name = "priority"
        workspace.train_env = Mock(name="mixed_train_env")
        priority_env = Mock(name="flat_priority_env")
        workspace._get_priority_eval_env = Mock(return_value=priority_env)
        workspace.agent = SimpleNamespace(_model=Mock())
        workspace.eval_loggers = {}
        workspace._write_shared_artifacts = False
        workspace.evaluations = {
            "priority": Mock(run=Mock(return_value=({}, None))),
            "monitoring": Mock(run=Mock(return_value=({}, None))),
        }

        workspace.eval(3_200_000, replay_buffer={})

        self.assertIs(workspace.evaluations["priority"].run.call_args.kwargs["env"], priority_env)
        self.assertIs(workspace.evaluations["monitoring"].run.call_args.kwargs["env"], workspace.train_env)
        workspace._get_priority_eval_env.assert_called_once_with()

    def test_existing_priority_sampling_update_chain_is_unchanged(self) -> None:
        source = inspect.getsource(Workspace.train_online)
        self.assertIn("update_sampling_weight_by_id", source)
        self.assertIn('replay_buffer["expert_slicer"].update_priorities', source)
        self.assertIn('prioritization_mode == "exp"', source)

    def test_emd_rejects_nonfinite_rollout_input(self) -> None:
        rollout = torch.zeros(3, 2)
        rollout[1, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "rollout observation contains non-finite values"):
            emd_numpy(rollout, torch.zeros_like(rollout))

    def test_tracking_metrics_reject_nonfinite_joint_state_with_motion_identity(self) -> None:
        state = torch.zeros(3, 2)
        joint_pos = torch.zeros(3, 2)
        joint_pos[1, 0] = torch.inf
        episode = {
            "observation": {"state": state},
            "tracking_target": {"state": state.clone()},
            "joint_pos": joint_pos,
            "target_joint_pos": torch.zeros_like(joint_pos),
            "motion_id": 17,
            "motion_file": "bad_motion",
        }
        with self.assertRaisesRegex(ValueError, "motion_id=17, motion_file=bad_motion"):
            _calc_metrics(episode)


if __name__ == "__main__":
    unittest.main()
