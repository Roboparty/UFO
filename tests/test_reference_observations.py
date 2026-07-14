from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from humanoidverse.agents.evaluations.humanoidverse_mjlab import (
    get_backward_observation as get_mjlab_eval_backward_observation,
)
from humanoidverse.utils.reference_observations import reference_base_ang_vel
from humanoidverse.utils.torch_utils import quat_rotate_inverse


REPO_ROOT = Path(__file__).resolve().parents[1]


class ReferenceObservationsTest(unittest.TestCase):
    def test_reference_base_ang_vel_matches_simulator_contract(self) -> None:
        env = SimpleNamespace(config=SimpleNamespace(obs=SimpleNamespace(obs_scales={"base_ang_vel": 0.25})))
        sqrt_half = 2.0**-0.5
        base_quat = torch.tensor(
            [
                [0.0, 0.0, sqrt_half, sqrt_half],
                [sqrt_half, 0.0, 0.0, sqrt_half],
            ],
            dtype=torch.float32,
        )
        world_ang_vel = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [-2.0, 4.0, 1.0],
            ],
            dtype=torch.float32,
        )

        expected = quat_rotate_inverse(base_quat, world_ang_vel, w_last=True) * 0.25
        actual = reference_base_ang_vel(env, base_quat, world_ang_vel)

        self.assertTrue(torch.allclose(actual, expected))

    def test_reference_base_ang_vel_defaults_to_unit_scale(self) -> None:
        env = SimpleNamespace(config=SimpleNamespace(obs=SimpleNamespace(obs_scales={})))
        base_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        world_ang_vel = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float32)

        actual = reference_base_ang_vel(env, base_quat, world_ang_vel)

        self.assertTrue(torch.equal(actual, world_ang_vel))

    def test_mjlab_eval_backward_observation_requires_obs_filter(self) -> None:
        env = SimpleNamespace(config=SimpleNamespace(obs=SimpleNamespace(use_obs_filter=False)))

        with self.assertRaisesRegex(ValueError, "requires obs.use_obs_filter=True"):
            get_mjlab_eval_backward_observation(env, motion_id=0, include_last_action=False)

    def test_reference_paths_do_not_directly_assign_world_base_ang_vel(self) -> None:
        paths = [
            "humanoidverse/agents/envs/expert_motion_loader.py",
            "humanoidverse/utils/helpers.py",
            "humanoidverse/agents/evaluations/humanoidverse_mjlab.py",
        ]
        for relative_path in paths:
            with self.subTest(path=relative_path):
                tree = ast.parse((REPO_ROOT / relative_path).read_text())
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Assign):
                        continue
                    assigns_ref_ang_vel = any(isinstance(target, ast.Name) and target.id == "ref_ang_vel" for target in node.targets)
                    if assigns_ref_ang_vel:
                        self.assertNotEqual(ast.unparse(node.value), "ref_body_angular_vels[:, 0]", relative_path)
