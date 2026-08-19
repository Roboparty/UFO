from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from humanoidverse.agents.envs.humanoidverse_mjlab import _soft_limit_action_affine
from humanoidverse.train import build_ufo_mjlab_config, parse_args


class SoftLimitActionMappingTest(unittest.TestCase):
    def test_cli_default_preserves_existing_mapping(self) -> None:
        with patch("sys.argv", ["train.py", "--smoke"]):
            args = parse_args()
        self.assertEqual(args.action_mapping, "effort_kp")

    def test_cli_mapping_reaches_environment_config(self) -> None:
        cfg = build_ufo_mjlab_config(
            device="cpu",
            work_dir="/tmp/ufo_action_mapping_test",
            num_envs=1,
            num_env_steps=1,
            seed=1,
            use_wandb=False,
            wandb_run_name=None,
            smoke=True,
            action_mapping="soft_limit_bias",
        )
        self.assertEqual(cfg.env.action_mapping, "soft_limit_bias")

    def test_endpoints_reconstruct_soft_limits(self) -> None:
        soft_limits = torch.tensor([[-1.2, 2.4], [-0.5, 0.7]], dtype=torch.float32)
        default_pos = torch.tensor([[0.3, -0.1]], dtype=torch.float32)
        target_scale = torch.tensor([[0.6, 0.2]], dtype=torch.float32)

        bias, half_range, lower, upper = _soft_limit_action_affine(
            soft_limits,
            default_pos,
            target_scale,
        )

        torch.testing.assert_close(bias - half_range, lower)
        torch.testing.assert_close(bias + half_range, upper)
        torch.testing.assert_close(default_pos[0] + target_scale[0] * lower, soft_limits[:, 0])
        torch.testing.assert_close(default_pos[0] + target_scale[0] * upper, soft_limits[:, 1])

    def test_asymmetric_limits_produce_nonzero_bias(self) -> None:
        soft_limits = torch.tensor([[-0.2, 1.0]], dtype=torch.float32)
        default_pos = torch.tensor([[0.0]], dtype=torch.float32)
        target_scale = torch.tensor([[0.5]], dtype=torch.float32)

        bias, half_range, _, _ = _soft_limit_action_affine(soft_limits, default_pos, target_scale)

        torch.testing.assert_close(bias, torch.tensor([0.8]))
        torch.testing.assert_close(half_range, torch.tensor([1.2]))

    def test_nonpositive_target_scale_is_rejected(self) -> None:
        soft_limits = torch.tensor([[-1.0, 1.0]], dtype=torch.float32)
        default_pos = torch.tensor([[0.0]], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "positive target scales"):
            _soft_limit_action_affine(soft_limits, default_pos, torch.tensor([[0.0]]))

    def test_invalid_limits_are_rejected(self) -> None:
        soft_limits = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
        default_pos = torch.tensor([[0.0]], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "invalid soft joint limits"):
            _soft_limit_action_affine(soft_limits, default_pos, torch.tensor([[1.0]]))


if __name__ == "__main__":
    unittest.main()
