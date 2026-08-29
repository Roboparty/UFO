from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from humanoidverse.agents.envs.humanoidverse_mjlab import (
    _random_yaw_quaternions,
    rotate_root_motion_by_yaw,
)
from humanoidverse.agents.presets.fb import build_fb_agent
from humanoidverse.agents.presets.fb_depth import build_fb_depth_agent
from humanoidverse.agents.presets.fb_terrain import build_fb_terrain_agent


def _yaw_quaternion(angle: torch.Tensor) -> torch.Tensor:
    half = 0.5 * angle
    return torch.stack(
        (torch.zeros_like(half), torch.zeros_like(half), torch.sin(half), torch.cos(half)),
        dim=-1,
    )


def test_reset_yaw_rotates_pose_linear_velocity_and_angular_velocity_together() -> None:
    yaw_rotation = _yaw_quaternion(torch.tensor([torch.pi / 2.0]))
    identity = _yaw_quaternion(torch.zeros(1))
    linear_velocity = torch.tensor([[1.0, 0.0, 2.0]])
    angular_velocity = torch.tensor([[0.0, 1.0, 3.0]])

    rotation, rotated_linear, rotated_angular = rotate_root_motion_by_yaw(
        identity,
        linear_velocity,
        angular_velocity,
        yaw_rotation,
    )

    torch.testing.assert_close(rotation, yaw_rotation, atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(rotated_linear, torch.tensor([[0.0, 1.0, 2.0]]), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(rotated_angular, torch.tensor([[-1.0, 0.0, 3.0]]), atol=1.0e-6, rtol=0.0)


def test_reset_yaw_sampler_is_full_range_and_yaw_only() -> None:
    torch.manual_seed(123)
    rotations = _random_yaw_quaternions(4096, "cpu")

    torch.testing.assert_close(rotations[:, :2], torch.zeros_like(rotations[:, :2]))
    torch.testing.assert_close(torch.linalg.vector_norm(rotations, dim=-1), torch.ones(4096))
    assert torch.any(rotations[:, 2] > 0.95)
    assert torch.any(rotations[:, 2] < -0.95)


@pytest.mark.parametrize("builder", [build_fb_agent, build_fb_terrain_agent, build_fb_depth_agent])
def test_fb_training_does_not_consume_hidden_reference_heading_reward(builder) -> None:
    cfg = builder(device="cpu", compile=False)

    assert "heading_reference_alignment" not in cfg.aux_rewards
    assert "heading_reference_alignment" not in cfg.aux_rewards_scaling


def test_environment_does_not_enable_hidden_reference_heading_reward() -> None:
    config_path = Path(__file__).parents[1] / "humanoidverse/config/rewards/reward_bfm_zero.yaml"
    rewards = OmegaConf.load(config_path).rewards

    assert "heading_reference_alignment" not in rewards.reward_scales
