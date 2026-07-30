from __future__ import annotations

import pytest
import torch

from humanoidverse.agents.evaluations.humanoidverse_mjlab import _calc_metrics, compute_joint_pos_metrics, emd_numpy


def _episode(
    num_dofs: int,
    *,
    time_steps: int = 4,
    observation_width: int | None = None,
    target_width: int | None = None,
    target_time_steps: int | None = None,
):
    state_width = 2 * num_dofs + 6
    observation_width = state_width if observation_width is None else observation_width
    target_width = state_width if target_width is None else target_width
    target_time_steps = time_steps if target_time_steps is None else target_time_steps
    return {
        "observation": {"state": torch.zeros(time_steps, observation_width)},
        "tracking_target": {"state": torch.zeros(target_time_steps, target_width)},
        "joint_pos": torch.zeros(time_steps, num_dofs),
        "target_joint_pos": torch.zeros(time_steps, num_dofs),
        "motion_id": 0,
        "motion_file": "test_motion",
    }


def test_obs_state_metrics_include_all_29_g1_joint_positions() -> None:
    ep = _episode(29)
    ep["observation"]["state"][:, 23:29] = 1.0

    metrics = _calc_metrics(ep)["test_motion"]

    assert metrics["obs_state_distance"] > 0
    assert metrics["obs_state_emd"] > 0
    assert metrics["mpjpe_l"] == 0.0
    assert metrics["vel_dist"] == 0.0

    legacy_observation = ep["observation"]["state"][:, :23]
    legacy_target = ep["tracking_target"]["state"][:, :23]
    assert torch.norm(legacy_observation - legacy_target, dim=-1).mean().item() == 0.0
    assert emd_numpy(legacy_observation, legacy_target)["emd"] == 0.0


def test_joint_metrics_difference_along_time() -> None:
    time = torch.arange(4, dtype=torch.float32)
    joint_pos = time.square().unsqueeze(-1).repeat(1, 3)
    target_joint_pos = torch.zeros_like(joint_pos)

    metrics = compute_joint_pos_metrics(joint_pos, target_joint_pos)
    expected_vel = torch.norm(
        torch.diff(joint_pos, dim=0) - torch.diff(target_joint_pos, dim=0),
        dim=-1,
    ).mean() * 1000
    expected_accel = torch.norm(
        torch.diff(joint_pos, n=2, dim=0) - torch.diff(target_joint_pos, n=2, dim=0),
        dim=-1,
    ).mean() * 100

    torch.testing.assert_close(metrics["vel_dist"], expected_vel)
    torch.testing.assert_close(metrics["accel_dist"], expected_accel)

    legacy_vel = joint_pos[:, 1:] - joint_pos[:, :-1]
    legacy_accel = joint_pos[:, :-2] - 2 * joint_pos[:, 1:-1] + joint_pos[:, 2:]
    torch.testing.assert_close(legacy_vel, torch.zeros_like(legacy_vel))
    torch.testing.assert_close(legacy_accel, torch.zeros_like(legacy_accel))


def test_static_joint_offsets_are_not_temporal_motion() -> None:
    base_pose = torch.arange(4, dtype=torch.float32)
    joint_pos = base_pose.unsqueeze(0).repeat(4, 1)
    target_joint_pos = torch.zeros_like(joint_pos)

    metrics = compute_joint_pos_metrics(joint_pos, target_joint_pos)

    torch.testing.assert_close(metrics["vel_dist"], torch.tensor(0.0))
    torch.testing.assert_close(metrics["accel_dist"], torch.tensor(0.0))


def test_joint_metrics_preserve_non_temporal_definitions() -> None:
    base_pose = torch.arange(4, dtype=torch.float32)
    joint_pos = base_pose.unsqueeze(0).repeat(4, 1)
    target_joint_pos = torch.zeros_like(joint_pos)
    distance = torch.norm(joint_pos - target_joint_pos, dim=-1)
    in_bounds = distance <= 2.0
    out_bounds = distance > 4.0
    expected_proximity = (in_bounds + ((4.0 - distance) / 2.0) * (~in_bounds) * (~out_bounds)).mean()

    metrics = compute_joint_pos_metrics(joint_pos, target_joint_pos)

    torch.testing.assert_close(metrics["mpjpe_l"], distance.mean() * 1000)
    torch.testing.assert_close(metrics["distance"], distance.mean())
    torch.testing.assert_close(metrics["proximity"], expected_proximity)
    assert metrics["emd"] == pytest.approx(distance.mean().item())


@pytest.mark.parametrize("num_dofs", [23, 29, 30])
def test_obs_state_metrics_support_dynamic_robot_dofs(num_dofs: int) -> None:
    ep = _episode(num_dofs)
    ep["observation"]["state"][:, num_dofs - 1] = 1.0

    metrics = _calc_metrics(ep)["test_motion"]

    assert metrics["obs_state_distance"] > 0
    assert metrics["obs_state_emd"] > 0


@pytest.mark.parametrize(
    ("episode", "expected_error"),
    [
        (_episode(0), "num_dofs must be greater than zero"),
        (_episode(29, observation_width=28), "observation state feature dimension is smaller than num_dofs"),
        (_episode(29, target_width=28), "tracking target state feature dimension is smaller than num_dofs"),
        (_episode(29, target_time_steps=5), "observation and tracking target time dimensions must match"),
        (
            {**_episode(3), "joint_pos": torch.zeros(4, 2)},
            "joint_pos and target_joint_pos shapes must match",
        ),
        ({**_episode(3), "joint_pos": torch.zeros(4)}, "joint_pos must be two-dimensional"),
        ({**_episode(3), "joint_pos": torch.zeros(4, 1, 3)}, "joint_pos must be two-dimensional"),
        (
            {**_episode(3), "target_joint_pos": torch.zeros(4, 1, 3)},
            "target_joint_pos must be two-dimensional",
        ),
        (_episode(3, time_steps=1), "joint_pos must contain at least 3 time steps"),
        (_episode(3, time_steps=2), "joint_pos must contain at least 3 time steps"),
        (
            {**_episode(3), "observation": {"state": torch.zeros(4, 1, 12)}},
            "observation state must be two-dimensional",
        ),
        (
            {**_episode(3), "tracking_target": {"state": torch.zeros(4, 1, 12)}},
            "tracking target state must be two-dimensional",
        ),
        (
            {
                **_episode(3),
                "observation": {"state": torch.zeros(5, 12)},
                "tracking_target": {"state": torch.zeros(5, 12)},
            },
            "observation state and joint_pos time dimensions must match",
        ),
    ],
)
def test_obs_state_metrics_validate_shapes(episode, expected_error: str) -> None:
    with pytest.raises(ValueError) as exc_info:
        _calc_metrics(episode)

    message = str(exc_info.value)
    assert expected_error in message
    assert "observation_state.shape=" in message
    assert "tracking_target_state.shape=" in message
    assert "joint_pos.shape=" in message
    assert "target_joint_pos.shape=" in message
    assert "num_dofs=" in message
