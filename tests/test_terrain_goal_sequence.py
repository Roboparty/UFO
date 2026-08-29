from __future__ import annotations

import pytest
import torch

from humanoidverse.terrain_transfer_inference import _compose_goal_and_forward_z, _expand_goal_sequence


def test_goal_sequence_switches_and_wraps_at_fixed_interval() -> None:
    goals = torch.tensor([[1.0], [2.0], [3.0]])

    sequence = _expand_goal_sequence(goals, episode_length=8, switch_interval=2)

    assert sequence[:, 0].tolist() == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0]


@pytest.mark.parametrize("episode_length,switch_interval", [(0, 1), (1, 0)])
def test_goal_sequence_rejects_non_positive_lengths(episode_length: int, switch_interval: int) -> None:
    with pytest.raises(ValueError):
        _expand_goal_sequence(
            torch.ones((1, 2)),
            episode_length=episode_length,
            switch_interval=switch_interval,
        )


def test_goal_forward_composition_broadcasts_and_projects_each_frame() -> None:
    class UnitNormModel:
        @staticmethod
        def project_z(z: torch.Tensor) -> torch.Tensor:
            return 4.0 * torch.nn.functional.normalize(z, dim=-1)

    goals = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    forward = torch.tensor([[1.0, 1.0]])

    composed = _compose_goal_and_forward_z(UnitNormModel(), goals, forward, weight=0.5)

    expected = 4.0 * torch.nn.functional.normalize(goals + 0.5 * forward, dim=-1)
    torch.testing.assert_close(composed, expected)
    torch.testing.assert_close(torch.linalg.vector_norm(composed, dim=-1), torch.full((2,), 4.0))
