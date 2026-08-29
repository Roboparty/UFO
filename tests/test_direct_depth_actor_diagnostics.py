import math

import numpy as np
import torch

from humanoidverse.direct_depth_actor_diagnostics import (
    _gradient_pair_metrics,
    _root_state_from_qpos_qvel,
    _stairs_column,
    counterfactual_action_metrics,
)


def test_gradient_pair_metrics_reports_conflict_and_projection() -> None:
    first = (torch.tensor([2.0, 0.0]), torch.tensor([0.0]))
    second = (torch.tensor([-1.0, 0.0]), torch.tensor([0.0]))

    metrics = _gradient_pair_metrics(first, second, (0, 1))

    assert math.isclose(metrics["first_norm"], 2.0)
    assert math.isclose(metrics["second_norm"], 1.0)
    assert math.isclose(metrics["cosine"], -1.0)
    assert math.isclose(metrics["first_projection_on_second"], -2.0)


def test_counterfactual_action_metrics_uses_actor_std_scale() -> None:
    stairs = torch.tensor([[0.1, -0.1] + [0.0] * 27])
    flat = torch.zeros_like(stairs)

    metrics = counterfactual_action_metrics(stairs, flat, actor_std=0.05)

    expected_rms = math.sqrt(0.02 / 29.0)
    assert math.isclose(metrics["rms_mean"], expected_rms, rel_tol=1.0e-6)
    assert math.isclose(metrics["rms_in_actor_std"], expected_rms / 0.05, rel_tol=1.0e-6)
    assert metrics["samples"] == 1
    assert metrics["action_dim"] == 29


def test_root_state_converts_wxyz_to_xyzw_and_rotates_angular_velocity() -> None:
    qpos = torch.zeros((1, 36))
    qvel = torch.zeros((1, 35))
    qpos[:, :3] = torch.tensor([[1.0, 2.0, 3.0]])
    qpos[:, 3] = 1.0
    qvel[:, :6] = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])

    state = _root_state_from_qpos_qvel(qpos, qvel)

    assert torch.equal(state["root_states"][:, :3], qpos[:, :3])
    assert torch.equal(state["root_states"][:, 3:7], torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
    assert torch.allclose(state["root_states"][:, 7:13], qvel[:, :6])
    assert state["dof_states"].shape == (1, 29, 2)


def test_stairs_column_matches_frozen_rp1_layout() -> None:
    y = np.asarray([-7.49, -2.51, -2.49, 2.49])

    assert np.array_equal(_stairs_column(y), np.asarray([2, 2, 3, 3]))
