from __future__ import annotations

from types import SimpleNamespace

import torch

from humanoidverse.agents.behavior_context import (
    HEADING_SOURCE_EXACT_TRACKING,
    align_heading_sequence,
    heading_observation,
    repeat_heading_sequence,
)
from humanoidverse.agents.fb.agent import FBAgent, RolloutContextState
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgent
from humanoidverse.agents.presets.fb_depth import build_fb_depth_agent


def _xy(degrees: list[float]) -> torch.Tensor:
    radians = torch.deg2rad(torch.tensor(degrees, dtype=torch.float32))
    return torch.stack((torch.cos(radians), torch.sin(radians)), dim=-1)


def test_invalid_heading_is_exact_zero() -> None:
    current = _xy([15.0, -120.0])
    target = _xy([80.0, 45.0])
    result = heading_observation(current, target, torch.zeros((2, 1), dtype=torch.bool))
    torch.testing.assert_close(result, torch.zeros_like(result), rtol=0.0, atol=0.0)


def test_valid_zero_error_is_identical_to_invalid_context() -> None:
    current = _xy([15.0, -120.0])
    invalid = heading_observation(current, _xy([80.0, 45.0]), torch.zeros((2, 1), dtype=torch.bool))
    valid_zero_error = heading_observation(current, current, torch.ones((2, 1), dtype=torch.bool))
    assert valid_zero_error.shape == (2, 2)
    torch.testing.assert_close(valid_zero_error, invalid, rtol=0.0, atol=0.0)


def test_heading_error_is_zero_centered_and_signed() -> None:
    current = _xy([0.0, 0.0, 0.0])
    target = _xy([90.0, -90.0, 180.0])
    result = heading_observation(current, target, torch.ones((3, 1), dtype=torch.bool))
    expected = torch.tensor([[1.0, 1.0], [1.0, -1.0], [2.0, 0.0]])
    torch.testing.assert_close(result, expected, rtol=0.0, atol=1.0e-6)


def test_heading_vectors_cross_pi_without_discontinuity() -> None:
    reference = _xy([170.0, 179.0, -179.0, -170.0]).unsqueeze(0)
    aligned = align_heading_sequence(reference, _xy([30.0]), torch.tensor([0]))[0]
    step_cos = torch.sum(aligned[:-1] * aligned[1:], dim=-1)
    assert bool((step_cos > 0.98).all())


def test_repeated_turning_heading_does_not_jump_back() -> None:
    reference = _xy([0.0, 45.0, 90.0])
    repeated = repeat_heading_sequence(reference, 2)
    torch.testing.assert_close(repeated[2], repeated[3], atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(repeated[-1], _xy([180.0])[0], atol=1.0e-5, rtol=0.0)


def test_collector_restart_preserves_seeded_context_boundary() -> None:
    agent = FBAgent.__new__(FBAgent)
    agent._model = SimpleNamespace(
        device=torch.device("cpu"),
        sample_z=lambda count, device: torch.zeros(count, 2, device=device),
    )
    agent.cfg = SimpleNamespace(
        model=SimpleNamespace(heading_context_enabled=False),
        train=SimpleNamespace(
            rollout_expert_trajectories=False,
            update_z_every_step=100,
            use_mix_rollout=False,
        ),
    )
    seeded = torch.tensor([[1_000_001], [1_000_001]], dtype=torch.long)

    context = agent.advance_rollout_context(
        RolloutContextState(context_id=seeded),
        step_count=torch.ones(2, dtype=torch.long),
    )

    torch.testing.assert_close(context.context_id, seeded)


def test_fb_depth_routes_heading_but_keeps_b_and_d_agnostic() -> None:
    cfg = build_fb_depth_agent(device="cpu", compile=False)
    assert cfg.train.discriminator_loss == "lsgan"
    assert cfg.train.discriminator_reward == "amp"
    assert cfg.model.heading_context_enabled
    assert not cfg.model.heading_critic_enabled
    assert cfg.train.reg_coeff_heading == 0.0
    assert "heading" in cfg.model.archi.actor.input_filter.key
    assert "heading" in cfg.model.archi.f.input_filter.key
    assert "heading" in cfg.model.archi.critic.input_filter.key
    assert "heading" in cfg.model.archi.aux_critic.input_filter.key
    assert "heading" in cfg.model.archi.heading_critic.input_filter.key
    assert "heading" not in cfg.model.archi.b.input_filter.key
    assert "heading" not in cfg.model.archi.discriminator.input_filter.key
    assert cfg.model.obs_normalizer.normalizers["heading"].name == "IdentityNormalizerConfig"
    assert cfg.model.archi.f.hidden_dim == 2048
    assert cfg.model.archi.f.hidden_layers == 6
    assert cfg.model.archi.critic.hidden_dim == 1024
    assert cfg.model.archi.critic.hidden_layers == 4
    assert cfg.model.archi.critic.num_parallel == 2
    assert cfg.model.archi.aux_critic.hidden_dim == 2048
    assert cfg.model.archi.aux_critic.hidden_layers == 4
    assert cfg.model.archi.aux_critic.num_parallel == 2

    qh_cfg = build_fb_depth_agent(
        device="cpu", compile=False, heading_reg_coeff=0.01
    )
    assert qh_cfg.model.heading_context_enabled
    assert qh_cfg.model.heading_critic_enabled
    assert qh_cfg.model.archi.heading_critic.hidden_dim == 1024
    assert qh_cfg.model.archi.heading_critic.hidden_layers == 3
    assert qh_cfg.model.archi.heading_critic.num_parallel == 2
    assert qh_cfg.model.archi.aux_critic.hidden_dim == 2048
    assert qh_cfg.model.archi.aux_critic.hidden_layers == 4


def test_qh_continuation_stops_at_tracking_context_boundary() -> None:
    agent = FBAgent.__new__(FBAgent)
    agent._model = SimpleNamespace(device=torch.device("cpu"))
    agent.cfg = SimpleNamespace(
        train=SimpleNamespace(
            rollout_expert_trajectories_length=4,
            update_z_every_step=3,
        )
    )
    state = RolloutContextState(
        heading_valid=torch.tensor([[True], [True], [False]]),
        expert_env_ids=torch.tensor([0, 1]),
    )
    result = agent.rollout_heading_context_continues(
        state,
        next_step_count=torch.tensor([[1], [4], [2]]),
        done=torch.tensor([False, False, False]),
    )
    assert result.tolist() == [[True], [False], [False]]


def test_qh_tracking_bootstrap_uses_next_time_varying_z() -> None:
    agent = FBAgent.__new__(FBAgent)
    agent._model = SimpleNamespace(device=torch.device("cpu"))
    agent.cfg = SimpleNamespace(
        train=SimpleNamespace(rollout_expert_trajectories_length=4)
    )
    tracking_z = torch.tensor(
        [[[0.0], [1.0], [2.0], [3.0]], [[10.0], [11.0], [12.0], [13.0]]]
    )
    state = RolloutContextState(
        z=torch.tensor([[1.0], [12.0], [99.0]]),
        expert_env_ids=torch.tensor([0, 1]),
        tracking_z=tracking_z,
    )
    next_z = agent.next_rollout_z(state, torch.tensor([1, 2, 7]))
    torch.testing.assert_close(next_z, torch.tensor([[2.0], [13.0], [99.0]]))


def test_expert_relabel_heading_is_atomic_and_goal_random_are_invalid() -> None:
    agent = FBcprAuxAgent.__new__(FBcprAuxAgent)
    agent._model = SimpleNamespace(device=torch.device("cpu"))

    def sample_components(_goal, expert_z):
        source = torch.tensor([[1], [0], [2]])
        permutation = torch.tensor([2, 0, 1])
        return expert_z.clone(), source, permutation

    agent.sample_mixed_context_components = sample_components
    root = _xy([10.0, 20.0, 30.0])
    next_root = _xy([12.0, 22.0, 32.0])
    expert = _xy([0.0, 0.0, 0.0])
    expert_next = _xy([15.0, 25.0, 35.0])
    context = agent._sample_mixed_behavior_context(
        train_goal=torch.zeros(3, 1),
        expert_z=torch.randn(3, 4),
        root_heading_xy=root,
        next_root_heading_xy=next_root,
        expert_heading_xy=expert,
        expert_next_heading_xy=expert_next,
    )
    assert context.heading_valid.tolist() == [[True], [False], [False]]
    assert context.source_type[:, 0].tolist() == [HEADING_SOURCE_EXACT_TRACKING, 0, 0]
    torch.testing.assert_close(context.heading[1:], torch.zeros_like(context.heading[1:]), atol=0.0, rtol=0.0)
    torch.testing.assert_close(context.next_heading[1:], torch.zeros_like(context.next_heading[1:]), atol=0.0, rtol=0.0)
