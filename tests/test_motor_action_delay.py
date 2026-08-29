from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from humanoidverse.agents.envs.humanoidverse_mjlab import HumanoidVerseMjlabCore


def _delay_core(*, num_envs: int, num_dof: int) -> HumanoidVerseMjlabCore:
    core = object.__new__(HumanoidVerseMjlabCore)
    core.device = "cpu"
    core._ctrl_delay_enabled = True
    core._ctrl_delay_min_step = 0
    core._ctrl_delay_max_step = 2
    core._ctrl_delay_steps = torch.zeros(num_envs, dtype=torch.long)
    core._ctrl_delay_env_indices = torch.arange(num_envs, dtype=torch.long)
    core._ctrl_delay_buffer = torch.zeros(3, num_envs, num_dof)
    return core


def test_motor_action_delay_is_enabled_for_zero_to_two_policy_steps() -> None:
    config_path = Path(__file__).parents[1] / "humanoidverse/config/domain_rand/domain_rand.yaml"
    config = OmegaConf.load(config_path).domain_rand
    assert config.randomize_ctrl_delay is True
    assert list(config.ctrl_delay_step_range) == [0, 2]


def test_motor_action_delay_fifo_is_independent_per_environment() -> None:
    core = _delay_core(num_envs=3, num_dof=1)
    core._ctrl_delay_steps[:] = torch.tensor([0, 1, 2])

    first = core._apply_ctrl_delay(torch.full((3, 1), 1.0))
    second = core._apply_ctrl_delay(torch.full((3, 1), 2.0))
    third = core._apply_ctrl_delay(torch.full((3, 1), 3.0))

    assert torch.equal(first[:, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(second[:, 0], torch.tensor([2.0, 1.0, 0.0]))
    assert torch.equal(third[:, 0], torch.tensor([3.0, 2.0, 1.0]))


def test_motor_action_delay_resamples_and_clears_only_reset_environments() -> None:
    core = _delay_core(num_envs=6, num_dof=2)
    core._ctrl_delay_buffer.fill_(1.0)
    env_ids = torch.tensor([1, 3, 5], dtype=torch.long)

    torch.manual_seed(17)
    core._sample_ctrl_delay(env_ids)

    sampled = core._ctrl_delay_steps[env_ids]
    assert torch.all((sampled >= 0) & (sampled <= 2))
    assert torch.count_nonzero(core._ctrl_delay_buffer[:, env_ids]) == 0
    assert torch.all(core._ctrl_delay_buffer[:, torch.tensor([0, 2, 4])] == 1.0)
