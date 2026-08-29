from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import gymnasium
import numpy as np
import pytest
import torch

from humanoidverse.agents.buffers.trajectory import TrajectoryDictBufferMultiDim
from humanoidverse.agents.envs.humanoidverse_mjlab import HumanoidVerseMjlabConfig
from humanoidverse.agents.evaluations.same_z_terrain import (
    SameZTerrainEvaluationConfig,
    _rotate_reference_to_course,
    make_same_z_terrain_eval_config,
)
from humanoidverse.agents.fb.agent import FBAgent, RolloutContextState
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgent, prior_transition_discount
from humanoidverse.agents.nn_models import eval_mode
from humanoidverse.agents.normalizers import BatchNormNormalizerConfig, ObsNormalizerConfig
from humanoidverse.perception.instinct_direct_depth import RP1DirectDepthConfig, RP1DirectDepthRuntime
from humanoidverse.terrain_transfer import tensor_checksum
from humanoidverse.terrains.rp1_simple import RP1_TERRAIN_COMPONENT_NAMES
from humanoidverse.training.workspace import (
    Workspace,
    _assert_canonical_plane_terrain_priv,
    clone_motion_lib_for_collector,
    make_canonical_plane_training_config,
)


def _terrain_env_cfg() -> HumanoidVerseMjlabConfig:
    return HumanoidVerseMjlabConfig(
        lafan_tail_path="motions.pkl",
        disable_obs_noise=False,
        disable_domain_randomization=False,
        seed=4728,
        hydra_overrides=[
            "robot=g1/g1_29dof",
            "terrain=terrain_ufo_v0",
            "terrain.terrain_type=rp1_simple",
            "terrain.seed=4728",
            "terrain.direct_depth.enabled=true",
        ],
    )


def test_canonical_plane_config_changes_geometry_only() -> None:
    main = _terrain_env_cfg()
    plane = make_canonical_plane_training_config(main)
    assert "terrain.terrain_type=rp1_simple" in main.hydra_overrides
    assert "terrain.terrain_type=plane" in plane.hydra_overrides
    assert "terrain.terrain_priv.mode=flat_zero" in plane.hydra_overrides
    assert "terrain.direct_depth.enabled=true" in plane.hydra_overrides
    assert plane.disable_obs_noise is main.disable_obs_noise
    assert plane.disable_domain_randomization is main.disable_domain_randomization
    assert plane.evaluation_fast_path is False
    assert plane.fixed_direct_depth_delay_frames is None
    assert plane.seed == main.seed


def test_plane_motion_lib_has_local_sampling_state_and_shared_read_only_fk() -> None:
    class _MotionLib:
        def _refresh_sampling_batch_prob(self) -> None:
            self._sampling_batch_prob = self._sampling_prob[self._curr_motion_ids].clone()

    main = _MotionLib()
    main.num_envs = 1024
    main._num_motions = 3
    main._num_unique_motions = 3
    main._curr_motion_ids = torch.arange(3)
    main.curr_motion_keys = ["a", "b", "c"]
    main._termination_history = torch.zeros(3)
    main._success_rate = torch.zeros(3)
    main._sampling_history = torch.zeros(3)
    main._sampling_prob = torch.full((3,), 1.0 / 3.0)
    main._sampling_batch_prob = main._sampling_prob.clone()
    main.gts = torch.randn(20, 3)

    plane = clone_motion_lib_for_collector(main, num_envs=128)
    assert plane is not main
    assert plane.num_envs == 128
    assert plane.gts.data_ptr() == main.gts.data_ptr()
    for field in (
        "_curr_motion_ids",
        "_termination_history",
        "_success_rate",
        "_sampling_history",
        "_sampling_prob",
        "_sampling_batch_prob",
    ):
        assert getattr(plane, field).data_ptr() != getattr(main, field).data_ptr()
    plane._sampling_prob[0] = 0.9
    plane.curr_motion_keys[0] = "changed"
    assert main._sampling_prob[0].item() == pytest.approx(1.0 / 3.0)
    assert main.curr_motion_keys[0] == "a"


def test_plane_terrain_priv_is_raw_zero_and_shared_normalization_is_constant() -> None:
    space = gymnasium.spaces.Dict(
        {
            "state": gymnasium.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32),
            "terrain_priv": gymnasium.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
        }
    )
    normalizer = ObsNormalizerConfig(
        normalizers={
            "state": BatchNormNormalizerConfig(),
            "terrain_priv": BatchNormNormalizerConfig(),
        }
    ).build(space)
    main = {
        "state": torch.randn(32, 2),
        "terrain_priv": torch.randn(32, 3) + torch.tensor([1.0, -2.0, 0.5]),
    }
    normalizer(main)
    before = {key: value.clone() for key, value in normalizer.state_dict().items()}
    raw_plane = {
        "state": torch.randn(16, 2),
        "terrain_priv": torch.zeros(16, 3),
    }
    _assert_canonical_plane_terrain_priv(raw_plane, label="test.plane")
    with torch.no_grad(), eval_mode(normalizer):
        normalized_plane = normalizer(raw_plane)
    after = normalizer.state_dict()
    assert all(torch.equal(before[key], after[key]) for key in before)
    assert normalized_plane["terrain_priv"].var(dim=0, unbiased=False).max().item() == 0.0


def test_rollout_context_is_collector_local() -> None:
    class _Model:
        device = "cpu"

        def __init__(self) -> None:
            self.counter = 0

        def sample_z(self, count: int, *, device: str) -> torch.Tensor:
            self.counter += 1
            return torch.full((count, 2), float(self.counter), device=device)

    agent = FBAgent.__new__(FBAgent)
    agent._model = _Model()
    agent.cfg = SimpleNamespace(
        train=SimpleNamespace(
            update_z_every_step=100,
            use_mix_rollout=False,
            rollout_expert_trajectories=False,
        )
    )
    agent.z_buffer = SimpleNamespace(empty=lambda: True)
    main = agent.advance_rollout_context(RolloutContextState(), torch.ones(3, dtype=torch.long))
    main_snapshot = main.z.clone()
    plane = agent.advance_rollout_context(RolloutContextState(), torch.ones(2, dtype=torch.long))
    torch.testing.assert_close(main.z, main_snapshot)
    assert main.z.shape == (3, 2)
    assert plane.z.shape == (2, 2)
    assert not torch.equal(main.z[:2], plane.z)
    assert not hasattr(agent, "tracking_z")


def test_plane_relabel_does_not_write_shared_z_buffer() -> None:
    class _Recorder:
        def __init__(self) -> None:
            self.values: list[torch.Tensor] = []

        def add(self, value: torch.Tensor) -> None:
            self.values.append(value.clone())

    agent = FBcprAuxAgent.__new__(FBcprAuxAgent)
    agent._model = SimpleNamespace(device="cpu")
    agent.cfg = SimpleNamespace(train=SimpleNamespace(batch_size=2, relabel_ratio=1.0))
    agent.z_buffer = _Recorder()
    calls = iter((torch.full((2, 3), 11.0), torch.full((2, 3), 22.0)))
    agent.sample_mixed_z = lambda **_kwargs: next(calls)
    main_z, prior_z = agent._relabel_main_and_prior_z(
        main_next_obs=torch.zeros(2, 1),
        prior_next_obs=torch.zeros(2, 1),
        expert_z=torch.zeros(2, 3),
        main_rollout_z=torch.zeros(2, 3),
        prior_rollout_z=torch.zeros(2, 3),
    )
    assert len(agent.z_buffer.values) == 1
    torch.testing.assert_close(agent.z_buffer.values[0], torch.full((2, 3), 11.0))
    torch.testing.assert_close(main_z, torch.full((2, 3), 11.0))
    torch.testing.assert_close(prior_z, torch.full((2, 3), 22.0))


def test_prior_transition_outcomes_align_with_action_and_never_bootstrap_done() -> None:
    batch = {
        "transition_terminated": torch.tensor([[False], [True], [False]]),
        "transition_truncated": torch.tensor([[False], [False], [True]]),
        "next": {
            # Deliberately contradictory legacy values prove the explicit
            # current-action outcomes are the source of truth.
            "terminated": torch.tensor([[True], [False], [False]]),
        },
    }
    discount = prior_transition_discount(batch, gamma=0.99, device="cpu")
    torch.testing.assert_close(discount, torch.tensor([[0.99], [0.0], [0.0]]))


def test_prior_compact_replay_preserves_current_outcomes_and_depth_contract() -> None:
    cfg = RP1DirectDepthConfig()
    buffer = TrajectoryDictBufferMultiDim(
        capacity=8,
        device="cpu",
        n_dim=2,
        end_key="episode_boundary",
        output_key_t=[
            "observation",
            "action",
            "z",
            "transition_terminated",
            "transition_truncated",
        ],
        output_key_tp1=["observation"],
        compact_depth_history=True,
        depth_history_offsets=cfg.sampled_ages,
    )
    steps = 4
    newest = torch.arange(steps, dtype=torch.uint8).reshape(steps, 1, 1, 1, 1)
    buffer.extend(
        {
            "observation": {
                "state": torch.arange(steps, dtype=torch.float32).reshape(steps, 1, 1),
                "depth_image": newest.expand(steps, 1, 8, 36, 32).clone(),
            },
            "action": torch.zeros(steps, 1, 2),
            "z": torch.zeros(steps, 1, 3),
            "episode_boundary": torch.tensor([False, False, False, True]).reshape(steps, 1, 1),
            "transition_terminated": torch.tensor([False, False, True, False]).reshape(steps, 1, 1),
            "transition_truncated": torch.zeros(steps, 1, 1, dtype=torch.bool),
        }
    )
    transitions = buffer.get_full_buffer()
    assert transitions["observation"]["depth_image"].shape[-3:] == (8, 36, 32)
    assert transitions["transition_terminated"].reshape(-1).tolist() == [False, False, True]
    discount = prior_transition_discount(transitions, gamma=0.99, device="cpu")
    torch.testing.assert_close(discount.reshape(-1), torch.tensor([0.99, 0.99, 0.0]))


def test_prior_reset_marks_an_administrative_episode_boundary() -> None:
    workspace = Workspace.__new__(Workspace)
    workspace.cfg = SimpleNamespace(prior_plane_envs=3)
    workspace.prior_env = SimpleNamespace(
        reset=lambda: (
            {"terrain_priv": np.zeros((3, 4), dtype=np.float32)},
            {},
        )
    )
    state = workspace._reset_prior_collector()
    assert state.terminated.tolist() == [True, True, True]
    assert state.truncated.tolist() == [False, False, False]
    assert state.rollout.z is None


def test_fixed_depth_delay_zero_is_deterministic_without_changing_training_contract() -> None:
    cfg = RP1DirectDepthConfig()
    training = RP1DirectDepthRuntime(8, "cpu", cfg, enable_noise=False)
    deterministic = RP1DirectDepthRuntime(
        8,
        "cpu",
        cfg,
        enable_noise=False,
        fixed_delay_frames=0,
    )
    frame = torch.zeros((8, cfg.output_height, cfg.output_width), dtype=torch.uint8)
    training.current_frame = lambda _sensor: frame  # type: ignore[method-assign]
    deterministic.current_frame = lambda _sensor: frame  # type: ignore[method-assign]
    torch.manual_seed(123)
    training.reset_from_sensor(object(), torch.arange(8))
    deterministic.reset_from_sensor(object(), torch.arange(8))
    assert set(training.delay_frames.tolist()).issubset({0, 1})
    assert deterministic.delay_frames.tolist() == [0] * 8
    assert cfg.delayed_frame_ranges == (0, 1)


def test_same_z_evaluator_uses_exact_seven_hashes_and_fixed_zero_depth_delay() -> None:
    eval_env = make_same_z_terrain_eval_config(_terrain_env_cfg(), seed=9)
    assert eval_env.disable_obs_noise
    assert eval_env.disable_domain_randomization
    assert eval_env.fixed_direct_depth_delay_frames == 0
    assert eval_env.max_episode_length_s == 30.0
    config = SameZTerrainEvaluationConfig()
    assert config.affects_motion_priority is False
    assert config.terrain_families == RP1_TERRAIN_COMPONENT_NAMES
    evaluator = config.build()
    context = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)
    expert = SimpleNamespace()
    with (
        patch(
            "humanoidverse.agents.evaluations.same_z_terrain._expert_motion_slice",
            return_value={"state": torch.zeros(5, 1)},
        ),
        patch(
            "humanoidverse.agents.evaluations.same_z_terrain._encode_motion_contexts",
            return_value=[context],
        ),
    ):
        encoded = evaluator._encode_once(SimpleNamespace(), expert, [17], device="cpu")[17]
    assert encoded["dtype"] == "torch.float32"
    assert encoded["shape"] == (4, 3)
    assert set(encoded["clones"]) == set(RP1_TERRAIN_COMPONENT_NAMES)
    assert {tensor_checksum(value) for value in encoded["clones"].values()} == {encoded["hash"]}


def test_same_z_reference_heading_is_shared_and_aligned_with_terrain_column() -> None:
    half_yaw = torch.tensor(0.35)
    root_rot = torch.tensor(
        [0.0, 0.0, torch.sin(half_yaw), torch.cos(half_yaw)],
        dtype=torch.float32,
    ).repeat(3, 1)
    reference = {
        "root_pos": torch.tensor([[1.0, 2.0, 0.8], [1.5, 2.5, 0.8], [2.0, 3.0, 0.8]]),
        "root_rot": root_rot,
        "root_vel": torch.tensor([[1.0, 1.0, 0.0]]).repeat(3, 1),
        "root_ang_vel": torch.tensor([[0.2, 0.3, 0.4]]).repeat(3, 1),
        "dof_pos": torch.zeros(3, 2),
        "dof_vel": torch.zeros(3, 2),
    }
    forward = _rotate_reference_to_course(reference, target_heading=0.0)
    inward_from_far_edge = _rotate_reference_to_course(reference, target_heading=np.pi)
    forward_delta = forward["root_pos"][-1, :2] - forward["root_pos"][0, :2]
    reverse_delta = (
        inward_from_far_edge["root_pos"][-1, :2]
        - inward_from_far_edge["root_pos"][0, :2]
    )
    assert forward_delta[0] > 0 and abs(float(forward_delta[1])) < 1.0e-6
    assert reverse_delta[0] < 0 and abs(float(reverse_delta[1])) < 1.0e-6
    torch.testing.assert_close(torch.linalg.vector_norm(forward_delta), torch.sqrt(torch.tensor(2.0)))
    torch.testing.assert_close(
        _yaw_xyzw_for_test(forward["root_rot"][-1]) - _yaw_xyzw_for_test(forward["root_rot"][0]),
        torch.tensor(0.0),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_same_z_reference_rejects_motion_without_meaningful_net_travel() -> None:
    reference = {
        "root_pos": torch.tensor([[0.0, 0.0, 0.8], [0.2, 0.0, 0.8]]),
        "root_rot": torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1),
        "root_vel": torch.zeros(2, 3),
        "root_ang_vel": torch.zeros(2, 3),
        "dof_pos": torch.zeros(2, 2),
        "dof_vel": torch.zeros(2, 2),
    }
    with pytest.raises(ValueError, match="insufficient horizontal travel"):
        _rotate_reference_to_course(reference, target_heading=0.0, minimum_distance=1.25)


def _yaw_xyzw_for_test(quaternion: torch.Tensor) -> torch.Tensor:
    x, y, z, w = quaternion.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))


def test_agent_routes_discriminator_prior_and_fb_aux_main_without_crossing_streams() -> None:
    batch_size = 2

    def _batch(marker: float, *, prior: bool, expert: bool = False):
        observation = {
            "state": torch.full((batch_size, 1), marker),
            "terrain_priv": torch.zeros(batch_size, 2),
        }
        result = {
            "observation": observation,
            "next": {
                "observation": {
                    "state": torch.full((batch_size, 1), marker + 0.5),
                    "terrain_priv": torch.zeros(batch_size, 2),
                },
                "terminated": torch.zeros(batch_size, 1, dtype=torch.bool),
            },
        }
        if not expert:
            result.update(
                {
                    "action": torch.zeros(batch_size, 1),
                    "z": torch.full((batch_size, 3), marker),
                }
            )
        if prior:
            result["transition_terminated"] = torch.tensor([[False], [True]])
            result["transition_truncated"] = torch.tensor([[False], [False]])
        else:
            result["aux_rewards"] = {"balance": torch.ones(batch_size, 1)}
        return result

    class _Replay:
        def __init__(self, value) -> None:
            self.value = value

        def sample(self, _batch_size: int):
            return self.value

    class _IdentityNormalizer(torch.nn.Module):
        def forward(self, value):
            return value

    class _ZRecorder:
        def __init__(self) -> None:
            self.add_calls = 0

        def add(self, _value) -> None:
            self.add_calls += 1

    model = SimpleNamespace(
        device="cpu",
        _obs_normalizer=_IdentityNormalizer(),
        _aux_reward_normalizer=torch.nn.Identity(),
    )
    agent = FBcprAuxAgent.__new__(FBcprAuxAgent)
    agent._model = model
    agent.cfg = SimpleNamespace(
        model=SimpleNamespace(),
        train=SimpleNamespace(
            batch_size=batch_size,
            discount=0.99,
            grad_penalty_discriminator=0.0,
            relabel_ratio=1.0,
            q_loss_coef=0.0,
            clip_grad_norm=0.0,
            fb_target_tau=0.01,
            critic_target_tau=0.01,
        ),
        aux_rewards=["balance"],
        aux_rewards_scaling={"balance": 1.0},
    )
    agent.z_buffer = _ZRecorder()
    agent._forward_map_paramlist = (torch.zeros(1),)
    agent._target_forward_map_paramlist = (torch.zeros(1),)
    agent._backward_map_paramlist = (torch.zeros(1),)
    agent._target_backward_map_paramlist = (torch.zeros(1),)
    agent._critic_map_paramlist = (torch.zeros(1),)
    agent._target_critic_map_paramlist = (torch.zeros(1),)
    agent._aux_critic_map_paramlist = (torch.zeros(1),)
    agent._aux_target_critic_map_paramlist = (torch.zeros(1),)
    agent.encode_expert = lambda **_kwargs: torch.full((batch_size, 3), 30.0)
    sampled = iter((torch.full((batch_size, 3), 11.0), torch.full((batch_size, 3), 22.0)))
    agent.sample_mixed_z = lambda **_kwargs: next(sampled)
    seen: dict[str, object] = {}

    def _record(name: str, **kwargs):
        seen[name] = kwargs
        return {name: torch.tensor(0.0)}

    agent.update_discriminator = lambda **kwargs: _record("D", **kwargs)
    agent.update_fb = lambda **kwargs: _record("FB", **kwargs)
    agent.update_critic = lambda **kwargs: _record("QD", **kwargs)
    agent.update_aux_critic = lambda **kwargs: _record("Aux", **kwargs)
    agent._run_actor_update = lambda **kwargs: _record("Actor", **kwargs)
    replay = {
        "expert_slicer": _Replay(_batch(30.0, prior=False, expert=True)),
        "train": _Replay(_batch(10.0, prior=False)),
        "prior": _Replay(_batch(20.0, prior=True)),
    }
    agent.update(replay, step=0)
    assert torch.all(seen["D"]["train_obs"]["state"] == 20.0)
    assert torch.all(seen["FB"]["obs"]["state"] == 10.0)
    assert torch.all(seen["Aux"]["obs"]["state"] == 10.0)
    assert torch.all(seen["QD"]["obs"]["state"] == 20.0)
    assert torch.all(seen["Actor"]["main_obs"]["state"] == 10.0)
    assert torch.all(seen["Actor"]["prior_obs"]["state"] == 20.0)
    torch.testing.assert_close(seen["QD"]["discount"], torch.tensor([[0.99], [0.0]]))
    assert agent.z_buffer.add_calls == 1


class _ActorDistribution:
    def __init__(self, action: torch.Tensor) -> None:
        self.action = action

    def sample(self, *, clip: float) -> torch.Tensor:
        return self.action.clamp(-clip, clip)


class _CountingActor(torch.nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.1))
        self.action_dim = action_dim
        self.batch_sizes: list[int] = []

    def forward(self, obs, z, std) -> _ActorDistribution:
        self.batch_sizes.append(int(z.shape[0]))
        return _ActorDistribution(self.bias.expand(z.shape[0], self.action_dim))


class _CountingQ(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, obs, z, action) -> torch.Tensor:
        self.batch_sizes.append(int(z.shape[0]))
        value = action.mean(dim=-1, keepdim=True)
        return torch.stack((value, value + 0.01), dim=0)


class _CountingForward(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, obs, z, action) -> torch.Tensor:
        self.batch_sizes.append(int(z.shape[0]))
        value = action.mean(dim=-1, keepdim=True).expand_as(z)
        return torch.stack((value, value + 0.01), dim=0)


class _CountingOptimizer:
    def __init__(self, parameters) -> None:
        self.inner = torch.optim.SGD(parameters, lr=0.01)
        self.steps = 0

    def zero_grad(self, *, set_to_none: bool) -> None:
        self.inner.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.steps += 1
        self.inner.step()


def test_actor_combines_main_and_plane_losses_in_one_optimizer_step() -> None:
    actor = _CountingActor(action_dim=2)
    critic = _CountingQ()
    aux_critic = _CountingQ()
    forward = _CountingForward()
    agent = FBcprAuxAgent.__new__(FBcprAuxAgent)
    agent._model = SimpleNamespace(
        device="cpu",
        amp_dtype=torch.bfloat16,
        cfg=SimpleNamespace(actor_std=0.05),
        _actor=actor,
        _critic=critic,
        _aux_critic=aux_critic,
        _forward_map=forward,
    )
    agent.cfg = SimpleNamespace(
        model=SimpleNamespace(amp=False),
        train=SimpleNamespace(
            stddev_clip=1.0,
            actor_pessimism_penalty=0.5,
            scale_reg=True,
            reg_coeff=0.05,
            reg_coeff_aux=0.02,
        ),
    )
    agent._distributed_training_stages = {}
    agent._sync_gradients_if_manual = lambda _parameters: None
    agent.actor_optimizer = _CountingOptimizer(actor.parameters())
    main_obs = {"state": torch.zeros(2, 1)}
    prior_obs = {"state": torch.ones(2, 1)}
    agent.update_actor(
        main_obs=main_obs,
        main_z=torch.ones(2, 3),
        prior_obs=prior_obs,
        prior_z=torch.ones(2, 3),
        clip_grad_norm=None,
    )
    assert actor.batch_sizes == [4]
    assert critic.batch_sizes == [2]
    assert aux_critic.batch_sizes == [2]
    assert forward.batch_sizes == [2]
    assert agent.actor_optimizer.steps == 1
