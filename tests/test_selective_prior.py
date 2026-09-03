from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from humanoidverse.agents.buffers.trajectory import TrajectoryDictBufferMultiDim
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgent
from humanoidverse.agents.presets.fb_depth import build_fb_depth_agent
from humanoidverse.agents.selective_prior import (
    PriorLabel,
    PriorPhase,
    SelectivePriorState,
    active_finalized_mask,
    actor_prior_interior_mask,
    qd_interior_mask,
)


def test_unknown_is_not_a_qd_terminal_or_training_sample() -> None:
    active = torch.tensor([[True], [False], [True], [True]])
    contexts = torch.tensor([[7], [7], [9], [9]])
    done = torch.tensor([[False], [False], [False], [True]])
    successor = torch.tensor([[True], [True], [True], [False]])

    mask = qd_interior_mask(
        active=active,
        context_id=contexts,
        transition_done=done,
        successor_available=successor,
    )

    # finalized -> UNKNOWN is omitted, not converted to a zero-value terminal.
    assert mask[:, 0].tolist() == [False, False, True, True]


def test_actor_d_requires_deeper_same_context_verified_interior() -> None:
    active = torch.ones(6, 1, dtype=torch.bool)
    contexts = torch.tensor([[1], [1], [1], [2], [2], [2]])
    done = torch.zeros(6, 1, dtype=torch.bool)
    successor = torch.tensor([[True], [True], [True], [True], [True], [False]])

    mask = actor_prior_interior_mask(
        active=active,
        context_id=contexts,
        transition_done=done,
        successor_available=successor,
        horizon=3,
    )

    assert mask[:, 0].tolist() == [True, False, False, True, False, False]


def test_fresh_finalized_mask_expires_without_relabeling_unknown_as_bad() -> None:
    labels = torch.tensor([[PriorLabel.VALIDATED], [PriorLabel.BAD], [PriorLabel.UNKNOWN]], dtype=torch.int8)
    label_steps = torch.tensor([[90], [1], [99]])

    mask = active_finalized_mask(labels, label_steps, step=100, ttl_steps=20)

    assert mask[:, 0].tolist() == [True, False, False]


def _make_replay() -> TrajectoryDictBufferMultiDim:
    replay = TrajectoryDictBufferMultiDim(
        capacity=5,
        n_dim=2,
        end_key="truncated",
        output_key_t=[
            "observation",
            "action",
            "z",
            "truncated",
            "prior_label",
            "prior_generation",
        ],
        output_key_tp1=["observation"],
    )
    time, env = 4, 2
    replay.extend(
        {
            "observation": {"state": torch.arange(time * env).reshape(time, env, 1).float()},
            "action": torch.zeros(time, env, 1),
            "z": torch.zeros(time, env, 2),
            "truncated": torch.zeros(time, env, 1, dtype=torch.bool),
            "prior_label": torch.zeros(time, env, 1, dtype=torch.int8),
            "prior_generation": torch.arange(time).reshape(time, 1, 1).expand(-1, env, -1),
        }
    )
    return replay


def test_masked_replay_sampling_reuses_original_storage_and_has_real_successor() -> None:
    replay = _make_replay()
    mask = torch.zeros(5, 2, dtype=torch.bool)
    mask[1, 0] = True

    batch, idxs = replay.sample_from_mask(mask, 8, return_indices=True)

    assert idxs[0].unique().tolist() == [1]
    assert idxs[1].unique().tolist() == [0]
    assert batch["observation"]["state"].eq(2).all()
    assert batch["next"]["observation"]["state"].eq(4).all()


def test_generation_checked_metadata_write_rejects_overwritten_slots() -> None:
    replay = _make_replay()
    idxs = (torch.tensor([0, 1]), torch.tensor([0, 0]))
    written = replay.set_fields_at_indices(
        idxs,
        {"prior_label": torch.tensor([[PriorLabel.VALIDATED], [PriorLabel.BAD]])},
        expected_generation=torch.tensor([[999], [1]]),
    )

    assert written == 1
    assert replay.storage["prior_label"][0, 0].item() == PriorLabel.UNKNOWN
    assert replay.storage["prior_label"][1, 0].item() == PriorLabel.BAD


def test_selective_prior_state_checkpoint_roundtrip() -> None:
    original = SelectivePriorState(
        phase=PriorPhase.ACTOR_PRIOR,
        bank_version=5,
        gate_teacher_version=3,
        discriminator_version=4,
        qd_reward_version=4,
        update_count=123,
    )

    restored = SelectivePriorState.from_state_dict(original.state_dict())

    assert restored == original
    assert restored.phase_enum is PriorPhase.ACTOR_PRIOR


def test_selective_discriminator_uses_raw_logits_and_effective_class_mass() -> None:
    class _Discriminator(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.offset = torch.nn.Parameter(torch.zeros(()))

        def compute_logits(self, obs, z):
            del z
            return obs["state"] + self.offset

        def forward(self, obs, z):
            # Deliberately different: the selective LSGAN must never use this.
            return torch.sigmoid(self.compute_logits(obs, z))

    discriminator = _Discriminator()
    agent = FBcprAuxAgent.__new__(FBcprAuxAgent)
    agent._model = SimpleNamespace(
        device="cpu",
        amp_dtype=torch.bfloat16,
        _discriminator=discriminator,
    )
    agent.cfg = SimpleNamespace(
        model=SimpleNamespace(amp=False),
        train=SimpleNamespace(
            selective_prior_expert_fraction=0.50,
            selective_prior_validated_fraction=0.33,
            selective_prior_bad_fraction=0.17,
            selective_prior_validated_weight=0.5,
            selective_prior_d_positive_min=99.0,
            selective_prior_d_bad_max=-99.0,
            selective_prior_d_expert_validated_gap_max=0.0,
            selective_prior_d_expert_validated_auc_max=0.0,
            selective_prior_d_validated_bad_auc_min=1.0,
            selective_prior_d_min_updates=999,
            selective_prior_d_ready_streak=999,
        ),
    )
    agent._distributed_training_stages = {}
    agent._selective_prior_state = SelectivePriorState(phase=PriorPhase.FIT_D)
    agent._sync_gradients_if_manual = lambda _parameters: None
    agent.discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr=0.0)
    obs_positive = {"state": torch.full((2, 1), 0.5)}
    obs_bad = {"state": torch.full((2, 1), -0.5)}

    metrics = agent.update_selective_discriminator(
        expert_obs=obs_positive,
        expert_z=torch.zeros(2, 1),
        validated_obs=obs_positive,
        validated_z=torch.zeros(2, 1),
        validated_confidence=torch.ones(2, 1),
        bad_obs=obs_bad,
        bad_z=torch.zeros(2, 1),
        grad_penalty=None,
    )

    expected = 0.50 * 0.125 + 0.33 * 0.5 * 0.125 + 0.17 * 0.125
    assert metrics["disc/data_loss"].item() == pytest.approx(expected)


def test_delayed_gate_promotes_only_same_context_exact_tracking_windows() -> None:
    class _IdentityTeacher(torch.nn.Module):
        def forward(self, obs):
            return obs["state"]

    class _IdentityNormalizer(torch.nn.Module):
        def forward(self, obs):
            return obs

    time, env, z_dim = 10, 2, 2
    replay = TrajectoryDictBufferMultiDim(
        capacity=12,
        n_dim=2,
        end_key="truncated",
        output_key_t=[
            "observation",
            "action",
            "z",
            "truncated",
            "transition_terminated",
            "transition_truncated",
            "heading_context_id",
            "heading_source_type",
            "prior_motion_id",
            "prior_label",
            "prior_label_step",
            "prior_teacher_version",
            "prior_confidence",
            "prior_generation",
        ],
        output_key_tp1=["observation"],
    )
    z = torch.tensor([1.0, 0.0]).reshape(1, 1, z_dim).expand(time, env, -1)
    replay.extend(
        {
            "observation": {
                "state": z.clone(),
                "heading": torch.zeros(time, env, 2),
            },
            "action": torch.zeros(time, env, 1),
            "z": z.clone(),
            "truncated": torch.zeros(time, env, 1, dtype=torch.bool),
            "transition_terminated": torch.zeros(time, env, 1, dtype=torch.bool),
            "transition_truncated": torch.zeros(time, env, 1, dtype=torch.bool),
            "heading_context_id": torch.ones(time, env, 1, dtype=torch.long),
            "heading_source_type": torch.full((time, env, 1), 2, dtype=torch.long),
            "prior_motion_id": torch.arange(env).reshape(1, env, 1).expand(time, -1, -1),
            "prior_label": torch.zeros(time, env, 1, dtype=torch.int8),
            "prior_label_step": torch.zeros(time, env, 1, dtype=torch.long),
            "prior_teacher_version": torch.zeros(time, env, 1, dtype=torch.long),
            "prior_confidence": torch.zeros(time, env, 1),
            "prior_generation": torch.arange(time).reshape(time, 1, 1).expand(-1, env, -1),
        }
    )
    agent = FBcprAuxAgent.__new__(FBcprAuxAgent)
    agent._model = SimpleNamespace(
        device="cpu",
        amp_dtype=torch.bfloat16,
        amp=False,
        _obs_normalizer=_IdentityNormalizer(),
        _target_backward_map=_IdentityTeacher(),
        project_z=lambda value: value,
    )
    agent.cfg = SimpleNamespace(
        model=SimpleNamespace(amp=False),
        train=SimpleNamespace(
            selective_prior_gate_teacher_refresh_updates=999,
            selective_prior_expansion_refresh_updates=999,
            selective_prior_gate_window=2,
            selective_prior_gate_future=2,
            selective_prior_gate_windows_per_refresh=8,
            selective_prior_good_cosine_mean=0.9,
            selective_prior_good_cosine_min=0.9,
            selective_prior_bad_cosine_mean=-0.5,
            selective_prior_bad_sustain_fraction=0.5,
            selective_prior_good_heading_cost_mean_max=0.3,
            selective_prior_bad_heading_cost_mean_min=1.0,
        ),
    )
    agent._gate_backward_teacher = _IdentityTeacher()
    agent._selective_prior_state = SelectivePriorState()
    agent._last_selective_mask_step = None
    agent._cached_selective_masks = None
    agent._last_selective_gate_step = None
    agent._last_selective_teacher_refresh_update = 0

    metrics = agent._refresh_selective_prior_labels(replay, step=100)

    assert metrics["prior/gate_good_window_fraction"].item() == pytest.approx(1.0)
    assert (replay.storage["prior_label"] == PriorLabel.VALIDATED).any()
    assert not (replay.storage["prior_label"] == PriorLabel.BAD).any()


def test_bootstrap_phase_trains_main_objectives_but_not_prior_branch() -> None:
    batch_size = 2
    main_batch = {
        "observation": {"state": torch.zeros(batch_size, 2)},
        "action": torch.zeros(batch_size, 1),
        "z": torch.ones(batch_size, 2),
        "next": {
            "observation": {"state": torch.ones(batch_size, 2)},
            "terminated": torch.zeros(batch_size, 1, dtype=torch.bool),
        },
        "aux_rewards": {"safe": torch.ones(batch_size, 1)},
    }
    expert_batch = {
        "observation": {"state": torch.zeros(batch_size, 2)},
        "next": {"observation": {"state": torch.ones(batch_size, 2)}},
    }

    class _Replay:
        def __init__(self, value):
            self.value = value

        def sample(self, _batch_size):
            return self.value

    class _IdentityNormalizer(torch.nn.Module):
        def forward(self, value):
            return value

    agent = FBcprAuxAgent.__new__(FBcprAuxAgent)
    agent._model = SimpleNamespace(
        device="cpu",
        _obs_normalizer=_IdentityNormalizer(),
        _aux_reward_normalizer=torch.nn.Identity(),
    )
    agent.cfg = SimpleNamespace(
        model=SimpleNamespace(),
        train=SimpleNamespace(
            selective_prior_enabled=True,
            behavior_prior_enabled=True,
            batch_size=batch_size,
            discount=0.99,
            q_loss_coef=0.0,
            clip_grad_norm=0.0,
            fb_target_tau=0.01,
            critic_target_tau=0.01,
            relabel_ratio=None,
        ),
        aux_rewards=["safe"],
        aux_rewards_scaling={"safe": 1.0},
    )
    agent._selective_prior_state = SelectivePriorState()
    agent._last_selective_mask_step = None
    agent._cached_selective_masks = None
    agent._refresh_selective_prior_labels = lambda *_args: {}
    agent._selective_prior_active_masks = lambda *_args: {}
    agent._selective_prior_phase_from_coverage = lambda *_args: {}
    agent.encode_expert = lambda **_kwargs: torch.ones(batch_size, 2)
    agent._relabel_main_z = lambda **kwargs: kwargs["main_rollout_z"]
    calls: list[str] = []

    def _record(name):
        def call(**_kwargs):
            calls.append(name)
            return {name: torch.tensor(0.0)}

        return call

    agent.update_fb = _record("FB")
    agent.update_aux_critic = _record("Aux")
    agent.update_selective_discriminator = _record("D")
    agent.update_selective_prior_critic = _record("QD")
    agent._run_actor_update = _record("Actor")
    agent._forward_map_paramlist = (torch.zeros(1),)
    agent._target_forward_map_paramlist = (torch.zeros(1),)
    agent._backward_map_paramlist = (torch.zeros(1),)
    agent._target_backward_map_paramlist = (torch.zeros(1),)
    agent._aux_critic_map_paramlist = (torch.zeros(1),)
    agent._aux_target_critic_map_paramlist = (torch.zeros(1),)

    agent.update({"train": _Replay(main_batch), "expert_slicer": _Replay(expert_batch)}, step=0)

    assert calls == ["FB", "Aux", "Actor"]


def test_fb_depth_selective_prior_preset_keeps_main_stream_lsgan_contract() -> None:
    config = build_fb_depth_agent(
        device="cpu",
        compile=False,
        selective_prior=True,
        behavior_prior=True,
        heading_context=True,
        heading_reg_coeff=0.002,
    )

    assert config.train.selective_prior_enabled
    assert config.train.behavior_prior_enabled
    assert config.train.discriminator_loss == "lsgan"
    assert config.train.discriminator_reward == "amp"
    assert config.model.heading_context_enabled
