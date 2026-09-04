from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from humanoidverse.agents.buffers.trajectory import TrajectoryDictBuffer, TrajectoryDictBufferMultiDim
from humanoidverse.agents.buffers.transition import dtype_numpytotorch
from humanoidverse.agents.fb_cpr.agent import FBcprAgent
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgent
from humanoidverse.agents.presets.fb_depth import build_fb_depth_agent
from humanoidverse.agents.selective_prior import (
    PriorCoordinateContract,
    PriorLabel,
    PriorPhase,
    ReferenceProvenanceBatch,
    SelectivePriorState,
    TemporalEncodingContract,
    active_finalized_mask,
    actor_prior_interior_mask,
    future_window_means,
    masked_temporal_mean,
    qd_interior_mask,
    resolve_prior_proposals,
    shadow_refresh_due,
)
from humanoidverse.selective_prior_audit import (
    GateThresholds,
    classify_gate_windows,
    validate_exact_tracking_windows,
)


def test_temporal_encoding_contract_matches_clipped_tracking_mean() -> None:
    embeddings = torch.tensor(
        [
            [[1.0], [2.0], [3.0], [4.0]],
            [[10.0], [20.0], [30.0], [40.0]],
        ]
    )
    horizon = torch.tensor([4, 2])
    torch.testing.assert_close(masked_temporal_mean(embeddings, horizon), torch.tensor([[2.5], [15.0]]))

    policy = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]])
    means = future_window_means(policy, torch.tensor([[3, 2, 1]]))
    torch.testing.assert_close(means, torch.tensor([[[2.0], [2.5], [3.0]]]))


def test_prior_coordinate_contract_detects_mixed_snapshot_metadata() -> None:
    temporal = TemporalEncodingContract(sequence_length=8)
    contract = PriorCoordinateContract(
        version=3,
        bank_version=7,
        encoder_fingerprint="encoder",
        normalizer_fingerprint="normalizer",
        expert_dataset_fingerprint="dataset",
        temporal_contract=temporal.state_dict(),
    )
    restored = PriorCoordinateContract.from_state_dict(contract.state_dict())
    assert restored == contract
    corrupted = contract.state_dict()
    corrupted["normalizer_fingerprint"] = "different"
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        PriorCoordinateContract.from_state_dict(corrupted)


def test_reference_provenance_requires_explicit_next_identity_for_bellman_use() -> None:
    provenance = ReferenceProvenanceBatch(
        motion_id=torch.tensor([[1], [2]]),
        reference_index=torch.tensor([[10], [20]]),
        reference_horizon=torch.tensor([[8], [3]]),
    )
    provenance.validate()
    with pytest.raises(ValueError, match=r"explicit p_t and p_t\+1"):
        provenance.validate(require_next=True)


def test_same_provenance_has_stationary_prior_z_and_current_actor_z() -> None:
    class _Expert:
        dataset_fingerprint = "expert-v1"

        def __init__(self) -> None:
            self.storage = {
                "motion_id": torch.zeros(5, 1, dtype=torch.long),
                "observation": {"state": torch.arange(5, dtype=torch.float32).reshape(5, 1)},
            }

        def __len__(self) -> int:
            return 5

    class _Scale(torch.nn.Module):
        def __init__(self, scale: float) -> None:
            super().__init__()
            self.register_buffer("scale", torch.tensor(scale))

        def forward(self, obs):
            return obs["state"] * self.scale

    class _Identity(torch.nn.Module):
        def forward(self, obs):
            return obs

    prior_encoder = _Scale(1.0)
    online_encoder = _Scale(2.0)
    normalizer = _Identity()
    agent = FBcprAuxAgent.__new__(FBcprAuxAgent)
    agent._model = SimpleNamespace(
        device="cpu",
        amp_dtype=torch.bfloat16,
        _backward_map=online_encoder,
        _obs_normalizer=normalizer,
        project_z=lambda value: value,
    )
    agent.cfg = SimpleNamespace(model=SimpleNamespace(amp=False, seq_length=2))
    agent._prior_backward_map = prior_encoder
    agent._prior_obs_normalizer = normalizer
    agent._prior_coordinate_contract = PriorCoordinateContract(
        version=1,
        bank_version=1,
        encoder_fingerprint="unused-by-this-unit-test",
        normalizer_fingerprint="unused-by-this-unit-test",
        expert_dataset_fingerprint="expert-v1",
        temporal_contract=TemporalEncodingContract(sequence_length=2).state_dict(),
    )
    provenance = ReferenceProvenanceBatch(
        motion_id=torch.tensor([[0]]),
        reference_index=torch.tensor([[0]]),
        reference_horizon=torch.tensor([[2]]),
    )
    expert = _Expert()

    prior_before = agent.encode_prior_provenance(provenance, expert_buffer=expert)
    actor_before = agent.encode_actor_provenance(provenance, expert_buffer=expert)
    online_encoder.scale.fill_(3.0)
    prior_after = agent.encode_prior_provenance(provenance, expert_buffer=expert)
    actor_after = agent.encode_actor_provenance(provenance, expert_buffer=expert)

    torch.testing.assert_close(prior_before, torch.tensor([[1.5]]))
    torch.testing.assert_close(prior_after, prior_before)
    torch.testing.assert_close(actor_before, torch.tensor([[3.0]]))
    torch.testing.assert_close(actor_after, torch.tensor([[4.5]]))
    assert not actor_after.requires_grad
    assert online_encoder.scale.grad is None


def test_qd_uses_online_actor_z_but_frozen_prior_z_for_reward_and_value() -> None:
    seen: dict[str, torch.Tensor] = {}

    class _Distribution:
        def __init__(self, action: torch.Tensor) -> None:
            self.action = action

        def sample(self, clip=None):
            del clip
            return self.action

    class _Actor(torch.nn.Module):
        def forward(self, obs, z, std):
            del obs, std
            seen["actor_z"] = z.detach().clone()
            return _Distribution(torch.zeros(z.shape[0], 1))

    class _RewardD(torch.nn.Module):
        def compute_logits(self, obs, z):
            del obs
            seen["reward_z"] = z.detach().clone()
            return torch.zeros(z.shape[0], 1)

    class _Critic(torch.nn.Module):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name
            self.offset = torch.nn.Parameter(torch.zeros(()))

        def forward(self, obs, z, action):
            del obs, action
            seen[f"{self.name}_z"] = z.detach().clone()
            return self.offset + torch.zeros(2, z.shape[0], 1)

    critic = _Critic("critic")
    target = _Critic("target")
    agent = FBcprAuxAgent.__new__(FBcprAuxAgent)
    agent._model = SimpleNamespace(
        device="cpu",
        amp_dtype=torch.bfloat16,
        _actor=_Actor(),
        _critic=critic,
        _target_critic=target,
        cfg=SimpleNamespace(actor_std=0.1),
    )
    agent._prior_reward_discriminator = _RewardD()
    agent._distributed_training_stages = {}
    agent._selective_prior_state = SelectivePriorState(phase=PriorPhase.FIT_QD)
    agent.cfg = SimpleNamespace(
        model=SimpleNamespace(amp=False, archi=SimpleNamespace(critic=SimpleNamespace(num_parallel=2))),
        train=SimpleNamespace(
            stddev_clip=0.3,
            critic_pessimism_penalty=0.0,
            selective_prior_qd_relative_uncertainty_max=1.0,
            selective_prior_qd_ready_streak=999,
            selective_prior_qd_min_updates=999,
        ),
    )
    agent.discriminator_reward_from_logits = lambda logits: torch.ones_like(logits)
    agent._sync_gradients_if_manual = lambda _parameters: None
    agent.critic_optimizer = torch.optim.SGD(critic.parameters(), lr=0.0)

    agent.update_selective_prior_critic(
        obs={"state": torch.zeros(2, 1)},
        action=torch.zeros(2, 1),
        discount=torch.full((2, 1), 0.99),
        next_obs={"state": torch.zeros(2, 1)},
        z=torch.full((2, 1), 3.0),
        next_z=torch.full((2, 1), 4.0),
        actor_next_obs={"state": torch.zeros(2, 1)},
        actor_next_z=torch.full((2, 1), 7.0),
    )

    torch.testing.assert_close(seen["reward_z"], torch.full((2, 1), 3.0))
    torch.testing.assert_close(seen["actor_z"], torch.full((2, 1), 7.0))
    torch.testing.assert_close(seen["target_z"], torch.full((2, 1), 4.0))


def test_replay_dtype_mapping_supports_numpy_int8_prior_metadata() -> None:
    assert dtype_numpytotorch(np.dtype("int8")) == torch.int8


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


def test_gate_conflict_is_unknown_and_good_requires_delayed_second_pass() -> None:
    label, candidate_step = resolve_prior_proposals(
        good=torch.tensor([True, True, True]),
        bad=torch.tensor([True, False, False]),
        getup=torch.tensor([False, False, False]),
        getup_success=torch.tensor([False, False, False]),
        old_label=torch.tensor([PriorLabel.UNKNOWN, PriorLabel.UNKNOWN, PriorLabel.CANDIDATE]),
        old_label_teacher_version=torch.tensor([0, 0, 0]),
        old_candidate_step=torch.tensor([0, 0, 10]),
        old_candidate_teacher_version=torch.tensor([0, 0, 4]),
        step=25,
        teacher_version=4,
        candidate_min_age_steps=10,
    )

    assert label.tolist() == [PriorLabel.UNKNOWN, PriorLabel.CANDIDATE, PriorLabel.VALIDATED]
    assert candidate_step[1].item() == 25


def test_new_teacher_requires_a_fresh_delayed_candidate_pass() -> None:
    label, candidate_step = resolve_prior_proposals(
        good=torch.tensor([True, True]),
        bad=torch.tensor([False, False]),
        getup=torch.tensor([False, False]),
        getup_success=torch.tensor([False, False]),
        old_label=torch.tensor([PriorLabel.CANDIDATE, PriorLabel.VALIDATED]),
        old_label_teacher_version=torch.tensor([0, 3]),
        old_candidate_step=torch.tensor([1, 0]),
        old_candidate_teacher_version=torch.tensor([3, 0]),
        step=100,
        teacher_version=4,
        candidate_min_age_steps=10,
    )

    assert label.tolist() == [PriorLabel.CANDIDATE, PriorLabel.CANDIDATE]
    assert candidate_step.tolist() == [100, 100]


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


def test_masked_expert_sampling_preserves_contiguous_sequences() -> None:
    episodes = []
    for motion_id in (0, 1):
        states = torch.arange(6, dtype=torch.float32).unsqueeze(-1) + 10 * motion_id
        episodes.append(
            {
                "observation": {"state": states},
                "motion_id": torch.full((6, 1), motion_id, dtype=torch.long),
            }
        )
    replay = TrajectoryDictBuffer(
        episodes,
        seq_length=2,
        output_key_t=["observation", "motion_id"],
        output_key_tp1=["observation"],
    )
    holdout = replay.storage["motion_id"].reshape(-1).eq(1)

    batch, (indices,) = replay.sample_from_mask(
        holdout,
        batch_size=8,
        seq_length=2,
        return_indices=True,
    )

    assert batch["motion_id"].eq(1).all()
    assert indices.reshape(-1, 2).diff(dim=1).eq(1).all()
    assert batch["observation"]["state"].reshape(-1, 2).diff(dim=1).eq(1).all()


def test_encode_expert_uses_actual_selective_batch_size() -> None:
    agent = FBcprAgent.__new__(FBcprAgent)
    agent._model = SimpleNamespace(
        device="cpu",
        amp_dtype=torch.bfloat16,
        _backward_map=lambda obs: obs,
        project_z=lambda z: z,
    )
    # Deliberately leave the historical main batch size at 1024. A selective
    # E/V/B quota supplies only two real 8-frame sequences here.
    agent.cfg = SimpleNamespace(
        model=SimpleNamespace(amp=False, seq_length=8),
        train=SimpleNamespace(batch_size=1024),
    )
    sequence = torch.arange(16, dtype=torch.float32).unsqueeze(-1)

    encoded = agent.encode_expert(sequence)

    assert encoded.shape == (16, 1)
    torch.testing.assert_close(encoded[:8], torch.full((8, 1), 3.5))
    torch.testing.assert_close(encoded[8:], torch.full((8, 1), 11.5))


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


def test_shadow_bank_swap_is_atomic_and_active_is_immutable_between_swaps() -> None:
    class _Replay:
        def __init__(self) -> None:
            shape = (3, 1, 1)
            self.storage = {
                "prior_label": torch.tensor([PriorLabel.VALIDATED, PriorLabel.BAD, PriorLabel.CANDIDATE], dtype=torch.int8).reshape(shape),
                "prior_label_step": torch.tensor([90, 90, 90], dtype=torch.long).reshape(shape),
                "prior_teacher_version": torch.full(shape, 2, dtype=torch.long),
                "prior_confidence": torch.tensor([0.9, 0.8, 0.7]).reshape(shape),
                "prior_active_label": torch.full(shape, PriorLabel.BAD, dtype=torch.int8),
                "prior_active_label_step": torch.ones(shape, dtype=torch.long),
                "prior_active_teacher_version": torch.ones(shape, dtype=torch.long),
                "prior_active_confidence": torch.ones(shape),
                "prior_active_bank_version": torch.ones(shape, dtype=torch.long),
            }

        def _valid_slot_mask(self):
            return torch.ones(3, 1, dtype=torch.bool)

    replay = _Replay()
    agent = FBcprAuxAgent.__new__(FBcprAuxAgent)
    agent.cfg = SimpleNamespace(train=SimpleNamespace(selective_prior_label_ttl_local_steps=100))
    agent._selective_prior_state = SelectivePriorState(
        phase=PriorPhase.ACTOR_PRIOR,
        bank_version=1,
        active_teacher_version=1,
        gate_teacher_version=2,
        shadow_building=1,
        update_count=12,
    )
    agent._last_selective_mask_step = 12
    agent._cached_selective_masks = {"old": torch.tensor(1)}
    agent._activate_prior_coordinate_contract = lambda _expert: None

    agent._commit_shadow_prior_bank(replay, step=100, expert_buffer=object())

    assert replay.storage["prior_active_label"][:, 0, 0].tolist() == [PriorLabel.VALIDATED, PriorLabel.BAD, PriorLabel.UNKNOWN]
    assert replay.storage["prior_active_bank_version"][:, 0, 0].tolist() == [2, 2, 0]
    assert agent._selective_prior_state.bank_version == 2
    assert agent._selective_prior_state.active_teacher_version == 2
    assert agent._selective_prior_state.phase_enum is PriorPhase.FIT_D
    replay.storage["prior_label"][0, 0, 0] = PriorLabel.BAD
    assert replay.storage["prior_active_label"][0, 0, 0].item() == PriorLabel.VALIDATED


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


def test_shadow_refresh_uses_policy_update_clock_before_active_support_expires() -> None:
    state = SelectivePriorState(
        phase=PriorPhase.ACTOR_PRIOR,
        policy_version=14_095,
        last_bank_swap_policy_version=10_000,
    )
    assert not shadow_refresh_due(
        phase=state.phase_enum,
        shadow_building=False,
        policy_version=state.policy_version,
        last_bank_swap_policy_version=state.last_bank_swap_policy_version,
        refresh_policy_updates=4096,
    )
    state.policy_version += 1
    assert shadow_refresh_due(
        phase=state.phase_enum,
        shadow_building=False,
        policy_version=state.policy_version,
        last_bank_swap_policy_version=state.last_bank_swap_policy_version,
        refresh_policy_updates=4096,
    )
    # The next SHADOW starts halfway through the 8192-update ACTIVE validity
    # horizon, rather than after 4096 collector steps (=65536 updates).
    assert state.policy_version - state.last_bank_swap_policy_version < 8192


def test_mature_candidate_promotion_does_not_require_random_slot_rehit_or_promote_getup() -> None:
    class _Replay:
        n_dim = 2

        def __init__(self) -> None:
            shape = (2, 1, 1)
            self.storage = {
                "prior_label": torch.full(shape, PriorLabel.CANDIDATE, dtype=torch.int8),
                "prior_label_step": torch.zeros(shape, dtype=torch.long),
                "prior_teacher_version": torch.zeros(shape, dtype=torch.long),
                "prior_candidate_step": torch.full(shape, 50, dtype=torch.long),
                "prior_candidate_teacher_version": torch.full(shape, 3, dtype=torch.long),
                "prior_policy_version": torch.full(shape, 100, dtype=torch.long),
                "prior_motion_id": torch.tensor([0, 1], dtype=torch.long).reshape(shape),
                "prior_generation": torch.tensor([1000, 1001], dtype=torch.long).reshape(shape),
            }

        def _valid_slot_mask(self):
            return torch.ones(2, 1, dtype=torch.bool)

        def set_fields_at_indices(self, idxs, values, *, expected_generation):
            actual = self.storage["prior_generation"][idxs]
            keep = actual.reshape(-1).eq(expected_generation.reshape(-1))
            for key, value in values.items():
                target = self.storage[key][idxs]
                self.storage[key][tuple(index[keep] for index in idxs)] = value.reshape_as(target)[keep]
            return int(keep.sum().item())

    class _Expert:
        motion_ids = [0, 1]
        file_names = ["walk1_subject1", "fallAndGetUp2_subject2"]

    replay = _Replay()
    agent = FBcprAuxAgent.__new__(FBcprAuxAgent)
    agent.cfg = SimpleNamespace(
        train=SimpleNamespace(
            selective_prior_candidate_min_age_local_steps=100,
            selective_prior_max_policy_version_age=8192,
        )
    )
    agent._selective_prior_state = SelectivePriorState(
        gate_teacher_version=3,
        shadow_building=1,
        policy_version=200,
    )

    promoted = agent._promote_mature_shadow_candidates(replay, step=200, expert_buffer=_Expert())

    assert promoted == 1
    assert replay.storage["prior_label"][:, 0, 0].tolist() == [PriorLabel.VALIDATED, PriorLabel.CANDIDATE]
    assert replay.storage["prior_label_step"][0, 0, 0].item() == 200
    assert replay.storage["prior_candidate_step"][0, 0, 0].item() == 0


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
    assert metrics["disc/effective_validated_mass"].item() == pytest.approx(0.33 * 0.5)
    assert metrics["disc/effective_bad_mass"].item() == pytest.approx(0.17)


def test_only_heldout_calibration_updates_d_readiness() -> None:
    class _Discriminator(torch.nn.Module):
        def compute_logits(self, obs, z):
            del z
            return obs["state"]

    agent = FBcprAuxAgent.__new__(FBcprAuxAgent)
    agent._model = SimpleNamespace(_discriminator=_Discriminator(), device="cpu")
    agent.cfg = SimpleNamespace(
        train=SimpleNamespace(
            selective_prior_d_positive_min=0.5,
            selective_prior_d_bad_max=0.0,
            selective_prior_d_expert_validated_gap_max=0.35,
            selective_prior_d_expert_validated_auc_max=0.65,
            selective_prior_d_validated_bad_auc_min=0.8,
            selective_prior_d_ready_streak=10,
            selective_prior_d_min_updates=999,
        )
    )
    agent._selective_prior_state = SelectivePriorState(
        phase=PriorPhase.FIT_D,
        discriminator_update_count=1,
    )
    positive = {"state": torch.ones(8, 1)}
    bad = {"state": -torch.ones(8, 1)}

    metrics = agent.evaluate_selective_discriminator_calibration(
        expert_obs=positive,
        expert_z=torch.zeros(8, 1),
        validated_obs=positive,
        validated_z=torch.zeros(8, 1),
        bad_obs=bad,
        bad_z=torch.zeros(8, 1),
    )

    assert metrics["prior/d_calibration_ready"].item() == 1.0
    assert agent._selective_prior_state.d_ready_streak == 1


def test_delayed_gate_promotes_only_same_context_exact_tracking_windows() -> None:
    class _IdentityTeacher(torch.nn.Module):
        def forward(self, obs):
            return obs["state"]

    class _IdentityNormalizer(torch.nn.Module):
        def forward(self, obs):
            return obs

    class _DriftingOnlineNormalizer(torch.nn.Module):
        def forward(self, obs):
            shifted = dict(obs)
            shifted["state"] = shifted["state"] + torch.tensor([0.0, 100.0])
            return shifted

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
            "prior_candidate_step",
            "prior_candidate_teacher_version",
            "prior_reference_index",
            "prior_reference_horizon",
            "prior_z_encoder_version",
            "prior_policy_version",
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
            "prior_motion_id": torch.zeros(time, env, 1, dtype=torch.long),
            "prior_label": torch.zeros(time, env, 1, dtype=torch.int8),
            "prior_label_step": torch.zeros(time, env, 1, dtype=torch.long),
            "prior_teacher_version": torch.zeros(time, env, 1, dtype=torch.long),
            "prior_confidence": torch.zeros(time, env, 1),
            "prior_generation": torch.arange(time).reshape(time, 1, 1).expand(-1, env, -1),
            "prior_candidate_step": torch.zeros(time, env, 1, dtype=torch.long),
            "prior_candidate_teacher_version": torch.zeros(time, env, 1, dtype=torch.long),
            "prior_reference_index": torch.arange(time).reshape(time, 1, 1).expand(-1, env, -1),
            "prior_reference_horizon": torch.full((time, env, 1), 2, dtype=torch.long),
            "prior_z_encoder_version": torch.zeros(time, env, 1, dtype=torch.long),
            "prior_policy_version": torch.zeros(time, env, 1, dtype=torch.long),
        }
    )
    agent = FBcprAuxAgent.__new__(FBcprAuxAgent)
    agent._model = SimpleNamespace(
        device="cpu",
        amp_dtype=torch.bfloat16,
        amp=False,
        _obs_normalizer=_DriftingOnlineNormalizer(),
        _target_backward_map=_IdentityTeacher(),
        project_z=lambda value: value,
        action_dim=1,
    )
    agent.cfg = SimpleNamespace(
        model=SimpleNamespace(amp=False, seq_length=2),
        train=SimpleNamespace(
            selective_prior_gate_teacher_refresh_updates=999,
            selective_prior_gate_bootstrap_updates=0,
            selective_prior_expansion_refresh_updates=999,
            selective_prior_candidate_min_age_local_steps=10,
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
    agent._gate_obs_normalizer = _IdentityNormalizer()
    agent._selective_prior_state = SelectivePriorState()
    agent._last_selective_candidate_promotion_step = 0
    agent._last_selective_mask_step = None
    agent._cached_selective_masks = None
    agent._last_selective_gate_step = None
    agent._last_selective_teacher_refresh_update = 0

    class _Expert:
        file_names = ["walk"]
        motion_ids = [0]

        def __init__(self):
            self.storage = {
                "motion_id": torch.zeros(16, 1, dtype=torch.long),
                "observation": {
                    "state": torch.tensor([1.0, 0.0]).reshape(1, 2).expand(16, -1).clone(),
                    "heading": torch.zeros(16, 2),
                },
            }

        def __len__(self):
            return 16

    metrics = agent._refresh_selective_prior_labels(replay, step=100, expert_buffer=_Expert())

    assert metrics["prior/gate_good_window_fraction"].item() == pytest.approx(1.0)
    assert (replay.storage["prior_label"] == PriorLabel.CANDIDATE).any()
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
    agent._last_selective_candidate_promotion_step = 0
    agent._last_selective_mask_step = None
    agent._cached_selective_masks = None
    agent._refresh_selective_prior_labels = lambda *_args, **_kwargs: {}
    agent._selective_masks_and_coverage = lambda *_args, **_kwargs: ({}, {})
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


def test_frozen_audit_rejects_context_or_reference_discontinuity() -> None:
    source = np.full((3, 4), 2, dtype=np.int64)
    context = np.full((3, 4), 7, dtype=np.int64)
    motion = np.full((3, 4), 11, dtype=np.int64)
    reference = np.tile(np.arange(20, 24, dtype=np.int64), (3, 1))
    done = np.zeros((3, 4), dtype=np.bool_)
    context[1, -1] = 8
    reference[2, 2] = 99

    valid = validate_exact_tracking_windows(
        source=source,
        context_id=context,
        motion_id=motion,
        reference_index=reference,
        transition_done=done,
    )

    assert valid.tolist() == [True, False, False]


def test_frozen_gate_good_bad_conflict_is_not_positive() -> None:
    cosine = torch.tensor(
        [
            [0.9, 0.9],
            [0.9, 0.9],
            [-0.4, -0.4],
        ]
    )
    heading = torch.zeros(3, 4)
    pathology = torch.zeros(3, 4, dtype=torch.bool)
    pathology[1, 0] = True
    thresholds = GateThresholds(
        good_cosine_mean=0.75,
        good_cosine_min=0.35,
        bad_cosine_mean=0.0,
        bad_sustain_fraction=0.5,
    )

    good, bad_local, semantic_mean, semantic_min = classify_gate_windows(
        cosine=cosine,
        heading_cost=heading,
        pathology=pathology,
        thresholds=thresholds,
    )

    assert good.tolist() == [True, False, False]
    assert bad_local.any(dim=1).tolist() == [False, True, True]
    torch.testing.assert_close(semantic_mean, torch.tensor([0.9, 0.9, -0.4]))
    torch.testing.assert_close(semantic_min, torch.tensor([0.9, 0.9, -0.4]))
