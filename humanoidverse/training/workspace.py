# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os

from humanoidverse.agents.envs.expert_motion_loader import (
    expert_buffer_cache_spec,
    find_compatible_expert_buffer_cache,
    load_expert_buffer_cache,
    load_expert_trajectories_from_motion_lib,
    save_expert_buffer_cache,
)
from humanoidverse.agents.envs.humanoidverse_mjlab import (
    RESET_REGION_NAMES,
    HumanoidVerseMjlabConfig,
)
from humanoidverse.agents.evaluations.humanoidverse_mjlab import (
    HumanoidVerseMjlabTrackingEvaluation,
    HumanoidVerseMjlabTrackingEvaluationConfig,
)
from humanoidverse.agents.evaluations.same_z_terrain import (
    SameZTerrainEvaluation,
    SameZTerrainEvaluationConfig,
)

os.environ["OMP_NUM_THREADS"] = "1"

import torch

torch.set_float32_matmul_precision("high")

import json
import time
import typing as tp
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import gymnasium
import numpy as np
import pydantic
import torch  # better to use scoped import if we use processes
import wandb
from packaging.version import Version
from torch.utils._pytree import tree_map
from tqdm import tqdm

from humanoidverse.agents.base import BaseConfig
from humanoidverse.agents.buffers.trajectory import TrajectoryDictBufferMultiDim
from humanoidverse.agents.buffers.transition import DictBuffer, dtype_numpytotorch_lower_precision
from humanoidverse.agents.fb.agent import RolloutContextState
from humanoidverse.agents.fb_cpr.agent import FBcprAgentConfig
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
from humanoidverse.agents.misc.loggers import CSVLogger
from humanoidverse.agents.tldr_dist_aux.agent import TldrDistAuxAgentConfig
from humanoidverse.agents.utils import EveryNStepsChecker, get_local_workdir, set_seed_everywhere
from humanoidverse.distributed import (
    all_gather_objects,
    barrier,
    broadcast_agent_state,
    broadcast_object,
    module_sync_report,
    reduce_metric_accumulators,
    sync_floating_buffers,
)
from humanoidverse.perception.instinct_direct_depth import RP1DirectDepthConfig

TRAIN_LOG_FILENAME = "train_log.txt"
REWARD_EVAL_LOG_FILENAME = "reward_eval_log.csv"
TRACKING_EVAL_LOG_FILENAME = "tracking_eval_log.csv"

CHECKPOINT_DIR_NAME = "checkpoint"


Evaluation = tp.Annotated[
    tp.Union[
        HumanoidVerseMjlabTrackingEvaluationConfig,
        SameZTerrainEvaluationConfig,
    ],
    pydantic.Field(discriminator="name"),
]

Agent = FBcprAgentConfig | FBcprAuxAgentConfig | TldrDistAuxAgentConfig


def make_flat_terrain_priority_eval_config(env_cfg: HumanoidVerseMjlabConfig) -> HumanoidVerseMjlabConfig:
    """Copy a terrain-aware MJLab config and select analytic fixed-flat observations."""
    overrides = list(env_cfg.hydra_overrides)
    terrain_preset = any(value == "terrain=terrain_ufo_v0" for value in overrides)
    terrain_mode_index = next(
        (index for index, value in enumerate(overrides) if value.startswith("terrain.terrain_type=")),
        None,
    )
    if not terrain_preset or terrain_mode_index is None:
        raise ValueError("fixed-flat priority evaluation requires the fb_terrain Hydra overrides")
    overrides[terrain_mode_index] = "terrain.terrain_type=plane"
    observation_mode_index = next(
        (index for index, value in enumerate(overrides) if value.startswith("terrain.terrain_priv.mode=")),
        None,
    )
    if observation_mode_index is None:
        overrides.append("terrain.terrain_priv.mode=flat_zero")
    else:
        overrides[observation_mode_index] = "terrain.terrain_priv.mode=flat_zero"
    return env_cfg.model_copy(update={"hydra_overrides": overrides, "evaluation_fast_path": True})


def make_canonical_plane_training_config(env_cfg: HumanoidVerseMjlabConfig) -> HumanoidVerseMjlabConfig:
    """Copy the main training config while changing only terrain geometry.

    Observation noise, physical/domain randomization, direct depth, reset yaw,
    and action delays remain exactly as configured for the main collector.
    """

    plane_cfg = make_flat_terrain_priority_eval_config(env_cfg)
    return plane_cfg.model_copy(update={"evaluation_fast_path": False})


def clone_motion_lib_for_collector(motion_lib, *, num_envs: int):
    """Create an independent runtime view over immutable loaded-FK tensors.

    A second MotionLib construction would redo expensive FK preprocessing and
    keep another tensor copy on the same GPU.  The plane collector only
    reads motion/FK tensors, so it can share those immutable storages while
    owning every mutable sampling field and all object-level bookkeeping.

    Formal 1024-env G1 runs have all 862 LaFAN clips loaded. Small smoke runs
    intentionally clone the main collector's smaller loaded subset so they
    preserve the same reset distribution without forcing full preprocessing.
    """

    if int(getattr(motion_lib, "_num_motions", 0)) <= 0:
        raise RuntimeError("Canonical-plane MotionLib cloning requires loaded FK motions")

    cloned = copy.copy(motion_lib)
    cloned.num_envs = int(num_envs)
    mutable_tensor_fields = (
        "_curr_motion_ids",
        "_termination_history",
        "_success_rate",
        "_sampling_history",
        "_sampling_prob",
        "_sampling_batch_prob",
    )
    for field in mutable_tensor_fields:
        value = getattr(motion_lib, field, None)
        if isinstance(value, torch.Tensor):
            setattr(cloned, field, value.clone())
    cloned.curr_motion_keys = list(motion_lib.curr_motion_keys)
    cloned._refresh_sampling_batch_prob()
    return cloned


def distributed_motion_ids(num_motions: int, rank: int, world_size: int) -> list[int]:
    if num_motions <= 0:
        raise ValueError("num_motions must be positive")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    return list(range(rank, num_motions, world_size))


def distributed_eval_num_envs(num_motions: int, world_size: int) -> int:
    """Return one shared per-rank capacity for a distributed motion evaluation."""
    if num_motions <= 0:
        raise ValueError("num_motions must be positive")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    return (num_motions + world_size - 1) // world_size


def merge_distributed_evaluation_results(
    shards: list[dict[str, dict[str, dict[str, tp.Any]]]],
) -> dict[str, dict[str, dict[str, tp.Any]]]:
    merged: dict[str, dict[str, dict[str, tp.Any]]] = {}
    motion_owners: dict[str, dict[int, int]] = {}
    for rank, shard in enumerate(shards):
        for evaluation_name, metrics in shard.items():
            evaluation_metrics = merged.setdefault(evaluation_name, {})
            evaluation_motion_owners = motion_owners.setdefault(evaluation_name, {})
            for metric_name, metric in metrics.items():
                if metric_name in evaluation_metrics:
                    raise RuntimeError(
                        f"Distributed evaluation produced duplicate metric {metric_name!r} "
                        f"for {evaluation_name!r}"
                    )
                motion_id = int(metric["motion_id"])
                if motion_id in evaluation_motion_owners:
                    raise RuntimeError(
                        f"Distributed evaluation produced motion_id={motion_id} on both "
                        f"rank {evaluation_motion_owners[motion_id]} and rank {rank}"
                    )
                evaluation_metrics[metric_name] = metric
                evaluation_motion_owners[motion_id] = rank

    for evaluation_name, metrics in merged.items():
        merged[evaluation_name] = dict(
            sorted(metrics.items(), key=lambda item: (int(item[1]["motion_id"]), item[0]))
        )
    return merged


def _trajectory_output_keys(agent: Agent) -> list[str]:
    keys = [
        "observation",
        "action",
        "z",
        "terminated",
        "truncated",
        "step_count",
        "reward",
    ]
    if getattr(agent, "aux_rewards", None):
        keys.append("aux_rewards")
    return keys


def _prior_trajectory_output_keys() -> list[str]:
    return [
        "observation",
        "action",
        "z",
        "episode_boundary",
        "transition_terminated",
        "transition_truncated",
        "step_count",
    ]


def _assert_canonical_plane_terrain_priv(observation: tp.Mapping[str, tp.Any], *, label: str) -> None:
    terrain_priv = observation.get("terrain_priv")
    if terrain_priv is None:
        raise RuntimeError(f"{label} is missing terrain_priv")
    if isinstance(terrain_priv, torch.Tensor):
        nonzero = int(torch.count_nonzero(terrain_priv).item())
    else:
        nonzero = int(np.count_nonzero(terrain_priv))
    if nonzero:
        raise RuntimeError(f"{label}.terrain_priv must be analytic zero, found {nonzero} nonzero values")


def _accumulate_metrics(
    total_metrics: dict[str, torch.Tensor] | None,
    metric_update_counts: dict[str, int],
    metrics: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    if total_metrics is None:
        total_metrics = {}
    for key, value in metrics.items():
        value = value.float()
        if key in total_metrics:
            total_metrics[key] = total_metrics[key] + value
        else:
            total_metrics[key] = value.clone()
        metric_update_counts[key] = metric_update_counts.get(key, 0) + 1
    return total_metrics, metric_update_counts


class TrainConfig(BaseConfig):
    # The "pydantic.Field" field is used to explicitely tell which field is the discriminative
    # feature
    agent: Agent = pydantic.Field(discriminator="name")
    motions: str | None = None
    motions_root: str | None = None

    env: HumanoidVerseMjlabConfig = pydantic.Field(discriminator="name")

    work_dir: str = pydantic.Field(default_factory=lambda: get_local_workdir("g1mujoco_train"))

    seed: int = 0
    online_parallel_envs: int = 50
    # Dedicated canonical-plane policy collector used only by D/Q_D. Zero
    # preserves the legacy single-replay behavior for non-depth presets.
    prior_plane_envs: int = 0
    # Note: this is in env steps (multiples of online_parallel_envs)
    log_every_updates: int = 100_000
    num_env_steps: int = 30_000_000
    # Note: this is in env steps (multiples of online_parallel_envs)
    update_agent_every: int = 500
    # Note: this is in env steps (multiples of online_parallel_envs)
    num_seed_steps: int = 50_000
    # Recovery-only policy rollout steps per rank. When resuming a checkpoint
    # without replay buffers, collect with the loaded policy and defer all
    # optimizer updates until this many new local env steps are available.
    resume_replay_warmup_steps: int = 0
    num_agent_updates: int = 50
    # Note: this is in env steps (multiples of online_parallel_envs)
    checkpoint_every_steps: int = 5_000_000
    checkpoint_buffer: bool = True
    prioritization: bool = False
    prioritization_min_val: float = 0.5
    prioritization_max_val: float = 5
    prioritization_scale: float = 2
    prioritization_mode: str = "bin"  # ["bin", "exp", "lin"]
    padding_beginning: int = 0
    padding_end: int = 0

    # Buffer
    use_trajectory_buffer: bool = False
    buffer_size: int = 5_000_000
    prior_buffer_size: int = 0

    # WANDB
    use_wandb: bool = False
    wandb_ename: str | None = None
    wandb_gname: str | None = None
    wandb_pname: str | None = None
    wandb_run_name: str | None = None

    # misc
    load_expert_data_from_motion_lib: bool = True
    buffer_device: str = "cpu"
    cache_expert_buffer: bool = True
    rebuild_expert_buffer_cache: bool = False
    expert_buffer_cache_root: str | None = None
    # Default to True; otherwise you will spam the console with tqdm
    disable_tqdm: bool = True
    log_torso_contact_forces: bool = True
    torso_contact_force_threshold: float = 1.0
    distributed_rank: int = 0
    distributed_world_size: int = 1
    rank0_only_writes: bool = True
    checkpoint_rank_buffers: bool = True
    distributed_sync: bool = True
    distributed_global_steps: bool = True
    distributed_average_metrics: bool = True
    distributed_gradient_sync: tp.Literal["manual", "ddp"] = "manual"
    ddp_bucket_cap_mb: float = 25.0
    fail_on_nonfinite: bool = True
    nonfinite_check_model_every_updates: int = 0
    nonfinite_check_rollout_every_local_steps: int = 0

    # If you want to add more available evaluations, Update "Evaluations" type above
    evaluations: Dict[str, Evaluation] | List[Evaluation] = pydantic.Field(default_factory=lambda: [])
    # Note: this is in env steps (multiples of online_parallel_envs)
    eval_every_steps: int = 1_000_000

    tags: dict = pydantic.Field(default_factory=lambda: {})

    infra: dict = pydantic.Field(default_factory=dict)

    def model_post_init(self, context):
        # TODO prioritization needs tracking eval to work, but this is bit hacky to check for it
        if self.load_expert_data_from_motion_lib and not isinstance(self.env, HumanoidVerseMjlabConfig):
            raise ValueError("Loading expert data from motion library is only supported for HumanoidVerseMjlabConfig")
        if self.ddp_bucket_cap_mb <= 0:
            raise ValueError("ddp_bucket_cap_mb must be positive")
        if self.prior_plane_envs < 0:
            raise ValueError("prior_plane_envs must be non-negative")
        if self.prior_buffer_size < 0:
            raise ValueError("prior_buffer_size must be non-negative")
        if self.prior_plane_envs > 0:
            if not isinstance(self.env, HumanoidVerseMjlabConfig):
                raise ValueError("canonical-plane prior collection requires HumanoidVerseMjlabConfig")
            if self.tags.get("agent") != "fb_depth":
                raise ValueError("canonical-plane prior collection is currently defined only for fb_depth")
            if not self.use_trajectory_buffer:
                raise ValueError("canonical-plane direct-depth collection requires trajectory replay")
            if self.prior_buffer_size < self.prior_plane_envs * 2:
                raise ValueError("prior_buffer_size must hold at least two steps per plane environment")
        if self.distributed_gradient_sync == "ddp" and not self.distributed_sync:
            raise ValueError("distributed_gradient_sync='ddp' requires distributed_sync=True")

        if self.prioritization:
            has_prioritization_eval = False
            for eval_type in self.evaluations:
                if isinstance(eval_type, HumanoidVerseMjlabTrackingEvaluationConfig):
                    has_prioritization_eval = True
                    break
            if not has_prioritization_eval:
                raise ValueError("Prioritization requires tracking evaluation to be enabled")


        if self.motions is None or self.motions_root is None:
            if self.prioritization:
                raise ValueError("Prioritization requires expert data to be provided (motions and motions_root)")
            elif self.agent == FBcprAgentConfig:
                # TODO how to do checks like these in pydantic or more systematically?
                raise ValueError("FBcprAgent requires expert data to be provided (motions and motions_root)")

        # Ensure all evaluations have unique log names
        if isinstance(self.evaluations, list):
            log_names = set()
            for eval_cfg in self.evaluations:
                if eval_cfg.name_in_logs in log_names:
                    raise ValueError(
                        f"Duplicate evaluation name_in_logs found: {eval_cfg.name}. These should be unique so we do not overwrite any logs"
                    )
                log_names.add(eval_cfg.name_in_logs)

    def build(self):
        return Workspace(self)


def create_agent_or_load_checkpoint(work_dir: Path, cfg: TrainConfig, agent_build_kwargs: dict[str, tp.Any]):
    checkpoint_dir = work_dir / CHECKPOINT_DIR_NAME
    train_status_path = checkpoint_dir / "train_status.json"
    checkpoint_status = _initial_train_status(cfg)
    if train_status_path.exists():
        with train_status_path.open("r") as f:
            train_status = json.load(f)
        checkpoint_status = _normalize_train_status(train_status, cfg)

        print(
            f"Loading the agent at local_time={checkpoint_status['local_time']} "
            f"global_time={checkpoint_status['global_time']} optimizer_steps={checkpoint_status['optimizer_steps']}"
        )
        agent = cfg.agent.object_class.load(checkpoint_dir, device=cfg.agent.model.device)
    else:
        agent = cfg.agent.build(**agent_build_kwargs)
    return agent, cfg, checkpoint_status


def _global_step_scale(cfg: TrainConfig) -> int:
    if cfg.distributed_sync and cfg.distributed_global_steps and int(cfg.distributed_world_size) > 1:
        return int(cfg.distributed_world_size)
    return 1


def _effective_batch_size(cfg: TrainConfig) -> int:
    return int(cfg.agent.train.batch_size) * _global_step_scale(cfg)


def _num_envs_per_rank(cfg: TrainConfig) -> int:
    return int(cfg.online_parallel_envs)


def _global_parallel_envs(cfg: TrainConfig) -> int:
    return _num_envs_per_rank(cfg) * _global_step_scale(cfg)


def _replay_capacity_per_rank(cfg: TrainConfig) -> int:
    return int(cfg.buffer_size)


def _effective_replay_capacity(cfg: TrainConfig) -> int:
    return _replay_capacity_per_rank(cfg) * _global_step_scale(cfg)


def _trajectory_steps_per_rank(cfg: TrainConfig) -> int:
    if cfg.use_trajectory_buffer:
        return int(cfg.buffer_size) // max(_num_envs_per_rank(cfg), 1)
    return int(cfg.buffer_size)


def _tensor_nonfinite_summary(value: tp.Any) -> str | None:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0 or not torch.is_floating_point(value):
            return None
        finite = torch.isfinite(value)
        if bool(finite.all().item()):
            return None
        bad = ~finite
        bad_count = int(bad.sum().item())
        with torch.no_grad():
            finite_values = value[finite]
            finite_min = float(finite_values.min().item()) if finite_values.numel() > 0 else None
            finite_max = float(finite_values.max().item()) if finite_values.numel() > 0 else None
            bad_indices = bad.nonzero(as_tuple=False)[:5].detach().cpu().tolist()
            bad_values = value[bad][:5].detach().cpu().tolist()
        return (
            f"type=torch shape={tuple(value.shape)} dtype={value.dtype} bad_count={bad_count} "
            f"finite_min={finite_min} finite_max={finite_max} bad_indices={bad_indices} bad_values={bad_values}"
        )
    if isinstance(value, np.ndarray):
        if value.size == 0 or not np.issubdtype(value.dtype, np.floating):
            return None
        finite = np.isfinite(value)
        if bool(finite.all()):
            return None
        bad = ~finite
        bad_count = int(bad.sum())
        finite_values = value[finite]
        finite_min = float(finite_values.min()) if finite_values.size > 0 else None
        finite_max = float(finite_values.max()) if finite_values.size > 0 else None
        bad_indices = np.argwhere(bad)[:5].tolist()
        bad_values = value[bad][:5].tolist()
        return (
            f"type=numpy shape={value.shape} dtype={value.dtype} bad_count={bad_count} "
            f"finite_min={finite_min} finite_max={finite_max} bad_indices={bad_indices} bad_values={bad_values}"
        )
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value):
            return None
        return f"type=scalar value={value}"
    return None


def _iter_nonfinite(value: tp.Any, prefix: str = "") -> tp.Iterator[tuple[str, str]]:
    summary = _tensor_nonfinite_summary(value)
    if summary is not None:
        yield prefix or "<root>", summary
        return
    if isinstance(value, dict):
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_nonfinite(item, child_prefix)
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            child_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            yield from _iter_nonfinite(item, child_prefix)


def _assert_finite(value: tp.Any, *, label: str, rank: int, local_time: int, global_time: int, optimizer_steps: int) -> None:
    bad = list(_iter_nonfinite(value, label))
    if not bad:
        return
    details = "\n".join(f"  - {path}: {summary}" for path, summary in bad[:20])
    more = "" if len(bad) <= 20 else f"\n  ... {len(bad) - 20} more non-finite fields"
    raise FloatingPointError(
        "Non-finite value detected "
        f"(rank={rank}, local_time={local_time}, global_time={global_time}, optimizer_steps={optimizer_steps}):\n"
        f"{details}{more}"
    )


def _assert_model_finite(module: torch.nn.Module, *, rank: int, local_time: int, global_time: int, optimizer_steps: int) -> None:
    for name, param in module.named_parameters():
        _assert_finite(
            param,
            label=f"model.parameters.{name}",
            rank=rank,
            local_time=local_time,
            global_time=global_time,
            optimizer_steps=optimizer_steps,
        )
    for name, buffer in module.named_buffers():
        _assert_finite(
            buffer,
            label=f"model.buffers.{name}",
            rank=rank,
            local_time=local_time,
            global_time=global_time,
            optimizer_steps=optimizer_steps,
        )


def _distributed_loss_mode(cfg: TrainConfig) -> str:
    if cfg.distributed_sync and int(cfg.distributed_world_size) > 1:
        return "local_loss_average"
    return "single_rank"


def _initial_train_status(cfg: TrainConfig) -> dict[str, tp.Any]:
    return {
        "local_time": 0,
        "global_time": 0,
        "optimizer_steps": 0,
        "world_size": int(cfg.distributed_world_size),
        "loss_mode": _distributed_loss_mode(cfg),
        "effective_batch_size": _effective_batch_size(cfg),
    }


def _normalize_train_status(train_status: dict[str, tp.Any], cfg: TrainConfig) -> dict[str, tp.Any]:
    status = _initial_train_status(cfg)
    scale = _global_step_scale(cfg)
    current_world_size = int(cfg.distributed_world_size)
    if "world_size" not in train_status and current_world_size > 1:
        raise RuntimeError(
            "Cannot safely resume this checkpoint because checkpoint/train_status.json does not record world_size. "
            "Distributed checkpoints contain rank-local replay buffer shards; start a fresh work_dir or migrate buffers explicitly."
        )
    checkpoint_world_size = int(train_status.get("world_size", current_world_size))
    if checkpoint_world_size != current_world_size:
        raise RuntimeError(
            "Cannot resume checkpoint with a different distributed world_size: "
            f"checkpoint world_size={checkpoint_world_size}, current world_size={current_world_size}. "
            "Rank-local replay buffer shards cannot be automatically migrated; use a matching GPU count, "
            "start a fresh work_dir, or perform an explicit buffer migration."
        )
    if "local_time" in train_status:
        status["local_time"] = int(train_status["local_time"])
        status["global_time"] = int(train_status.get("global_time", status["local_time"] * scale))
    else:
        legacy_time = int(train_status.get("time", 0))
        status["global_time"] = legacy_time
        status["local_time"] = (legacy_time + scale - 1) // scale
    status["optimizer_steps"] = int(train_status.get("optimizer_steps", 0))
    status["world_size"] = checkpoint_world_size
    status["loss_mode"] = str(train_status.get("loss_mode", _distributed_loss_mode(cfg)))
    status["effective_batch_size"] = int(train_status.get("effective_batch_size", _effective_batch_size(cfg)))
    return status


def _make_train_status(cfg: TrainConfig, *, local_time: int, global_time: int, optimizer_steps: int) -> dict[str, tp.Any]:
    return {
        "time": int(global_time),
        "local_time": int(local_time),
        "global_time": int(global_time),
        "optimizer_steps": int(optimizer_steps),
        "world_size": int(cfg.distributed_world_size),
        "loss_mode": _distributed_loss_mode(cfg),
        "effective_batch_size": _effective_batch_size(cfg),
    }


def init_wandb(cfg: TrainConfig):
    from pathlib import Path
    exp_name = cfg.wandb_run_name if cfg.wandb_run_name else Path(cfg.work_dir).name
    wandb_config = cfg.model_dump()
    wandb.init(entity=cfg.wandb_ename, project=cfg.wandb_pname, group=cfg.wandb_gname, name=exp_name, config=wandb_config, dir="./_wandb")


@dataclass
class _PriorCollectorState:
    td: dict[str, tp.Any]
    info: dict[str, tp.Any]
    terminated: np.ndarray
    truncated: np.ndarray
    rollout: RolloutContextState


class Workspace:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.distributed_rank = int(self.cfg.distributed_rank)
        self.distributed_world_size = int(self.cfg.distributed_world_size)
        self._write_shared_artifacts = (not self.cfg.rank0_only_writes) or self.distributed_rank == 0

        # MJLab environments are created once and shared with evaluation.
        if isinstance(cfg.env, HumanoidVerseMjlabConfig):
            from omegaconf import OmegaConf

            self.train_env, self.train_env_info = cfg.env.build(num_envs=cfg.online_parallel_envs)
            self.obs_space = self.train_env.single_observation_space
            self.action_space = self.train_env.single_action_space
            self.prior_env = None
            self.prior_env_info = None
            if cfg.prior_plane_envs > 0:
                prior_cfg = make_canonical_plane_training_config(cfg.env)
                # Both collectors are live at the same time. Give the plane
                # collector an independent sampling/runtime view while sharing
                # only read-only full-FK tensor storage.
                prior_motion_lib = clone_motion_lib_for_collector(
                    self.train_env._env._motion_lib,
                    num_envs=cfg.prior_plane_envs,
                )
                self.prior_env, self.prior_env_info = prior_cfg.build(
                    num_envs=cfg.prior_plane_envs,
                    motion_lib=prior_motion_lib,
                )
                if self.prior_env.single_observation_space != self.obs_space:
                    raise RuntimeError("Canonical-plane and main collectors produced different observation spaces")
                if self.prior_env.single_action_space != self.action_space:
                    raise RuntimeError("Canonical-plane and main collectors produced different action spaces")
                print(
                    "[INFO] Built canonical-plane training collector: "
                    f"num_envs={cfg.prior_plane_envs}, terrain=plane, terrain_priv=flat_zero, "
                    "evaluation_fast_path=False, training_DR=preserved",
                    flush=True,
                )
        else:
            sample_env, _ = cfg.env.build(num_envs=1)
            self.obs_space = sample_env.observation_space
            self.action_space = sample_env.action_space
            self.prior_env = None
            self.prior_env_info = None

        assert "time" in self.obs_space.keys(), "Observation space must contain 'obs' and 'time' (TimeAwareObservation wrapper)"
        assert len(self.action_space.shape) == 1, "Only 1D action space is supported (first dim should be vector env)"
        # TODO for backwards consistency, we do not pass "time" to the agent, so we remove it from the obs_space we pass to the agent/model
        #      but would we need it at some point?
        del self.obs_space.spaces["time"]

        self.action_dim = self.action_space.shape[0]

        print(f"Workdir: {self.cfg.work_dir}")
        self.work_dir = Path(self.cfg.work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)

        if self._write_shared_artifacts and isinstance(cfg.env, HumanoidVerseMjlabConfig):
            with open(self.work_dir / "config.yaml", "w") as file:
                OmegaConf.save(self.train_env_info["unresolved_conf"], file)

        self.train_logger = CSVLogger(filename=self.work_dir / TRAIN_LOG_FILENAME) if self._write_shared_artifacts else None

        set_seed_everywhere(self.cfg.seed)

        self.agent, self.cfg, self._checkpoint_status = create_agent_or_load_checkpoint(
            self.work_dir, self.cfg, agent_build_kwargs=dict(obs_space=self.obs_space, action_dim=self.action_dim)
        )
        self._checkpoint_local_time = int(self._checkpoint_status["local_time"])
        self._checkpoint_global_time = int(self._checkpoint_status["global_time"])
        self._optimizer_steps = int(self._checkpoint_status["optimizer_steps"])
        if self.cfg.distributed_sync:
            broadcast_agent_state(self.agent, src=0)
            if self.cfg.distributed_gradient_sync == "ddp":
                enable_ddp = getattr(self.agent, "enable_distributed_gradient_sync", None)
                if not callable(enable_ddp):
                    raise RuntimeError(
                        f"Agent {type(self.agent).__name__} does not support DDP gradient overlap; "
                        "use distributed_gradient_sync='manual'."
                    )
                enable_ddp(bucket_cap_mb=self.cfg.ddp_bucket_cap_mb)
                if self._write_shared_artifacts:
                    print(
                        "[INFO] Gradient synchronization: "
                        f"mode=ddp bucket_cap_mb={self.cfg.ddp_bucket_cap_mb:g} "
                        "gradient_as_bucket_view=True",
                        flush=True,
                    )
        self.agent._model.train()

        if isinstance(self.cfg.evaluations, list):
            self.evaluations = {eval_cfg.name_in_logs: eval_cfg.build() for eval_cfg in self.cfg.evaluations}
        else:
            self.evaluations = {eval_cfg: eval_cfg.build() for name, eval_cfg in self.cfg.evaluations.items()}
        self.evaluate = len(self.evaluations) > 0

        self.eval_loggers = {
            name: CSVLogger(filename=self.work_dir / f"{name}.csv") for name in self.evaluations.keys()
        } if self._write_shared_artifacts else {}

        if self._write_shared_artifacts and self.cfg.use_wandb:
            init_wandb(self.cfg)

        if self._write_shared_artifacts:
            with (self.work_dir / "config.json").open("w") as f:
                f.write(self.cfg.model_dump_json(indent=4))

        self.priorization_eval_name = None
        self._priority_eval_env = None
        if self.cfg.prioritization:
            for name, evaluation in self.evaluations.items():
                if isinstance(evaluation.cfg, HumanoidVerseMjlabTrackingEvaluationConfig):
                    self.priorization_eval_name = name
                    break
            if self.priorization_eval_name is None:
                raise ValueError("Prioritization requires tracking evaluation to be enabled")

        self.training_with_expert_data = True

        self.manager = None

    def _checkpoint_buffer_path(self, checkpoint_dir: Path, name: str = "train") -> Path:
        if name not in {"train", "prior"}:
            raise ValueError(f"Unsupported replay buffer name: {name!r}")
        if self.cfg.checkpoint_rank_buffers and self.distributed_world_size > 1:
            return checkpoint_dir / "buffers" / f"{name}_rank_{self.distributed_rank}"
        return checkpoint_dir / "buffers" / name

    def train(self):
        self.start_time = time.time()
        try:
            self.train_online()
        finally:
            self._close_priority_eval_env()
            self._close_prior_env()

    def _close_prior_env(self) -> None:
        if self.prior_env is not None:
            self.prior_env.close()
            self.prior_env = None

    def _reset_prior_collector(self) -> _PriorCollectorState:
        if self.prior_env is None:
            raise RuntimeError("Canonical-plane collector is not enabled")
        td, info = self.prior_env.reset()
        _assert_canonical_plane_terrain_priv(td, label="prior.reset.obs")
        zeros = np.zeros(self.cfg.prior_plane_envs, dtype=bool)
        # The reset observation must not be joined to the preceding replay
        # trajectory (startup, resume, and shared-motion-lib evaluation all
        # pass through this path).  Marking the current slot as an episode
        # boundary also makes compact depth reconstruction fill from the new
        # reset frame instead of reading pre-reset history.
        boundary = np.ones(self.cfg.prior_plane_envs, dtype=bool)
        return _PriorCollectorState(
            td=td,
            info=info,
            terminated=boundary,
            truncated=zeros.copy(),
            rollout=RolloutContextState(),
        )

    def _advance_rollout_state(
        self,
        state: RolloutContextState,
        step_count: torch.Tensor,
        replay_buffer: dict[str, tp.Any],
    ) -> RolloutContextState:
        advance = getattr(self.agent, "advance_rollout_context", None)
        if callable(advance):
            return advance(state, step_count, replay_buffer)
        z = self.agent.maybe_update_rollout_context(
            z=state.z,
            step_count=step_count,
            replay_buffer=replay_buffer,
        )
        return RolloutContextState(z=z)

    def _step_prior_collector(
        self,
        state: _PriorCollectorState,
        *,
        replay_buffer: dict[str, tp.Any],
        local_time: int,
        global_time: int,
        optimizer_steps: int,
    ) -> tuple[_PriorCollectorState, float]:
        if self.prior_env is None:
            raise RuntimeError("Canonical-plane collector is not enabled")
        collection_start = time.perf_counter()
        with torch.no_grad():
            obs = tree_map(
                lambda x: torch.tensor(
                    x,
                    dtype=dtype_numpytotorch_lower_precision(x.dtype),
                    device=self.agent.device,
                ),
                state.td,
            )
            step_count = obs.pop("time")
            state.rollout = self._advance_rollout_state(
                state.rollout,
                step_count,
                replay_buffer,
            )
            if state.rollout.z is None:
                raise RuntimeError("Canonical-plane rollout did not produce z")
            if local_time < self.cfg.num_seed_steps:
                action = self.prior_env.action_space.sample().astype(np.float32)
            else:
                action = self.agent.act(obs=obs, z=state.rollout.z, mean=False)
                action = action.cpu().detach().numpy()

        new_td, _new_reward, new_terminated, new_truncated, new_info = self.prior_env.step(action)
        _assert_canonical_plane_terrain_priv(new_td, label="prior.step.obs")
        if self.cfg.fail_on_nonfinite:
            _assert_finite(
                new_td,
                label="prior.env.step.obs",
                rank=self.distributed_rank,
                local_time=local_time,
                global_time=global_time,
                optimizer_steps=optimizer_steps,
            )

        episode_boundary = np.logical_or(state.terminated, state.truncated)
        data = {
            "observation": tree_map(lambda x: x[None, ...], obs),
            "action": action[None, ...],
            "z": state.rollout.z[None, ...],
            "episode_boundary": episode_boundary[None, ..., None],
            "transition_terminated": new_terminated[None, ..., None],
            "transition_truncated": new_truncated[None, ..., None],
            "step_count": step_count[None, ..., None],
        }
        data["observation"].pop("history", None)
        replay_buffer["prior"].extend(data)
        return (
            _PriorCollectorState(
                td=new_td,
                info=new_info,
                terminated=new_terminated,
                truncated=new_truncated,
                rollout=state.rollout,
            ),
            time.perf_counter() - collection_start,
        )

    def _uses_fixed_flat_priority_eval(self, evaluation_name: str) -> bool:
        return (
            self.cfg.prioritization
            and self.cfg.tags.get("agent") in {"fb_terrain", "fb_depth"}
            and evaluation_name == self.priorization_eval_name
        )

    def _get_priority_eval_env(self):
        if self._priority_eval_env is None:
            eval_cfg = make_flat_terrain_priority_eval_config(self.cfg.env)
            eval_num_envs = int(self.cfg.online_parallel_envs)
            local_motion_count = None
            if self.cfg.distributed_sync and self.distributed_world_size > 1:
                num_motions = int(self.train_env._env._motion_lib._num_unique_motions)
                local_motion_count = len(
                    distributed_motion_ids(num_motions, self.distributed_rank, self.distributed_world_size)
                )
                # Every rank must derive balanced_motion_chunks() with the same
                # capacity.  Allocating only the local shard size made the last
                # ranks use a smaller chunk size whenever the motion count was
                # not divisible by world_size, producing overlapping/missing
                # motion IDs at the final gather.
                eval_num_envs = distributed_eval_num_envs(num_motions, self.distributed_world_size)
            self._priority_eval_env, _ = eval_cfg.build(
                num_envs=eval_num_envs,
                motion_lib=self.train_env._env._motion_lib,
            )
            tags = getattr(self.cfg, "tags", {})
            print(
                "[INFO] Built persistent terrain-aware priority evaluation environment: "
                f"terrain=plane, terrain_height_observation=flat_zero, "
                f"direct_depth_sensor={'on' if tags.get('agent') == 'fb_depth' else 'off'}, "
                f"num_envs={eval_num_envs}, rank={self.distributed_rank}, "
                f"motion_shard_size={local_motion_count}",
                flush=True,
            )
        return self._priority_eval_env

    def _close_priority_eval_env(self) -> None:
        if self._priority_eval_env is not None:
            self._priority_eval_env.close()
            self._priority_eval_env = None

    def _get_torso_contact_force_metrics(self, train_env) -> dict[str, float]:
        if not self.cfg.log_torso_contact_forces:
            return {}

        raw_env = getattr(train_env, "_env", train_env)
        simulator = getattr(raw_env, "simulator", None)
        torso_index = getattr(raw_env, "torso_index", None)
        if simulator is None or torso_index is None:
            return {}

        with torch.no_grad():
            torso_force = simulator.contact_forces[:, torso_index, :].float()
            torso_norm = torch.linalg.vector_norm(torso_force, dim=-1)
            torso_mean = torso_force.mean(dim=0)
            torso_force_z = torso_force[:, 2]
            torso_z = simulator._rigid_body_pos[:, torso_index, 2].float()
            contact_mask = torso_norm > self.cfg.torso_contact_force_threshold
            contact_count = contact_mask.sum()
            num_envs = max(torso_norm.numel(), 1)

            metrics = {
                "torso_contact_force_mean": np.round(torso_norm.mean().item(), 6),
                "torso_contact_force_max": np.round(torso_norm.max().item(), 6),
                "torso_contact_force_p95": np.round(torch.quantile(torso_norm, 0.95).item(), 6),
                "torso_contact_force_contact_count": int(contact_count.item()),
                "torso_contact_force_contact_frac": np.round((contact_count.float() / num_envs).item(), 6),
                "torso_contact_force_mean_x": np.round(torso_mean[0].item(), 6),
                "torso_contact_force_mean_y": np.round(torso_mean[1].item(), 6),
                "torso_contact_force_mean_z": np.round(torso_mean[2].item(), 6),
                "torso_contact_force_z_mean": np.round(torso_force_z.mean().item(), 6),
                "torso_contact_force_z_max": np.round(torso_force_z.max().item(), 6),
                "torso_contact_force_z_p95": np.round(torch.quantile(torso_force_z, 0.95).item(), 6),
                "torso_z_mean": np.round(torso_z.mean().item(), 6),
                "torso_z_min": np.round(torso_z.min().item(), 6),
                "torso_z_p05": np.round(torch.quantile(torso_z, 0.05).item(), 6),
            }
            if contact_count.item() > 0:
                torso_z_contact = torso_z[contact_mask]
                metrics.update(
                    {
                        "torso_z_contact_mean": np.round(torso_z_contact.mean().item(), 6),
                        "torso_z_contact_min": np.round(torso_z_contact.min().item(), 6),
                        "torso_z_contact_p05": np.round(torch.quantile(torso_z_contact, 0.05).item(), 6),
                    }
                )
            else:
                metrics.update(
                    {
                        "torso_z_contact_mean": 0.0,
                        "torso_z_contact_min": 0.0,
                        "torso_z_contact_p05": 0.0,
                    }
                )
            return metrics

    def _get_terrain_metrics(self, train_env) -> dict[str, float]:
        raw_env = getattr(train_env, "_env", train_env)
        if not getattr(raw_env, "terrain_enabled", False):
            return {}
        with torch.no_grad():
            root_clearance, _terrain_actor, _terrain_priv = raw_env._terrain_observations()
            pelvis_world_z = raw_env.body_pos[:, 0, 2]
            root_clearance = root_clearance[:, 0]
            local_ground_z = pelvis_world_z - root_clearance
            # Once tiles are connected, the assigned spawn column no longer
            # describes the physical terrain currently under the robot.
            terrain_ids = raw_env._current_terrain_type_ids()
            metrics: dict[str, float] = {}
            for terrain_id, name in enumerate(raw_env.terrain_component_names):
                mask = terrain_ids == terrain_id
                if not torch.any(mask):
                    continue
                for metric_name, values in (
                    ("pelvis_world_z", pelvis_world_z),
                    ("local_ground_z", local_ground_z),
                    ("pelvis_clearance", root_clearance),
                ):
                    selected = values[mask]
                    prefix = f"terrain/{name}/{metric_name}"
                    metrics[f"{prefix}_mean"] = np.round(selected.mean().item(), 6)
                    metrics[f"{prefix}_min"] = np.round(selected.min().item(), 6)
                    metrics[f"{prefix}_p05"] = np.round(torch.quantile(selected, 0.05).item(), 6)

            boundary_margin = raw_env._terrain_boundary_margin()
            if boundary_margin is not None:
                boundary_min = boundary_margin.min()
                boundary_historical_min = raw_env._terrain_boundary_min.clone()
                boundary_violation_count = raw_env._terrain_boundary_violation_count.clone()
                if self.cfg.distributed_sync and self.distributed_world_size > 1:
                    torch.distributed.all_reduce(boundary_min, op=torch.distributed.ReduceOp.MIN)
                    torch.distributed.all_reduce(boundary_historical_min, op=torch.distributed.ReduceOp.MIN)
                    torch.distributed.all_reduce(boundary_violation_count, op=torch.distributed.ReduceOp.SUM)
                metrics["terrain/boundary_margin_min"] = np.round(boundary_min.item(), 6)
                metrics["terrain/boundary_margin_p05"] = np.round(torch.quantile(boundary_margin, 0.05).item(), 6)
                metrics["terrain/boundary_margin_historical_min"] = np.round(
                    boundary_historical_min.item(), 6
                )
                metrics["terrain/boundary_required_margin"] = np.round(
                    raw_env._terrain_boundary_required, 6
                )
                metrics["terrain/boundary_violation_count"] = int(boundary_violation_count.item())

            tile_crossings = raw_env._terrain_tile_crossing_count.clone()
            transition_counts = raw_env._terrain_transition_counts.clone()
            reset_region_counts = raw_env._reset_region_counts.clone()
            lie_down_resets = raw_env._lie_down_reset_count.clone()
            if self.cfg.distributed_sync and self.distributed_world_size > 1:
                torch.distributed.all_reduce(tile_crossings, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(transition_counts, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(reset_region_counts, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(lie_down_resets, op=torch.distributed.ReduceOp.SUM)
            metrics["terrain/tile_crossing_count"] = int(tile_crossings.item())
            for source_id, source_name in enumerate(raw_env.terrain_component_names):
                for target_id, target_name in enumerate(raw_env.terrain_component_names):
                    count = int(transition_counts[source_id, target_id].item())
                    if count:
                        metrics[f"terrain/transition/{source_name}_to_{target_name}"] = count
            for region_id, region_name in enumerate(RESET_REGION_NAMES):
                metrics[f"reset/{region_name}"] = int(reset_region_counts[region_id].item())
            metrics["reset/lie_down"] = int(lie_down_resets.item())
            return metrics

    def _load_expert_buffer(self):
        if not self.training_with_expert_data:
            return None
        if not self.cfg.load_expert_data_from_motion_lib:
            raise RuntimeError(
                "This MJLab-focused build only supports expert data loaded from the motion library. "
                "Set load_expert_data_from_motion_lib=True."
            )

        started_at = time.time()
        if self.cfg.cache_expert_buffer:
            expert_buffer = self._load_or_build_cached_expert_buffer()
        else:
            if self._write_shared_artifacts:
                print(f"[INFO] Building expert motion buffer on {self.cfg.buffer_device}; cache disabled", flush=True)
            expert_buffer = load_expert_trajectories_from_motion_lib(
                self.train_env._env,
                self.cfg.agent,
                device=self.cfg.buffer_device,
            )
        if self._write_shared_artifacts:
            print(
                f"[INFO] Expert motion buffer ready: motions={len(expert_buffer.motion_ids)} "
                f"frames={len(expert_buffer)} elapsed={time.time() - started_at:.1f}s",
                flush=True,
            )
        return expert_buffer

    def _load_or_build_cached_expert_buffer(self):
        cache_dir, fingerprint, metadata = expert_buffer_cache_spec(
            self.train_env._env,
            self.cfg.agent,
            cache_root=self.cfg.expert_buffer_cache_root,
        )
        existing = None
        if not self.cfg.rebuild_expert_buffer_cache:
            existing = find_compatible_expert_buffer_cache(cache_dir, fingerprint, metadata)
        if existing is not None:
            cache_dir, fingerprint = existing

        expert_buffer = None
        rank0_builder = self.distributed_rank == 0 or not self.cfg.distributed_sync
        if rank0_builder:
            if existing is not None:
                if self._write_shared_artifacts:
                    print(f"[INFO] Loading cached expert buffer: {cache_dir}", flush=True)
                try:
                    expert_buffer = load_expert_buffer_cache(
                        self.train_env._env,
                        cache_dir,
                        fingerprint,
                        device=self.cfg.buffer_device,
                    )
                except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
                    if self._write_shared_artifacts:
                        print(f"[WARN] Expert buffer cache could not be loaded; rebuilding: {exc}", flush=True)
                    existing = None
            if existing is None:
                if self._write_shared_artifacts:
                    print(
                        f"[INFO] Rank 0 building expert buffer and full-FK cache: {cache_dir}",
                        flush=True,
                    )
                expert_buffer = load_expert_trajectories_from_motion_lib(
                    self.train_env._env,
                    self.cfg.agent,
                    device=self.cfg.buffer_device,
                )
                save_expert_buffer_cache(
                    self.train_env._env,
                    expert_buffer,
                    cache_dir,
                    fingerprint,
                    metadata,
                )
                if self._write_shared_artifacts:
                    cache_bytes = sum(
                        path.stat().st_size
                        for path in (cache_dir / "expert_buffer.pt", cache_dir / "motion_lib_fk.pt")
                    )
                    print(
                        f"[INFO] Expert/full-FK cache atomically published: {cache_dir} "
                        f"size={cache_bytes / (1024**3):.2f} GiB",
                        flush=True,
                    )

        if self.cfg.distributed_sync:
            selection = broadcast_object(
                {"cache_dir": str(cache_dir), "fingerprint": fingerprint},
                src=0,
            )
            cache_dir = Path(selection["cache_dir"])
            fingerprint = str(selection["fingerprint"])
            barrier()
        if self.distributed_rank != 0 and self.cfg.distributed_sync:
            expert_buffer = load_expert_buffer_cache(
                self.train_env._env,
                cache_dir,
                fingerprint,
                device=self.cfg.buffer_device,
            )
        if expert_buffer is None:
            raise RuntimeError("Expert buffer cache initialization did not produce a buffer")
        return expert_buffer

    def train_online(self) -> None:
        expert_buffer = self._load_expert_buffer()
        print("Creating the training environment")

        if isinstance(self.cfg.env, HumanoidVerseMjlabConfig):
            train_env = self.train_env
            train_env_info = self.train_env_info
        else:
            train_env, train_env_info = self.cfg.env.build(num_envs=self.cfg.online_parallel_envs)

        print("Allocating buffers")
        replay_buffer = {}
        checkpoint_dir = self.work_dir / CHECKPOINT_DIR_NAME
        checkpoint_buffer_dir = self._checkpoint_buffer_path(checkpoint_dir, "train")
        loaded_checkpoint_buffer = checkpoint_buffer_dir.exists()
        loaded_prior_checkpoint_buffer = False
        if checkpoint_buffer_dir.exists():
            print("Loading checkpointed buffer")
            if self.cfg.use_trajectory_buffer:
                replay_buffer["train"] = TrajectoryDictBufferMultiDim.load(checkpoint_buffer_dir, device=self.cfg.buffer_device)
            else:
                replay_buffer["train"] = DictBuffer.load(checkpoint_buffer_dir, device=self.cfg.buffer_device)
            print(f"Loaded buffer of size {len(replay_buffer['train'])}")
        else:
            if self.cfg.use_trajectory_buffer:
                compact_depth_history = self.cfg.tags.get("agent") == "fb_depth"
                replay_buffer["train"] = TrajectoryDictBufferMultiDim(
                    capacity=self.cfg.buffer_size // self.cfg.online_parallel_envs,  # make sure to divide by num_envs
                    device=self.cfg.buffer_device,
                    n_dim=2,
                    end_key="truncated",
                    output_key_t=_trajectory_output_keys(self.cfg.agent),
                    output_key_tp1=["observation", "terminated"],
                    compact_depth_history=compact_depth_history,
                    depth_history_offsets=RP1DirectDepthConfig().sampled_ages,
                )
                if compact_depth_history:
                    print(
                        "[INFO] Compact RP1 depth replay enabled: storing one uint8 frame per transition "
                        "and reconstructing the 8-frame history when sampling",
                        flush=True,
                    )
            else:
                replay_buffer["train"] = DictBuffer(capacity=self.cfg.buffer_size, device=self.cfg.buffer_device)
        if self.prior_env is not None:
            prior_checkpoint_buffer_dir = self._checkpoint_buffer_path(checkpoint_dir, "prior")
            if prior_checkpoint_buffer_dir.exists():
                loaded_prior_checkpoint_buffer = True
                replay_buffer["prior"] = TrajectoryDictBufferMultiDim.load(
                    prior_checkpoint_buffer_dir,
                    device=self.cfg.buffer_device,
                )
                prior_env_axis = int(replay_buffer["prior"].storage[replay_buffer["prior"].end_key].shape[1])
                if prior_env_axis != self.cfg.prior_plane_envs:
                    raise RuntimeError(
                        "Checkpointed canonical-plane replay env axis does not match the configured collector: "
                        f"buffer={prior_env_axis} configured={self.cfg.prior_plane_envs}"
                    )
                print(f"Loaded canonical-plane buffer of size {len(replay_buffer['prior'])}")
            else:
                replay_buffer["prior"] = TrajectoryDictBufferMultiDim(
                    capacity=self.cfg.prior_buffer_size // self.cfg.prior_plane_envs,
                    device=self.cfg.buffer_device,
                    n_dim=2,
                    end_key="episode_boundary",
                    output_key_t=_prior_trajectory_output_keys(),
                    output_key_tp1=["observation"],
                    compact_depth_history=True,
                    depth_history_offsets=RP1DirectDepthConfig().sampled_ages,
                )
                print(
                    "[INFO] Compact canonical-plane replay enabled: "
                    f"envs={self.cfg.prior_plane_envs}, capacity={self.cfg.prior_buffer_size}, "
                    f"time_slots={self.cfg.prior_buffer_size // self.cfg.prior_plane_envs}",
                    flush=True,
                )
            prior_sizes = all_gather_objects(int(replay_buffer["prior"].size()))
            if len(set(prior_sizes)) != 1:
                raise RuntimeError(
                    "Canonical-plane replay sizes differ across ranks; refusing to enter mismatched DDP update schedules: "
                    f"{prior_sizes}"
                )
        if self.training_with_expert_data:
            replay_buffer["expert_slicer"] = expert_buffer

        print("Starting training")
        replay_updates_start_local_time = self._checkpoint_local_time
        if self.cfg.resume_replay_warmup_steps > 0:
            if self._checkpoint_local_time <= 0:
                raise RuntimeError("resume_replay_warmup_steps requires a nonzero checkpoint step")
            if loaded_checkpoint_buffer:
                raise RuntimeError(
                    "resume_replay_warmup_steps was requested but a checkpoint replay buffer exists; "
                    "remove or quarantine the buffer explicitly before recovery"
                )
            replay_updates_start_local_time += int(self.cfg.resume_replay_warmup_steps)
            print(
                "[RECOVERY] Rebuilding replay with the loaded policy and no optimizer updates: "
                f"start_local_time={self._checkpoint_local_time}, "
                f"warmup_local_steps={self.cfg.resume_replay_warmup_steps}, "
                f"updates_start_local_time={replay_updates_start_local_time}, "
                f"updates_start_global_time={replay_updates_start_local_time * _global_step_scale(self.cfg)}",
                flush=True,
            )
        replay_warmup_complete_logged = self.cfg.resume_replay_warmup_steps <= 0
        global_step_scale = _global_step_scale(self.cfg)
        local_step_increment = self.cfg.online_parallel_envs
        global_step_increment = local_step_increment * global_step_scale
        max_local_time = (self.cfg.num_env_steps + global_step_scale - 1) // global_step_scale
        if self._write_shared_artifacts:
            print(
                "[INFO] Step accounting: "
                f"num_envs_per_rank={_num_envs_per_rank(self.cfg)}, global_parallel_envs={_global_parallel_envs(self.cfg)}, "
                f"local_step_increment={local_step_increment}, global_step_increment={global_step_increment}, "
                f"num_env_steps_global={self.cfg.num_env_steps}, num_seed_steps_local={self.cfg.num_seed_steps}, "
                f"update_agent_every_local={self.cfg.update_agent_every}, world_size={self.distributed_world_size}, "
                f"loss_mode={_distributed_loss_mode(self.cfg)}, effective_batch_size={_effective_batch_size(self.cfg)}, "
                f"replay_capacity_per_rank={_replay_capacity_per_rank(self.cfg)}, "
                f"effective_replay_capacity={_effective_replay_capacity(self.cfg)}, "
                f"trajectory_steps_per_rank={_trajectory_steps_per_rank(self.cfg)}, "
                f"compile={self.cfg.agent.compile}"
            )
        progb = tqdm(
            total=self.cfg.num_env_steps,
            initial=min(self._checkpoint_global_time, self.cfg.num_env_steps),
            disable=self.cfg.disable_tqdm,
        )
        td, info = train_env.reset()
        if self.cfg.fail_on_nonfinite:
            _assert_finite(
                td,
                label="env.reset.obs",
                rank=self.distributed_rank,
                local_time=self._checkpoint_local_time,
                global_time=self._checkpoint_global_time,
                optimizer_steps=self._optimizer_steps,
            )
        # see https://farama.org/Vector-Autoreset-Mode
        terminated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
        truncated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
        done = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
        main_rollout_state = RolloutContextState()
        prior_collector_state = self._reset_prior_collector() if self.prior_env is not None else None
        completed_main_iterations = (
            self._checkpoint_local_time // self.cfg.online_parallel_envs
            if loaded_prior_checkpoint_buffer
            else 0
        )
        prior_env_steps = completed_main_iterations * self.cfg.prior_plane_envs
        prior_collection_seconds = 0.0
        prior_log_start_steps = prior_env_steps
        prior_log_start_seconds = 0.0
        total_metrics = None
        metric_update_counts: dict[str, int] = {}
        start_time = time.time()
        fps_start_time = time.time()
        checkpoint_time_checker = EveryNStepsChecker(self._checkpoint_global_time, self.cfg.checkpoint_every_steps)
        eval_time_checker = EveryNStepsChecker(self._checkpoint_global_time, self.cfg.eval_every_steps)
        update_agent_time_checker = EveryNStepsChecker(self._checkpoint_local_time, self.cfg.update_agent_every)
        log_time_checker = EveryNStepsChecker(self._checkpoint_global_time, self.cfg.log_every_updates)

        eval_instances = [
            isinstance(evaluation, HumanoidVerseMjlabTrackingEvaluation)
            for evaluation in self.evaluations.values()
        ]
        uses_humanoidverse_eval = any(
            isinstance(evaluation, (HumanoidVerseMjlabTrackingEvaluation, SameZTerrainEvaluation))
            for evaluation in self.evaluations.values()
        )
        distributed_tracking_eval = (
            self.cfg.distributed_sync
            and self.distributed_world_size > 1
            and any(eval_instances)
        )

        for local_time in range(self._checkpoint_local_time, max_local_time + local_step_increment, local_step_increment):
            global_time = local_time * global_step_scale
            if global_time > self.cfg.num_env_steps:
                break
            if (local_time != self._checkpoint_local_time) and checkpoint_time_checker.check(global_time):
                checkpoint_time_checker.update_last_step(global_time)
                self.save(local_time=local_time, global_time=global_time, optimizer_steps=self._optimizer_steps, replay_buffer=replay_buffer)

            if global_time >= self.cfg.num_env_steps:
                break

            if (self.evaluate and eval_time_checker.check(global_time)) or (
                self.evaluate and global_time == self._checkpoint_global_time
            ):
                eval_metrics = {}
                run_eval_on_this_rank = (
                    distributed_tracking_eval
                    or (not self.cfg.distributed_sync)
                    or self.distributed_rank == 0
                )
                if run_eval_on_this_rank:
                    eval_metrics = self.eval(
                        global_time,
                        replay_buffer=replay_buffer,
                        distributed_shard=distributed_tracking_eval,
                        write_outputs=not distributed_tracking_eval,
                    )
                if distributed_tracking_eval:
                    if self.distributed_rank == 0:
                        self._record_evaluation_results(global_time, eval_metrics)
                    else:
                        eval_metrics = {}
                if self.cfg.distributed_sync:
                    barrier()
                eval_time_checker.update_last_step(global_time)
                if uses_humanoidverse_eval:
                    # reset if there is a humanoidverse evaluation
                    td, info = train_env.reset()
                    if self.cfg.fail_on_nonfinite:
                        _assert_finite(
                            td,
                            label="env.post_eval_reset.obs",
                            rank=self.distributed_rank,
                            local_time=local_time,
                            global_time=global_time,
                            optimizer_steps=self._optimizer_steps,
                        )
                    terminated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
                    truncated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
                    done = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
                    main_rollout_state = RolloutContextState()
                    # The priority evaluator temporarily switches the shared
                    # motion library into evaluation mode.  Main and plane
                    # collectors must both restart after it is restored.
                    if self.prior_env is not None:
                        prior_collector_state = self._reset_prior_collector()

                if self.cfg.prioritization:
                    # priorities
                    priority_payload = None
                    # Distributed tracking evaluation runs on every rank, but the
                    # merged metrics only exist on rank 0. Compute priorities once
                    # there, then broadcast the unchanged payload to all ranks.
                    compute_priority_on_this_rank = (
                        not self.cfg.distributed_sync or self.distributed_rank == 0
                    )
                    if compute_priority_on_this_rank:
                        assert len(eval_metrics[self.priorization_eval_name]) == len(replay_buffer["expert_slicer"].motion_ids), (
                            "Mismatch in number of motions returned by the eval"
                        )
                        index_in_buffer, name_in_buffer = {}, {}
                        for i, motion_id in enumerate(replay_buffer["expert_slicer"].motion_ids):
                            index_in_buffer[motion_id] = i
                            if hasattr(replay_buffer["expert_slicer"], "file_names"):
                                name_in_buffer[motion_id] = replay_buffer["expert_slicer"].file_names[i]
                        motions_id, priorities, idxs = [], [], []
                        for _, metr in eval_metrics[self.priorization_eval_name].items():
                            motions_id.append(metr["motion_id"])
                            priorities.append(metr["emd"])
                            idxs.append(index_in_buffer[metr["motion_id"]])
                        non_finite_priorities = [
                            (motion_id, priority)
                            for motion_id, priority in zip(motions_id, priorities)
                            if not np.isfinite(priority)
                        ]
                        if non_finite_priorities:
                            raise RuntimeError(
                                "Priority evaluation produced non-finite EMD values; refusing to update "
                                f"motion sampling weights: {non_finite_priorities[:8]}"
                            )
                        priorities = (
                            torch.clamp(
                                torch.tensor(priorities, dtype=torch.float32, device=self.agent.device),
                                min=self.cfg.prioritization_min_val,
                                max=self.cfg.prioritization_max_val,
                            )
                            * self.cfg.prioritization_scale
                        )

                        if self.cfg.prioritization_mode == "lin":
                            pass
                        elif self.cfg.prioritization_mode == "exp":
                            priorities = 2**priorities
                        elif self.cfg.prioritization_mode == "bin":
                            bins = torch.floor(priorities)
                            for i in range(int(bins.min().item()), int(bins.max().item()) + 1):
                                mask = bins == i
                                n = mask.sum().item()
                                if n > 0:
                                    priorities[mask] = 1 / n
                        else:
                            raise ValueError(f"Unsupported prioritization mode {self.cfg.prioritization_mode}")
                        priority_payload = {
                            "priorities": priorities.detach().cpu(),
                            "motion_ids": motions_id,
                            "idxs": idxs,
                            "file_name": name_in_buffer,
                        }
                    if self.cfg.distributed_sync:
                        priority_payload = broadcast_object(priority_payload, src=0)
                    if priority_payload is None:
                        raise RuntimeError("Prioritization requires evaluation metrics, but no priority payload was produced.")
                    priorities = priority_payload["priorities"].to(self.agent.device)
                    motions_id = priority_payload["motion_ids"]
                    idxs = priority_payload["idxs"]
                    name_in_buffer = priority_payload["file_name"]

                    train_env._env._motion_lib.update_sampling_weight_by_id(
                        priorities=list(priorities), motions_id=motions_id, file_name=name_in_buffer
                    )
                    if self.prior_env is not None:
                        self.prior_env._env._motion_lib.update_sampling_weight_by_id(
                            priorities=list(priorities),
                            motions_id=motions_id,
                            file_name=name_in_buffer,
                        )

                    replay_buffer["expert_slicer"].update_priorities(
                        priorities=priorities.to(self.cfg.buffer_device), idxs=torch.tensor(np.array(idxs), device=self.cfg.buffer_device)
                    )

            if global_time + global_step_increment > self.cfg.num_env_steps:
                if self._write_shared_artifacts:
                    print(
                        "[INFO] Stopping before next rollout to avoid exceeding global sample budget: "
                        f"current_global_time={global_time}, next_global_time={global_time + global_step_increment}, "
                        f"num_env_steps_global={self.cfg.num_env_steps}"
                    )
                break

            with torch.no_grad():
                obs = tree_map(lambda x: torch.tensor(x, dtype=dtype_numpytotorch_lower_precision(x.dtype), device=self.agent.device), td)
                # TODO consistency with obs_space: remove time assigned by TimeAwareObservationWrapper
                step_count = obs.pop("time")

                history_context = None
                if "history" in obs:
                    # this works in inference mode
                    if len(obs["history"]["action"]) == 0:
                        history_context = self.agent._model._context_encoder.get_initial_context(self.cfg.online_parallel_envs)
                    else:
                        history_context = self.agent.history_inference(obs=obs["history"]["observation"], action=obs["history"]["action"])[
                            :, -1
                        ].clone()

                main_rollout_state = self._advance_rollout_state(
                    main_rollout_state,
                    step_count,
                    replay_buffer,
                )
                context = main_rollout_state.z
                if context is None:
                    raise RuntimeError("Main rollout did not produce z")
                if local_time < self.cfg.num_seed_steps:
                    action = train_env.action_space.sample().astype(np.float32)
                else:
                    # this works in inference mode
                    if history_context is not None:
                        action = self.agent.act(obs=obs, z=context, context=history_context, mean=False)
                    else:
                        action = self.agent.act(obs=obs, z=context, mean=False)
                    # TODO a bit hard-coded -- just to avoid moving stuff from cpu to cuda
                    if isinstance(self.cfg.env, HumanoidVerseMjlabConfig):
                        action = action.cpu().detach().numpy()
                check_rollout_nonfinite = (
                    self.cfg.fail_on_nonfinite
                    and self.cfg.nonfinite_check_rollout_every_local_steps > 0
                    and local_time % self.cfg.nonfinite_check_rollout_every_local_steps == 0
                )
                if check_rollout_nonfinite:
                    _assert_finite(
                        obs,
                        label="rollout.obs",
                        rank=self.distributed_rank,
                        local_time=local_time,
                        global_time=global_time,
                        optimizer_steps=self._optimizer_steps,
                    )
                    _assert_finite(
                        context,
                        label="rollout.context",
                        rank=self.distributed_rank,
                        local_time=local_time,
                        global_time=global_time,
                        optimizer_steps=self._optimizer_steps,
                    )
                    _assert_finite(
                        history_context,
                        label="rollout.history_context",
                        rank=self.distributed_rank,
                        local_time=local_time,
                        global_time=global_time,
                        optimizer_steps=self._optimizer_steps,
                    )
                    _assert_finite(
                        action,
                        label="rollout.action",
                        rank=self.distributed_rank,
                        local_time=local_time,
                        global_time=global_time,
                        optimizer_steps=self._optimizer_steps,
                    )
            new_td, new_reward, new_terminated, new_truncated, new_info = train_env.step(action)
            if check_rollout_nonfinite:
                _assert_finite(
                    new_td,
                    label="env.step.obs",
                    rank=self.distributed_rank,
                    local_time=local_time,
                    global_time=global_time,
                    optimizer_steps=self._optimizer_steps,
                )
                _assert_finite(
                    new_reward,
                    label="env.step.reward",
                    rank=self.distributed_rank,
                    local_time=local_time,
                    global_time=global_time,
                    optimizer_steps=self._optimizer_steps,
                )

            # we check if at the next iteration we will evaluate
            next_local_time = local_time + local_step_increment
            next_global_time = next_local_time * global_step_scale
            if (self.evaluate and eval_time_checker.check(next_global_time)) or (
                self.evaluate and next_global_time == self._checkpoint_global_time
            ):
                if isinstance(self.cfg.env, HumanoidVerseMjlabConfig) and uses_humanoidverse_eval:
                    # make sure we set truncated since at the next iteration we are forced to reset the environment
                    # after the evaluation. This is because we share the environment with the evaluation
                    new_truncated = np.ones_like(new_truncated, dtype=bool)
                    truncated = np.ones_like(new_truncated, dtype=bool)

            if Version(gymnasium.__version__) >= Version("1.0"):
                if self.cfg.use_trajectory_buffer:
                    data = {
                        "observation": tree_map(lambda x: x[None, ...], obs),
                        "action": action[None, ...],
                        "terminated": terminated[None, ..., None],
                        "truncated": truncated[None, ..., None],
                        "step_count": step_count[None, ..., None],
                        "reward": new_reward[None, ..., None],
                    }
                    data["observation"].pop("history", None)
                    if context is not None:
                        data["z"] = context[None, ...]
                    if history_context is not None:
                        data["history_context"] = history_context[None, ...]
                    if "qpos" in info:
                        data["qpos"] = info["qpos"][None, ...]
                    if "qvel" in info:
                        data["qvel"] = info["qvel"][None, ...]
                    if "aux_rewards" in new_info:
                        data["aux_rewards"] = {k: v[None, ..., None] for k, v in new_info["aux_rewards"].items() if not k.startswith("_")}
                else:
                    # We add only transitions corresponding to environments that have not reset in the previous step.
                    # For environments that have reset in the previous step, the new observation corresponds to the state after reset.
                    indexes = ~done

                    real_next_obs = tree_map(lambda x: x.astype(np.float32 if x.dtype == np.float64 else x.dtype)[indexes], new_td)
                    # TODO again, we need to remove "time" from the observation (to stay consistent with obs_space)
                    _ = real_next_obs.pop("time")
                    _ = real_next_obs.pop("history", None)

                    data = {
                        "observation": tree_map(lambda x: x[indexes], obs),
                        "action": action[indexes],
                        "step_count": step_count[indexes],
                        "reward": new_reward[indexes].reshape(-1, 1),
                        "next": {
                            "observation": real_next_obs,
                            "terminated": new_terminated[indexes].reshape(-1, 1),
                            "truncated": new_truncated[indexes].reshape(-1, 1),
                        },
                    }
                    data["observation"].pop("history", None)
                    if context is not None:
                        data["z"] = context[indexes]
                    if history_context is not None:
                        data["history_context"] = history_context[indexes]
                    if "qpos" in info:
                        data["qpos"] = info["qpos"][indexes]
                        data["next"]["qpos"] = new_info["qpos"][indexes]
                    if "qvel" in info:
                        data["qvel"] = info["qvel"][indexes]
                        data["next"]["qvel"] = new_info["qvel"][indexes]
                    if "aux_rewards" in new_info:
                        data["aux_rewards"] = {
                            k: v[indexes].reshape(-1, 1) for k, v in new_info["aux_rewards"].items() if not k.startswith("_")
                        }
            else:
                raise NotImplementedError("still some work to do for gymnasium < 1.0")
            if check_rollout_nonfinite:
                _assert_finite(
                    data,
                    label="replay.extend.data",
                    rank=self.distributed_rank,
                    local_time=local_time,
                    global_time=global_time,
                    optimizer_steps=self._optimizer_steps,
                )
            replay_buffer["train"].extend(data)

            if prior_collector_state is not None:
                prior_collector_state, prior_elapsed = self._step_prior_collector(
                    prior_collector_state,
                    replay_buffer=replay_buffer,
                    local_time=local_time,
                    global_time=global_time,
                    optimizer_steps=self._optimizer_steps,
                )
                prior_env_steps += self.cfg.prior_plane_envs
                prior_collection_seconds += prior_elapsed

            replay_warmup_complete = local_time >= replay_updates_start_local_time
            if replay_warmup_complete and not replay_warmup_complete_logged:
                print(
                    "[RECOVERY] Replay warmup complete; optimizer updates are enabled at "
                    f"local_time={local_time} global_time={global_time} "
                    f"replay_time_steps={len(replay_buffer['train'])}",
                    flush=True,
                )
                replay_warmup_complete_logged = True
            if (
                replay_warmup_complete
                and len(replay_buffer["train"]) > 0
                # Trajectory sampling draws slices with replacement, so the
                # plane replay does not need batch_size distinct transitions.
                # It only needs a real (s_t, s_{t+1}) segment after the
                # administrative startup boundary. Main num_seed_steps is far
                # longer than this in formal runs; the explicit guard keeps
                # small smoke runs useful as well.
                and ("prior" not in replay_buffer or len(replay_buffer["prior"]) >= 3)
                and local_time > self.cfg.num_seed_steps
                and update_agent_time_checker.check(local_time)
            ):
                update_agent_time_checker.update_last_step(local_time)
                for _ in range(self.cfg.num_agent_updates):
                    metrics = self.agent.update(replay_buffer, local_time)
                    self._optimizer_steps += 1
                    if self.cfg.fail_on_nonfinite:
                        _assert_finite(
                            metrics,
                            label="agent.update.metrics",
                            rank=self.distributed_rank,
                            local_time=local_time,
                            global_time=global_time,
                            optimizer_steps=self._optimizer_steps,
                        )
                        if (
                            self.cfg.nonfinite_check_model_every_updates > 0
                            and self._optimizer_steps % self.cfg.nonfinite_check_model_every_updates == 0
                        ):
                            _assert_model_finite(
                                self.agent._model,
                                rank=self.distributed_rank,
                                local_time=local_time,
                                global_time=global_time,
                                optimizer_steps=self._optimizer_steps,
                            )
                    if self.cfg.distributed_sync:
                        sync_floating_buffers(self.agent._model)
                    total_metrics, metric_update_counts = _accumulate_metrics(
                        total_metrics,
                        metric_update_counts,
                        metrics,
                    )

            if log_time_checker.check(global_time) and total_metrics is not None:
                log_time_checker.update_last_step(global_time)
                m_dict = {}
                reduced_metrics = (
                    reduce_metric_accumulators(total_metrics, metric_update_counts)
                    if self.cfg.distributed_average_metrics
                    else {
                        key: (total_metrics[key] / metric_update_counts[key]).mean()
                        for key in sorted(total_metrics)
                    }
                )
                for key, value in reduced_metrics.items():
                    m_dict[key] = np.round(value.item(), 6)
                m_dict.update(self._get_torso_contact_force_metrics(train_env))
                m_dict.update(self._get_terrain_metrics(train_env))
                m_dict["duration [minutes]"] = (time.time() - start_time) / 60
                m_dict["FPS"] = (1 if global_time == 0 else self.cfg.log_every_updates) / (time.time() - fps_start_time)
                if self.cfg.distributed_sync and self.distributed_world_size > 1:
                    m_dict["distributed/world_size"] = self.distributed_world_size
                    m_dict["distributed/loss_mode"] = _distributed_loss_mode(self.cfg)
                    m_dict["distributed/effective_batch_size"] = _effective_batch_size(self.cfg)
                    m_dict["distributed/num_envs_per_rank"] = _num_envs_per_rank(self.cfg)
                    m_dict["distributed/global_parallel_envs"] = _global_parallel_envs(self.cfg)
                    m_dict["distributed/replay_capacity_per_rank"] = _replay_capacity_per_rank(self.cfg)
                    m_dict["distributed/effective_replay_capacity"] = _effective_replay_capacity(self.cfg)
                    m_dict["distributed/trajectory_steps_per_rank"] = _trajectory_steps_per_rank(self.cfg)
                    m_dict["distributed/compile"] = int(bool(self.cfg.agent.compile))
                    m_dict["distributed/gradient_sync"] = self.cfg.distributed_gradient_sync
                    m_dict["distributed/ddp_bucket_cap_mb"] = float(self.cfg.ddp_bucket_cap_mb)
                m_dict["distributed/local_env_steps"] = int(local_time)
                m_dict["distributed/global_env_steps"] = int(global_time)
                m_dict["distributed/optimizer_steps"] = int(self._optimizer_steps)
                if prior_collector_state is not None:
                    prior_steps_since_log = prior_env_steps - prior_log_start_steps
                    prior_seconds_since_log = prior_collection_seconds - prior_log_start_seconds
                    prior_global_steps = prior_env_steps * global_step_scale
                    m_dict["prior/plane_env_steps"] = int(prior_global_steps)
                    m_dict["prior/total_sim_transitions"] = int(global_time + prior_global_steps)
                    m_dict["prior/collection_fps"] = float(
                        prior_steps_since_log * global_step_scale / max(prior_seconds_since_log, 1.0e-9)
                    )
                    m_dict["prior/replay_size_per_rank"] = int(replay_buffer["prior"].size())
                    prior_log_start_steps = prior_env_steps
                    prior_log_start_seconds = prior_collection_seconds
                if self._write_shared_artifacts and self.cfg.use_wandb:
                    wandb.log(
                        {f"train/{k}": v for k, v in m_dict.items()},
                        step=global_time,
                    )
                if self._write_shared_artifacts:
                    print(m_dict)
                total_metrics = None
                metric_update_counts = {}
                fps_start_time = time.time()
                m_dict["timestep"] = global_time
                m_dict["local_timestep"] = local_time
                if self.train_logger is not None:
                    self.train_logger.log(m_dict)

            progb.update(global_step_increment)
            td = new_td
            terminated = new_terminated
            truncated = new_truncated
            done = np.logical_or(new_terminated.ravel(), new_truncated.ravel())
            info = new_info
        train_env.close()

    def eval(self, t, replay_buffer, *, distributed_shard: bool = False, write_outputs: bool = True):
        print(
            f"Starting evaluation at time {t} on rank {self.distributed_rank}"
            + (" (distributed motion shard)" if distributed_shard else ""),
            flush=True,
        )
        evaluation_results = {}

        # This will contain the results, mapping evaluation.cfg.name --> dict of metrics
        evaluation_results = {}
        for evaluation_name in self.evaluations.keys():
            logger = self.eval_loggers.get(evaluation_name) if write_outputs else None
            evaluation = self.evaluations[evaluation_name]

            if (
                distributed_shard
                and self.distributed_rank != 0
                and not isinstance(evaluation, HumanoidVerseMjlabTrackingEvaluation)
            ):
                continue

            # NOTE we have this inside the loop so that the agent is not moved to cpu if we don't evaluate
            if not isinstance(self.cfg.env, HumanoidVerseMjlabConfig):
                self.agent._model.to("cpu")
            self.agent._model.train(False)

            if isinstance(self.cfg.env, HumanoidVerseMjlabConfig):
                if isinstance(evaluation, SameZTerrainEvaluation):
                    evaluation_metrics, wandb_dict = evaluation.run(
                        timestep=t,
                        agent_or_model=self.agent,
                        replay_buffer=replay_buffer,
                        logger=logger,
                        base_env_config=self.cfg.env,
                        write_outputs=write_outputs,
                        motion_lib=self.train_env._env._motion_lib,
                        expert_buffer=replay_buffer.get("expert_slicer"),
                    )
                    if write_outputs and self._write_shared_artifacts and self.cfg.use_wandb and wandb_dict is not None:
                        wandb.log(
                            {f"eval/{evaluation_name}/{k}": v for k, v in wandb_dict.items()},
                            step=t,
                        )
                    evaluation_results[evaluation_name] = evaluation_metrics
                    continue
                eval_env = (
                    self._get_priority_eval_env()
                    if self._uses_fixed_flat_priority_eval(evaluation_name)
                    else self.train_env
                )
                evaluation_metrics, wandb_dict = evaluation.run(
                    timestep=t,
                    agent_or_model=self.agent,
                    replay_buffer=replay_buffer,
                    logger=logger,
                    env=eval_env,
                    write_outputs=write_outputs,
                    motion_lib=self.train_env._env._motion_lib,
                    expert_buffer=replay_buffer.get("expert_slicer"),
                    distributed=distributed_shard,
                )
            else:
                evaluation_metrics, wandb_dict = evaluation.run(
                    timestep=t,
                    agent_or_model=self.agent,
                    replay_buffer=replay_buffer,
                    logger=logger,
                )
            # For wandb dict, put it on wandb
            if write_outputs and self._write_shared_artifacts and self.cfg.use_wandb and wandb_dict is not None:
                wandb.log(
                    {f"eval/{evaluation_name}/{k}": v for k, v in wandb_dict.items()},
                    step=t,
                )

            evaluation_results[evaluation_name] = evaluation_metrics

        # ---------------------------------------------------------------
        # this is important, move back the agent to cuda and
        # restart the training
        if not isinstance(self.cfg.env, HumanoidVerseMjlabConfig):
            self.agent._model.to(self.cfg.agent.model.device)
        self.agent._model.train()

        return evaluation_results

    def _record_evaluation_results(self, timestep: int, evaluation_results: dict[str, dict[str, dict]]) -> None:
        if not self._write_shared_artifacts:
            return
        for evaluation_name, metrics in evaluation_results.items():
            evaluation = self.evaluations[evaluation_name]
            logger = self.eval_loggers.get(evaluation_name)
            evaluation.record_results(metrics, timestep=timestep, logger=logger)
            if self.cfg.use_wandb:
                wandb_dict = evaluation.summarize(metrics)
                wandb.log(
                    {f"eval/{evaluation_name}/{key}": value for key, value in wandb_dict.items()},
                    step=timestep,
                )

    def save(self, *, local_time: int, global_time: int, optimizer_steps: int, replay_buffer: Dict[str, tp.Any]) -> None:
        checkpoint_dir = self.work_dir / CHECKPOINT_DIR_NAME
        sync_report = None
        if self.cfg.distributed_sync:
            barrier()
            sync_report = module_sync_report(self.agent._model, src=0)
            if sync_report["max_abs_diff_from_rank0"] > 1.0e-5:
                raise RuntimeError(f"Distributed model state diverged before checkpoint: {sync_report}")
        if self._write_shared_artifacts:
            print(f"Checkpointing at local_time={local_time} global_time={global_time} optimizer_steps={optimizer_steps}")
            self.agent.save(str(checkpoint_dir))
        if self.cfg.checkpoint_buffer:
            replay_buffer["train"].save(self._checkpoint_buffer_path(checkpoint_dir, "train"))
            if "prior" in replay_buffer:
                replay_buffer["prior"].save(self._checkpoint_buffer_path(checkpoint_dir, "prior"))
        if self.cfg.distributed_sync:
            barrier()
        if self._write_shared_artifacts:
            if sync_report is not None:
                with (checkpoint_dir / "distributed_sync.json").open("w+") as f:
                    json.dump(sync_report, f, indent=4)
            with (checkpoint_dir / "train_status.json").open("w+") as f:
                json.dump(
                    _make_train_status(
                        self.cfg,
                        local_time=local_time,
                        global_time=global_time,
                        optimizer_steps=optimizer_steps,
                    ),
                    f,
                    indent=4,
                )
        if self.cfg.distributed_sync:
            barrier()


def train_bfm_zero():
    raise RuntimeError(
        "Legacy train_bfm_zero entrypoint is disabled in this MJLab build. "
        "Use humanoidverse.train or ./run_train.sh."
    )

if __name__ == "__main__":
    # This is the bare minimum CLI interface to launch experiments, but ideally you should
    # launch your experiments from Python code (e.g., see under "scripts")
    train_bfm_zero()

# uv run --no-cache -m humanoidverse.meta_online_entry_point
