
import collections
import copy
import dataclasses
import gc
import numbers
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Mapping

import numpy as np
import ot
import torch
import torch.distributed as dist
from scipy.spatial.distance import cdist
from torch.utils._pytree import tree_map
from tqdm import tqdm

from humanoidverse.utils.reference_observations import reference_base_ang_vel

from ..envs.humanoidverse_mjlab import HumanoidVerseMjlabConfig
from .base import BaseEvalConfig, extract_model


def get_next(field: str, data: Any):
    if "next" in data and field in data["next"]:
        return data["next"][field]
    elif f"next_{field}" in data:
        return data[f"next_{field}"]
    else:
        raise ValueError(f"No next of {field} found in data.")


@dataclasses.dataclass
class Episode:
    storage: Dict | None = None

    def initialise(self, observation: np.ndarray | Dict[str, np.ndarray], info: Dict) -> None:
        if self.storage is None:
            self.storage = defaultdict(list)
        if isinstance(observation, Mapping):
            self.storage["observation"] = {k: [copy.deepcopy(v)] for k, v in observation.items()}
        else:
            self.storage["observation"].append(copy.deepcopy(observation))
        self.storage["info"] = {
            k: [copy.deepcopy(v)] for k, v in info.items() if not k.startswith("_") and k not in ["final_observation", "final_info"]
        }

    def add(
        self,
        observation: np.ndarray | Dict[str, np.ndarray],
        reward: np.ndarray,
        action: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        info: Dict,
    ) -> None:
        if isinstance(observation, Mapping):
            for k, v in observation.items():
                self.storage["observation"][k].append(copy.deepcopy(v))
        else:
            self.storage["observation"].append(copy.deepcopy(observation))
        for k, v in info.items():
            if not k.startswith("_") and k not in ["final_observation", "final_info"]:
                if k in self.storage["info"]:
                    # this is a quick fix but not robust fix to the problem that certain environments may return keys that are different from the one in the first step
                    # We may want to find a different solution to deal with this problem
                    self.storage["info"][k].append(copy.deepcopy(v))
                else:
                    self.storage["info"][k] = [copy.deepcopy(v)]
        self.storage["reward"].append(copy.deepcopy(reward))
        self.storage["action"].append(copy.deepcopy(action))
        self.storage["terminated"].append(copy.deepcopy(terminated))
        self.storage["truncated"].append(copy.deepcopy(truncated))

    def get(self) -> Dict[str, np.ndarray]:
        output = {}
        for k, v in self.storage.items():
            if k in ["observation", "info"]:
                if isinstance(v, Mapping):
                    output[k] = {}
                    for k2, v2 in v.items():
                        output[k][k2] = np.array(v2)
                else:
                    output[k] = np.array(v)
            else:
                output[k] = np.array(v)
        return output

xpos_bodies = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    # "left_wrist_roll_link",
    # "left_wrist_pitch_link",
    # "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    # "right_wrist_roll_link",
    # "right_wrist_pitch_link",
    # "right_wrist_yaw_link",
]

def get_backward_observation(env, motion_id, include_last_action, velocity_multiplier: float = 1.0) -> torch.Tensor:
    import numpy as np

    from humanoidverse.envs.motion_observations import (
        compute_humanoid_observations_max,
        compute_humanoid_observations_max_with_contact,
    )
    from humanoidverse.utils.torch_utils import quat_rotate_inverse

    if not env.config.obs.use_obs_filter:
        raise ValueError("MJLab tracking evaluation requires obs.use_obs_filter=True")

    motion_times = torch.arange(int(np.ceil((env._motion_lib._motion_lengths[motion_id] / env.dt).cpu()))).to(env.device) * env.dt
    # motion_times = torch.arange(0, env._motion_lib._motion_num_frames.item() * frame_interval, frame_interval, device=env.device)

    # get blend motion state
    motion_state = env._motion_lib.get_motion_state(motion_id, motion_times)

    ref_body_pos = motion_state["rg_pos_t"]
    ref_body_rots = motion_state["rg_rot_t"]
    ref_body_vels = motion_state["body_vel_t"] * velocity_multiplier
    ref_body_angular_vels = motion_state["body_ang_vel_t"] * velocity_multiplier
    ref_dof_pos = motion_state["dof_pos"] - env.default_dof_pos[0]
    ref_dof_vel = motion_state["dof_vel"] * velocity_multiplier

    # npz_path = "dataprocess/push_door-3steps-no_door_amass-robot_only-processed/motion.npz"
    # data_npz = np.load(npz_path)
    # data_dict = {key: data_npz[key] for key in data_npz.files}

    def process_body_data(data_dict, device="cuda"):
        def extract_and_concat_vec3(arr: torch.Tensor) -> torch.Tensor:
            arr = arr.float()
            # arr: (T, N, 3)
            return torch.cat(
                [
                    arr[:, 0:13],
                    torch.zeros_like(arr[:, 14:16]),  # 两个补 0
                    arr[:, 13:18],
                    arr[:, 21:25],
                    arr[:, 20:21],
                    arr[:, 27:28],
                    arr[:, 12:13],
                ],
                dim=1,
            )

        def extract_and_concat_quat_torch(arr: torch.Tensor) -> torch.Tensor:
            arr = arr.float()
            # arr: (T, N, 4), wxyz
            seg1 = arr[:, 0:13]
            seg2 = torch.tensor([1, 0, 0, 0], dtype=arr.dtype, device=arr.device).view(1, 1, 4).repeat(arr.shape[0], 2, 1)
            seg3 = arr[:, 13:18]
            seg4 = arr[:, 21:25]
            seg7 = arr[:, 12:13]
            seg5 = arr[:, 20:21]
            seg6 = arr[:, 27:28]
            quat_wxyz = torch.cat([seg1, seg2, seg3, seg4, seg5, seg6, seg7], dim=1)  # (T, 24, 4)
            quat_xyzw = torch.cat([quat_wxyz[..., 1:], quat_wxyz[..., :1]], dim=-1)  # (T, 24, 4)
            return quat_xyzw

        # 转为 torch tensor 并送到 GPU
        body_pos = torch.from_numpy(data_dict["body_pos_w"]).to(device)
        body_quat = torch.from_numpy(data_dict["body_quat_w"]).to(device)
        body_lin_vel = torch.from_numpy(data_dict["body_lin_vel_w"]).to(device)
        body_ang_vel = torch.from_numpy(data_dict["body_ang_vel_w"]).to(device)
        joint_pos = torch.from_numpy(data_dict["joint_pos"]).to(device)[:, 1:]
        joint_vel = torch.from_numpy(data_dict["joint_vel"]).to(device)[:, 1:]

        # 处理
        ref_body_pos = extract_and_concat_vec3(body_pos)
        ref_body_vels = extract_and_concat_vec3(body_lin_vel)
        ref_body_angular_vels = extract_and_concat_vec3(body_ang_vel)
        ref_body_rots = extract_and_concat_quat_torch(body_quat)
        ref_dof_pos = joint_pos.float() - env.default_dof_pos[0]
        ref_dof_vel = joint_vel.float()

        return ref_body_pos, ref_body_rots, ref_body_vels, ref_body_angular_vels, ref_dof_pos, ref_dof_vel

    # import ipdb; ipdb.set_trace()
    # ref_body_pos, ref_body_rots, ref_body_vels, ref_body_angular_vels, ref_dof_pos, ref_dof_vel = process_body_data(data_dict)

    # construct observation
    if env.use_contact_in_obs_max:
        contact_binary = env.foot_contact_detect(ref_body_pos, ref_body_vels)
        obs_dict = compute_humanoid_observations_max_with_contact(
            ref_body_pos,
            ref_body_rots,
            ref_body_vels,
            ref_body_angular_vels,
            local_root_obs=True,
            root_height_obs=env.config.obs.root_height_obs,
            contact_binary=contact_binary,
        )
    else:
        obs_dict = compute_humanoid_observations_max(
            ref_body_pos,
            ref_body_rots,
            ref_body_vels,
            ref_body_angular_vels,
            local_root_obs=True,
            root_height_obs=env.config.obs.root_height_obs,
        )
    max_local_self_obs = torch.cat([v for v in obs_dict.values()], dim=-1)

    if env.config.obs.use_obs_filter:
        base_quat = ref_body_rots[:, 0]  # root orientation
        # ref_dof_pos = motion_state["dof_pos"] - env.default_dof_pos[0]
        # ref_dof_vel = motion_state["dof_vel"]
        ref_ang_vel = reference_base_ang_vel(env, base_quat, ref_body_angular_vels[:, 0])
        projected_gravity = quat_rotate_inverse(base_quat, env.gravity_vec[0:1].repeat(max_local_self_obs.shape[0], 1), w_last=True)
        bogus_actions = torch.zeros_like(ref_dof_pos)

        bogus_history_actor = torch.cat([bogus_actions, ref_ang_vel, ref_dof_pos, ref_dof_vel, projected_gravity], dim=-1).repeat(1, 4)
        # obs = torch.cat([bogus_actions, ref_ang_vel, ref_dof_pos, ref_dof_vel, bogus_history_actor, max_local_self_obs, projected_gravity], dim=-1)
        ref_dict = {
            "actions": bogus_actions,
            "ref_ang_vel": ref_ang_vel,
            "ref_dof_pos": ref_dof_pos,
            "dof_pos": motion_state["dof_pos"],
            "ref_dof_vel": ref_dof_vel,
            "dof_vel": motion_state["dof_vel"],
            "fake_history": bogus_history_actor,
            "max_local_self_obs": max_local_self_obs,
            "projected_gravity": projected_gravity,
            "ref_body_pos": ref_body_pos,
            "ref_body_rots": ref_body_rots,
            "ref_body_vels": ref_body_vels,
            "ref_body_angular_vels": ref_body_angular_vels,
            "root_pos": motion_state["root_pos"],
            "root_rot": motion_state["root_rot"],
            "root_vel": motion_state["root_vel"],
            "root_ang_vel": motion_state["root_ang_vel"],
        }
    else:
        # obs = max_local_self_obs
        ref_dict = {
            "max_local_self_obs": max_local_self_obs,
        }

    projected_gravity = quat_rotate_inverse(base_quat, env.gravity_vec[0:1].repeat(max_local_self_obs.shape[0], 1), w_last=True)

    # TODO ensure this is correct
    g1env_state = torch.cat(
        [
            ref_dof_pos,
            ref_dof_vel,
            projected_gravity,
            ref_ang_vel,
        ],
        dim=-1,
    )
    g1env_last_action = torch.zeros_like(ref_dof_pos)

    g1env_privileged_obs = ref_dict["max_local_self_obs"]

    g1env_obs = {
        "state": g1env_state,
        "privileged_state": g1env_privileged_obs,
    }

    if include_last_action:
        g1env_obs["last_action"] = g1env_last_action

    return g1env_obs, ref_dict


class HumanoidVerseMjlabTrackingEvaluationConfig(BaseEvalConfig):
    name_in_logs: str = "humanoidverse_tracking_eval"
    env: HumanoidVerseMjlabConfig | None = None
    # Used if above env is provided
    num_envs: int = 1024

    n_episodes_per_motion: int = 1
    include_results_from_all_envs: bool = False  # If True, include results from all num_envs envs, not just first num_motions

    disable_tqdm: bool = True
    context_batch_size: int = 65536
    emd_workers_per_rank: int = 8

    def build(self):
        return HumanoidVerseMjlabTrackingEvaluation(self)


class HumanoidVerseMjlabTrackingEvaluation:
    def __init__(self, config: HumanoidVerseMjlabTrackingEvaluationConfig):
        self.cfg = config

    def run(
        self,
        *,
        timestep,
        agent_or_model,
        logger,
        env: Any | None = None,
        motion_ids: list[int] | None = None,
        write_outputs: bool = True,
        motion_lib=None,
        expert_buffer=None,
        distributed: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        owns_env = env is None
        if env is None:
            if self.cfg.env is None:
                raise ValueError("Either env or cfg.env must be provided")
            eval_cfg = self.cfg.env.model_copy(update={"evaluation_fast_path": expert_buffer is not None})
            env, _ = eval_cfg.build(num_envs=self.cfg.num_envs, motion_lib=motion_lib)
        else:
            if self.cfg.env is not None:
                raise ValueError("Both env and cfg.env are provided, please provide only one (which one you want to evaluate with?)")
        if expert_buffer is not None:
            if self.cfg.n_episodes_per_motion != 1 or self.cfg.include_results_from_all_envs:
                raise ValueError(
                    "Fast tracking evaluation requires n_episodes_per_motion=1 and "
                    "include_results_from_all_envs=False"
                )
            if motion_ids is not None:
                raise ValueError("Fast distributed tracking evaluation assigns motion IDs internally")
            try:
                return self._run_fast(
                    timestep=timestep,
                    agent_or_model=agent_or_model,
                    logger=logger if write_outputs else None,
                    env=env,
                    expert_buffer=expert_buffer,
                    distributed=distributed,
                )
            finally:
                if owns_env:
                    env.close()
                    del env
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        # NOTE: this is not used as it disables some domain randomization we want to keep for evaluation
        # env._env.set_is_evaluating()

        num_unique_motions = int(env._env._motion_lib._num_unique_motions)
        if motion_ids is None:
            self.motion_ids = list(range(num_unique_motions))
        else:
            self.motion_ids = [int(motion_id) for motion_id in motion_ids]
            if not self.motion_ids:
                raise ValueError("Tracking evaluation motion_ids must not be empty")
            if len(set(self.motion_ids)) != len(self.motion_ids):
                raise ValueError("Tracking evaluation motion_ids must be unique")
            invalid = [motion_id for motion_id in self.motion_ids if not 0 <= motion_id < num_unique_motions]
            if invalid:
                raise ValueError(
                    f"Tracking evaluation motion_ids are outside [0, {num_unique_motions}): {invalid[:8]}"
                )
        n_envs = env._env.num_envs

        # "Parallelization" is handled on the worker level.
        metrics = {}
        for repetition_i in range(self.cfg.n_episodes_per_motion):
            run_metrics = {}
            # If we have less envs than motions, we run iteratively over the motions
            for motion_id_chunk_start in range(0, len(self.motion_ids), n_envs):
                motion_id_chunk = self.motion_ids[motion_id_chunk_start : motion_id_chunk_start + n_envs]
                motion_chunk_results = _async_tracking_worker(
                    (motion_id_chunk, 0, agent_or_model),
                    env=env,
                    disable_tqdm=self.cfg.disable_tqdm,
                    include_results_from_all_envs=self.cfg.include_results_from_all_envs,
                )
                for k, v in motion_chunk_results.items():
                    assert k not in run_metrics, "Tried to override existing metric"
                    run_metrics[k] = v

            if self.cfg.n_episodes_per_motion == 1:
                # Just return metrics, do not change names
                metrics = run_metrics
            else:
                # Append run number to metric names
                for k, v in run_metrics.items():
                    metrics[f"{k}_repetition#{repetition_i}"] = v

        wandb_dict = self.summarize(metrics)
        if write_outputs:
            self.record_results(metrics, timestep=timestep, logger=logger)

        # Resume back to original state of the motion lib if we were using shared env
        if self.cfg.env is None:
            env._env._motion_lib.load_motions_for_training()
        env._env.set_is_training()

        if owns_env:
            env.close()
        return metrics, wandb_dict

    def _run_fast(self, *, timestep, agent_or_model, logger, env: Any, expert_buffer, distributed: bool):
        core = env._env
        use_distributed = distributed and dist.is_initialized() and dist.get_world_size() > 1
        world_size = dist.get_world_size() if use_distributed else 1
        rank = dist.get_rank() if use_distributed else 0
        motion_lengths = {
            int(motion_id): int(length)
            for motion_id, length in zip(
                expert_buffer.motion_ids,
                expert_buffer.lengths.detach().cpu().tolist(),
            )
        }
        motion_ids = sorted(motion_lengths)
        rank_chunks = balanced_motion_chunks(
            motion_ids,
            motion_lengths,
            chunk_size=core.num_envs,
            world_size=world_size,
        )
        local_chunks = rank_chunks[rank]

        pending = {}
        local_metrics_by_id: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(
            max_workers=self.cfg.emd_workers_per_rank,
            thread_name_prefix="joint-emd",
        ) as executor:
            for motion_id_chunk in local_chunks:
                records = _fast_tracking_chunk(
                    motion_id_chunk,
                    env=env,
                    agent=agent_or_model,
                    expert_buffer=expert_buffer,
                    context_batch_size=self.cfg.context_batch_size,
                    disable_tqdm=self.cfg.disable_tqdm,
                )
                for record in records:
                    future = executor.submit(
                        _joint_tracking_metrics,
                        record["joint_pos"],
                        record["target_joint_pos"],
                        record["invalid"],
                    )
                    pending[future] = (record["motion_file"], record["motion_id"])
                while len(pending) > core.num_envs * 2:
                    completed = [future for future in pending if future.done()]
                    if not completed:
                        completed = [next(iter(pending))]
                        completed[0].result()
                    for future in completed:
                        motion_file, motion_id = pending.pop(future)
                        local_metrics_by_id[motion_id] = {
                            **future.result(),
                            "motion_id": motion_id,
                            "motion_file": motion_file,
                        }
            for future, (motion_file, motion_id) in list(pending.items()):
                local_metrics_by_id[motion_id] = {
                    **future.result(),
                    "motion_id": motion_id,
                    "motion_file": motion_file,
                }

        if use_distributed:
            gathered = [None] * world_size if rank == 0 else None
            dist.gather_object(local_metrics_by_id, gathered, dst=0)
        else:
            gathered = [local_metrics_by_id]

        metrics: dict[str, dict[str, Any]] = {}
        if rank == 0:
            metrics_by_id: dict[int, dict[str, Any]] = {}
            for shard in gathered:
                if shard is None:
                    continue
                duplicate_ids = set(metrics_by_id).intersection(shard)
                if duplicate_ids:
                    raise RuntimeError(
                        f"Distributed evaluation produced duplicate motion IDs: {sorted(duplicate_ids)[:8]}"
                    )
                metrics_by_id.update(shard)
            if set(metrics_by_id) != set(motion_ids):
                missing = sorted(set(motion_ids).difference(metrics_by_id))
                extra = sorted(set(metrics_by_id).difference(motion_ids))
                raise RuntimeError(
                    "Distributed evaluation did not cover every motion exactly once: "
                    f"expected={len(motion_ids)} actual={len(metrics_by_id)} missing={missing[:8]} extra={extra[:8]}"
                )
            for motion_id in motion_ids:
                metric = metrics_by_id[motion_id]
                key = str(metric["motion_file"])
                if key in metrics:
                    key = f"{key}#motion_id={motion_id}"
                metrics[key] = metric

        aggregate = self.summarize(metrics)
        valid = sum(
            1
            for metric in metrics.values()
            if isinstance(metric.get("emd"), numbers.Number) and np.isfinite(metric["emd"])
        )
        aggregate["valid_motion_count"] = valid
        aggregate["invalid_motion_count"] = len(metrics) - valid
        if rank == 0 and logger is not None:
            self.record_results(metrics, timestep=timestep, logger=logger)
        return (metrics, aggregate) if rank == 0 else ({}, None)

    @staticmethod
    def summarize(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        aggregate = collections.defaultdict(list)
        wandb_dict = {}
        for metr in metrics.values():
            for key, value in metr.items():
                if isinstance(value, numbers.Number) and np.isfinite(value):
                    aggregate[key].append(value)
        for key, values in aggregate.items():
            wandb_dict[key] = np.mean(values)
            wandb_dict[f"{key}#std"] = np.std(values)
        return wandb_dict

    @staticmethod
    def record_results(metrics: Mapping[str, Mapping[str, Any]], *, timestep: int, logger) -> None:
        if logger is None:
            return
        for motion_name, metric in metrics.items():
            row = dict(metric)
            row["motion_name"] = motion_name
            row["timestep"] = timestep
            logger.log(row)

    def close(self) -> None:
        mp_manager = getattr(self, "mp_manager", None)
        if mp_manager is not None:
            mp_manager.shutdown()


def balanced_motion_chunks(
    motion_ids: list[int],
    lengths: Mapping[int, int],
    *,
    chunk_size: int,
    world_size: int,
) -> list[list[list[int]]]:
    """Pack full simulator batches, then balance estimated frame cost across ranks."""
    if chunk_size <= 0 or world_size <= 0:
        raise ValueError("chunk_size and world_size must be positive")
    if not motion_ids:
        return [[] for _ in range(world_size)]
    if len(set(motion_ids)) != len(motion_ids):
        raise ValueError("motion_ids must be unique")
    num_chunks = (len(motion_ids) + chunk_size - 1) // chunk_size
    capacities = [chunk_size] * num_chunks
    capacities[-1] = len(motion_ids) - chunk_size * (num_chunks - 1)
    chunks: list[list[int]] = [[] for _ in capacities]
    chunk_costs = [0] * num_chunks
    for motion_id in sorted(motion_ids, key=lambda value: (-int(lengths[value]), value)):
        chunk_index = min(
            (index for index in range(num_chunks) if len(chunks[index]) < capacities[index]),
            key=lambda index: (chunk_costs[index], len(chunks[index]), index),
        )
        chunks[chunk_index].append(motion_id)
        chunk_costs[chunk_index] += int(lengths[motion_id])

    rank_chunks: list[list[list[int]]] = [[] for _ in range(world_size)]
    rank_costs = [0] * world_size
    for chunk_index in sorted(range(num_chunks), key=lambda index: (-chunk_costs[index], index)):
        rank_index = min(
            range(world_size),
            key=lambda index: (rank_costs[index], len(rank_chunks[index]), index),
        )
        rank_chunks[rank_index].append(chunks[chunk_index])
        rank_costs[rank_index] += chunk_costs[chunk_index]
    return rank_chunks


def _expert_motion_slice(expert_buffer, motion_id: int) -> dict[str, torch.Tensor]:
    mapping = getattr(expert_buffer, "_evaluation_motion_index", None)
    if mapping is None:
        mapping = {int(value): index for index, value in enumerate(expert_buffer.motion_ids)}
        expert_buffer._evaluation_motion_index = mapping
    trajectory_index = mapping[motion_id]
    start = int(expert_buffer.start_idx[trajectory_index, 0].item())
    length = int(expert_buffer.lengths[trajectory_index].item())
    indices = torch.arange(start, start + length, device=expert_buffer.start_idx.device) % expert_buffer.capacity
    return tree_map(lambda value: value[indices], expert_buffer.storage["observation"])


@torch.no_grad()
def _encode_motion_contexts(model, observations, lengths, *, device: str, batch_size: int):
    joined = {
        key: torch.cat([observation[key][1:] for observation in observations], dim=0)
        for key in observations[0]
    }
    encoded_parts = []
    total = next(iter(joined.values())).shape[0]
    for start in range(0, total, batch_size):
        batch = tree_map(lambda value: value[start : start + batch_size].to(device), joined)
        encoded_parts.append(model.backward_map(batch))
    encoded = model.project_z(torch.cat(encoded_parts, dim=0))
    return list(torch.split(encoded, [length - 1 for length in lengths]))


@torch.no_grad()
def _fast_tracking_chunk(
    motion_ids: list[int],
    *,
    env,
    agent,
    expert_buffer,
    context_batch_size: int,
    disable_tqdm: bool,
) -> list[dict[str, Any]]:
    core = env._env
    model = extract_model(agent)
    observations = [_expert_motion_slice(expert_buffer, motion_id) for motion_id in motion_ids]
    lengths = [int(observation["state"].shape[0]) for observation in observations]
    if any(length < 3 for length in lengths):
        raise ValueError("Tracking evaluation requires every motion to contain at least three frames")
    contexts = _encode_motion_contexts(
        model,
        observations,
        lengths,
        device=core.device,
        batch_size=context_batch_size,
    )
    default_dof_pos = core.default_dof_pos[0]
    targets = [
        (observation["state"][:, : core.num_dof].float().to(core.device) + default_dof_pos).detach().cpu()
        for observation in observations
    ]

    assigned_ids = motion_ids + [motion_ids[-1]] * (core.num_envs - len(motion_ids))
    assigned = torch.tensor(assigned_ids, dtype=torch.long, device=core.device)
    initial = core._motion_lib.get_motion_state(
        assigned,
        torch.zeros(core.num_envs, dtype=torch.float32, device=core.device),
        offset=core.env_origins,
    )
    target_states = {
        "dof_states": torch.stack((initial["dof_pos"], initial["dof_vel"]), dim=-1),
        "root_states": torch.cat(
            (initial["root_pos"], initial["root_rot"], initial["root_vel"], initial["root_ang_vel"]),
            dim=-1,
        ),
    }
    observation, _ = env.reset(target_states=target_states, to_numpy=False)

    all_context = torch.cat(contexts, dim=0)
    context_lengths = torch.tensor([context.shape[0] for context in contexts], device=core.device)
    offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=core.device),
            context_lengths.cumsum(0)[:-1],
        )
    )
    assigned_local = torch.arange(core.num_envs, device=core.device).clamp_max(len(motion_ids) - 1)
    joint_positions = [core.dof_pos.clone()]
    invalid = torch.zeros(core.num_envs, dtype=torch.bool, device=core.device)
    for step in tqdm(
        range(int(context_lengths.max().item())),
        desc="Tracking Evaluation",
        disable=disable_tqdm,
    ):
        context_indices = offsets[assigned_local] + torch.remainder(step, context_lengths[assigned_local])
        action = agent.act(observation, all_context[context_indices], mean=True)
        observation, _, terminated, truncated, _ = env.step(action, to_numpy=False)
        active = step < context_lengths[assigned_local]
        step_invalid = (
            terminated.reshape(-1).bool()
            | truncated.reshape(-1).bool()
            | ~torch.isfinite(core.dof_pos).all(dim=-1)
        )
        invalid |= active & step_invalid
        joint_positions.append(core.dof_pos.clone())

    joint_positions_cpu = torch.stack(joint_positions).cpu()
    invalid_cpu = invalid.cpu()
    records = []
    for env_index, (motion_id, target, length) in enumerate(zip(motion_ids, targets, lengths)):
        records.append(
            {
                "motion_id": motion_id,
                "motion_file": str(core._motion_lib._motion_data_keys[motion_id]),
                "joint_pos": joint_positions_cpu[:length, env_index].numpy(),
                "target_joint_pos": target.numpy(),
                "invalid": bool(invalid_cpu[env_index]),
            }
        )
    return records


def _proximity(distance: np.ndarray, *, bound: float = 2.0, margin: float = 2.0) -> float:
    in_bounds = distance <= bound
    transition = (distance > bound) & (distance <= bound + margin)
    score = in_bounds.astype(np.float64)
    score[transition] = (bound + margin - distance[transition]) / margin
    return float(score.mean())


def _joint_tracking_metrics(
    joint_pos: np.ndarray,
    target_joint_pos: np.ndarray,
    invalid: bool,
) -> dict[str, Any]:
    if invalid or not np.isfinite(joint_pos).all() or not np.isfinite(target_joint_pos).all():
        return {"distance": float("nan"), "emd": float("nan")}
    distance = np.linalg.norm(joint_pos - target_joint_pos, axis=-1)
    cost = cdist(joint_pos, target_joint_pos, metric="euclidean")
    source_weights = np.full(joint_pos.shape[0], 1.0 / joint_pos.shape[0], dtype=np.float64)
    target_weights = np.full(target_joint_pos.shape[0], 1.0 / target_joint_pos.shape[0], dtype=np.float64)
    emd = float(ot.emd2(source_weights, target_weights, cost, numItermax=100000, numThreads=1))
    velocity_distance = np.linalg.norm(np.diff(joint_pos, axis=0) - np.diff(target_joint_pos, axis=0), axis=-1)
    acceleration_distance = np.linalg.norm(
        np.diff(joint_pos, n=2, axis=0) - np.diff(target_joint_pos, n=2, axis=0),
        axis=-1,
    )
    return {
        "obs_state_proximity": _proximity(distance),
        "obs_state_distance": float(distance.mean()),
        "obs_state_emd": emd,
        # Keep the fast path's public metrics identical to the legacy path.
        # Scalar values also keep the one-time cross-rank gather compact.
        "mpjpe_l": float(distance.mean() * 1000.0),
        "vel_dist": float(velocity_distance.mean() * 1000.0),
        "accel_dist": float(acceleration_distance.mean() * 100.0),
        "proximity": _proximity(distance),
        "distance": float(distance.mean()),
        "emd": emd,
    }


def group_assign_motions_to_envs_with_map(motion_ids, num_envs, device=None):
    motion_ids = torch.tensor(motion_ids, device=device)
    num_motions = len(motion_ids)

    env_idxs = list(range(num_envs))
    # Shuffle the environment indices so repeated tracking evaluations do not
    # bind the same motion to the same per-env randomized state every time.
    random.shuffle(env_idxs)
    shuffled_env_idxs = torch.tensor(env_idxs, device=device, dtype=torch.long)

    assigned = motion_ids[shuffled_env_idxs % num_motions]

    motion_to_envs = {}
    for env_id, motion_id in enumerate(assigned.tolist()):
        motion_to_envs.setdefault(motion_id, []).append(env_id)

    return assigned, motion_to_envs


def _async_tracking_worker(
    inputs, env: Any, disable_tqdm: bool = False, include_results_from_all_envs=False
):
    motion_ids, pos, agent = inputs
    model = extract_model(agent)
    # import ipdb; ipdb.set_trace()
    # motion_ids = torch.tensor(motion_ids, device=env.device)
    core_env = env._env

    if not core_env._motion_lib.all_motions_loaded:
        core_env._motion_lib.all_motions_loaded = True
        core_env._motion_lib.load_motions(random_sample=False, num_motions_to_load=core_env._motion_lib._num_unique_motions, start_idx=0)

    metrics = {}
    num_envs = core_env.num_envs

    # Assign motions to envs in grouped fashion
    assigned_motions, motion_to_envs = group_assign_motions_to_envs_with_map(motion_ids, num_envs)

    ctx_dict = {}
    tracking_targets = {}
    # target_xpos_dict = {}
    tracking_joint_pos = {}
    dof_states_list = [None] * num_envs
    root_states_list = [None] * num_envs

    # Precompute context and target states per motion
    for m_id in motion_ids:
        tracking_target, tracking_target_dict = get_backward_observation(env._env, m_id, include_last_action=env.include_last_action)
        # first z should try to reach the next state, ie 1:
        z = model.backward_map(tree_map(lambda x: x[1:], tracking_target)).clone()
        for step in range(z.shape[0]):
            end_idx = min(step + 1, z.shape[0])
            z[step] = z[step:end_idx].mean(dim=0)
        ctx = model.project_z(z)
        ctx_dict[m_id] = ctx
        tracking_targets[m_id] = tree_map(lambda x: x.cpu(), tracking_target)
        tracking_joint_pos[m_id] = tracking_target_dict["dof_pos"].clone()
        # import ipdb; ipdb.set_trace()

        ref_body_rots = tracking_target_dict["ref_body_rots"][0, 0]

        ref_root_init_state = torch.cat(
            [
                tracking_target_dict["ref_body_pos"][0, 0],
                ref_body_rots,
                tracking_target_dict["ref_body_vels"][0, 0],
                tracking_target_dict["ref_body_angular_vels"][0, 0],
            ]
        )
        # import ipdb; ipdb.set_trace()
        for env_id in motion_to_envs[m_id]:
            dof_init_state = torch.zeros_like(core_env.simulator.dof_state.view(num_envs, -1, 2)[0])
            dof_init_state[..., 0] = tracking_target_dict["dof_pos"][0]
            dof_init_state[..., 1] = tracking_target_dict["ref_dof_vel"][0]
            dof_states_list[env_id] = dof_init_state
            root_state = ref_root_init_state.clone()
            if getattr(core_env, "terrain_enabled", False):
                root_state[:3] += core_env.env_origins[env_id]
            root_states_list[env_id] = root_state
            # target_xpos_dict[env_id] = tracking_target_dict["ref_body_pos"][:, : len(xpos_bodies)]

    # this is for environments that are not initialized
    for i in range(num_envs):
        if dof_states_list[i] is None:
            dof_states_list[i] = torch.zeros_like(core_env.simulator.dof_state.view(num_envs, -1, 2)[0])
        if root_states_list[i] is None:
            root_states_list[i] = torch.zeros_like(root_states_list[0])

    target_states = {"dof_states": torch.stack(dof_states_list), "root_states": torch.stack(root_states_list)}

    env_ids = list(range(num_envs))
    env_ids = torch.tensor(env_ids, dtype=torch.long, device=core_env.device)

    with torch.no_grad():
        observation, info = env.reset(target_states=target_states, to_numpy=False)
        assert torch.allclose(env._env.simulator.dof_pos.clone()[env_ids], target_states["dof_states"][..., 0])

        if env.context_length:
            env._update_history(torch.arange(num_envs, device=core_env.device), observation, None)
            observation["history"] = {"observation": [], "action": []}

        # info = {}
        obs_cpu = tree_map(lambda x: x.cpu() if torch.is_tensor(x) else x, observation)
        info_cpu = tree_map(lambda x: x.cpu() if torch.is_tensor(x) else x, info)

        _episode = Episode()
        ooo = {k: v for k, v in obs_cpu.items() if k != "history"}
        _episode.initialise(ooo, info_cpu)

        xpos_log = [core_env.simulator._rigid_body_pos.reshape(num_envs, -1, 3)]
        joint_pos = [core_env.simulator.dof_state[..., 0]]
        joint_vel = [core_env.simulator.dof_state[..., 1]]

        max_ctx_len = max([ctx.shape[0] for ctx in ctx_dict.values()])
        for step in tqdm(range(max_ctx_len), desc="Tracking Evaluation", disable=disable_tqdm):
            ctx_batch = []
            for env_id in range(num_envs):
                m_id = assigned_motions[env_id].item()
                ctx = ctx_dict[m_id]
                ctx_t = ctx[step % ctx.shape[0]]
                ctx_batch.append(ctx_t)
            ctx_batch = torch.stack(ctx_batch)
            action = agent.act(observation, ctx_batch, mean=True)
            observation, reward, terminated, truncated, info = env.step(action, to_numpy=False)
            joint_pos.append(core_env.simulator.dof_state[..., 0])
            joint_vel.append(core_env.simulator.dof_state[..., 1])
            xpos_log.append(core_env.simulator._rigid_body_pos.reshape(num_envs, -1, 3))

            ooo = {k: v for k, v in observation.items() if k != "history"}
            _episode.add(
                tree_map(lambda x: x.cpu(), ooo),
                reward.cpu(),
                action.cpu(),
                terminated.cpu(),
                truncated.cpu(),
                tree_map(lambda x: x.cpu(), info),
            )

        episode_data = _episode.get()
        episode_data["xpos"] = torch.stack(xpos_log)[:-1]
        joint_pos = torch.stack(joint_pos)
        joint_vel = torch.stack(joint_vel)

        for m_id, envs_with_current_motion in motion_to_envs.items():
            for motion_repetition, env_id in enumerate(envs_with_current_motion):
                # A motion was executed on multiple environments
                # we average the results
                _joint_pos = joint_pos[: ctx_dict[m_id].shape[0] + 1, env_id]
                _target_joint_pos = tracking_joint_pos[m_id]
                local_metrics = _calc_metrics(
                    {
                        # "xpos": episode_data["xpos"][0 : ctx_dict[m_id].shape[0], env_id],
                        # "target_xpos": target_xpos_dict[env_id],
                        "tracking_target": tracking_targets[m_id],
                        "motion_id": m_id,
                        "motion_file": core_env._motion_lib.curr_motion_keys[m_id],
                        "observation": tree_map(lambda x: x[0 : ctx_dict[m_id].shape[0] + 1, env_id], episode_data["observation"]),
                        "joint_pos": _joint_pos,
                        "target_joint_pos": _target_joint_pos,
                    },
                )

                # Rename the metrics so that repetitions do not overlap
                assert len(local_metrics) == 1
                metric_key = list(local_metrics.keys())[0]

                if include_results_from_all_envs:
                    metrics[f"{metric_key}_repetition#{motion_repetition}"] = local_metrics[metric_key]
                else:
                    # Retain the original motion name without "repetition#"
                    # Note that this means that motions with the same name get overriden, so you only get one result
                    metrics[metric_key] = local_metrics[metric_key]
                    break

    return metrics


def distance_matrix(X: torch.Tensor, Y: torch.Tensor):
    X_norm = X.pow(2).sum(1).reshape(-1, 1)
    Y_norm = Y.pow(2).sum(1).reshape(1, -1)
    val = X_norm + Y_norm - 2 * torch.matmul(X, Y.T)
    return torch.sqrt(torch.clamp(val, min=0))


def emd_numpy(next_obs: torch.Tensor, tracking_target: torch.Tensor, prefix=""):
    if not torch.isfinite(next_obs).all():
        raise ValueError(f"Cannot compute {prefix}EMD: rollout observation contains non-finite values")
    if not torch.isfinite(tracking_target).all():
        raise ValueError(f"Cannot compute {prefix}EMD: tracking target contains non-finite values")
    # keep only pose part of the observations
    agent_obs = next_obs.to("cpu")
    tracked_obs = tracking_target.to("cpu")
    # compute optimal transport cost
    cost_matrix = distance_matrix(agent_obs, tracked_obs).cpu().detach().numpy()
    if not np.isfinite(cost_matrix).all():
        raise ValueError(f"Cannot compute {prefix}EMD: cost matrix contains non-finite values")
    X_pot = np.ones(agent_obs.shape[0]) / agent_obs.shape[0]
    Y_pot = np.ones(tracked_obs.shape[0]) / tracked_obs.shape[0]
    transport_cost = ot.emd2(X_pot, Y_pot, cost_matrix, numItermax=100000)
    if not np.isfinite(transport_cost):
        raise ValueError(f"Cannot compute {prefix}EMD: POT returned non-finite transport cost {transport_cost}")
    return {f"{prefix}emd": transport_cost}


def distance_proximity(next_obs: torch.Tensor, tracking_target: torch.Tensor, bound: float = 2.0, margin: float = 2, prefix=""):
    stats = {}
    dist = torch.norm(next_obs - tracking_target, dim=-1)
    in_bounds_mask = dist <= bound
    out_bounds_mask = dist > bound + margin
    stats[f"{prefix}proximity"] = (in_bounds_mask + ((bound + margin - dist) / margin) * (~in_bounds_mask) * (~out_bounds_mask)).mean()
    stats[f"{prefix}distance"] = dist.mean()
    return stats


def _calc_metrics(ep):
    metr = {}
    observation_state = torch.as_tensor(ep["observation"]["state"], dtype=torch.float32)
    target_state = torch.as_tensor(ep["tracking_target"]["state"], dtype=torch.float32)
    joint_pos = torch.as_tensor(ep["joint_pos"], dtype=torch.float32)
    target_joint_pos = torch.as_tensor(ep["target_joint_pos"], dtype=torch.float32)
    num_dofs = target_joint_pos.shape[-1] if target_joint_pos.ndim > 0 else 0

    shape_errors = []
    if num_dofs <= 0:
        shape_errors.append("num_dofs must be greater than zero")
    if observation_state.ndim != 2:
        shape_errors.append("observation state must be two-dimensional [T, D]")
    elif observation_state.shape[-1] < num_dofs:
        shape_errors.append("observation state feature dimension is smaller than num_dofs")
    if target_state.ndim != 2:
        shape_errors.append("tracking target state must be two-dimensional [T, D]")
    elif target_state.shape[-1] < num_dofs:
        shape_errors.append("tracking target state feature dimension is smaller than num_dofs")
    if joint_pos.ndim != 2:
        shape_errors.append("joint_pos must be two-dimensional [T, D]")
    elif joint_pos.shape[-1] <= 0:
        shape_errors.append("joint_pos feature dimension must be greater than zero")
    if target_joint_pos.ndim != 2:
        shape_errors.append("target_joint_pos must be two-dimensional [T, D]")
    elif target_joint_pos.shape[-1] <= 0:
        shape_errors.append("target_joint_pos feature dimension must be greater than zero")
    if joint_pos.ndim == 2 and target_joint_pos.ndim == 2 and joint_pos.shape != target_joint_pos.shape:
        shape_errors.append("joint_pos and target_joint_pos shapes must match")
    if joint_pos.ndim == 2 and joint_pos.shape[0] < 3:
        shape_errors.append("joint_pos must contain at least 3 time steps")
    if target_joint_pos.ndim == 2 and target_joint_pos.shape[0] < 3:
        shape_errors.append("target_joint_pos must contain at least 3 time steps")
    if observation_state.ndim == 2 and target_state.ndim == 2 and observation_state.shape[0] != target_state.shape[0]:
        shape_errors.append("observation and tracking target time dimensions must match")
    if observation_state.ndim == 2 and joint_pos.ndim == 2 and observation_state.shape[0] != joint_pos.shape[0]:
        shape_errors.append("observation state and joint_pos time dimensions must match")
    if target_state.ndim == 2 and target_joint_pos.ndim == 2 and target_state.shape[0] != target_joint_pos.shape[0]:
        shape_errors.append("tracking target state and target_joint_pos time dimensions must match")
    if shape_errors:
        raise ValueError(
            "Invalid tracking metric shapes: "
            f"observation_state.shape={tuple(observation_state.shape)}, "
            f"tracking_target_state.shape={tuple(target_state.shape)}, "
            f"joint_pos.shape={tuple(joint_pos.shape)}, "
            f"target_joint_pos.shape={tuple(target_joint_pos.shape)}, "
            f"num_dofs={num_dofs}; "
            + "; ".join(shape_errors)
        )

    finite_inputs = {
        "observation_state": observation_state,
        "tracking_target_state": target_state,
        "joint_pos": joint_pos,
        "target_joint_pos": target_joint_pos,
    }
    nonfinite = [name for name, value in finite_inputs.items() if not torch.isfinite(value).all()]
    if nonfinite:
        raise ValueError(
            "Tracking metrics contain non-finite inputs: "
            f"motion_id={ep['motion_id']}, motion_file={ep['motion_file']}, tensors={nonfinite}"
        )

    next_qpos = observation_state[..., :num_dofs]
    target_qpos = target_state[..., :num_dofs]
    dist_prox_res = distance_proximity(next_obs=next_qpos, tracking_target=target_qpos, prefix="obs_state_")
    metr.update(dist_prox_res)
    emd_res = emd_numpy(next_obs=next_qpos, tracking_target=target_qpos, prefix="obs_state_")
    metr.update(emd_res)
    # # add distance wrt xpos
    # xpos = torch.tensor(ep["xpos"], dtype=torch.float32).view(ep["xpos"].shape[0], -1)  # [keyframes, 72]
    # target_xpos = torch.tensor(ep["target_xpos"], dtype=torch.float32).view(ep["target_xpos"].shape[0], -1)
    # target_xpos = target_xpos[:tracking_length]

    # dist_prox_res = distance_proximity(next_obs=xpos, tracking_target=target_xpos)
    # for k, v in dist_prox_res.items():
    #     metr[f"{k}_xpos"] = v
    # # import ipdb; ipdb.set_trace()
    # metr["emd_xpos"] = emd_numpy(next_obs=xpos, tracking_target=target_xpos)["emd"]
    # jpos_pred = xpos.reshape(xpos.shape[0], -1, 3)  # length x 23 x 3
    # jpos_gt = target_xpos.reshape(target_xpos.shape[0], -1, 3)  # length x 23 x 3
    # metr["success_phc_linf_xpos"] = torch.all(torch.norm(jpos_pred - jpos_gt, dim=-1) <= 0.5).float()
    # metr["success_phc_mean_xpos"] = torch.all(torch.norm(jpos_pred - jpos_gt, dim=-1).mean(dim=-1) <= 0.5).float()

    # phc metrics
    phc_metrics = compute_joint_pos_metrics(joint_pos=joint_pos, target_joint_pos=target_joint_pos)
    metr.update(phc_metrics)
    for k, v in metr.items():
        if isinstance(v, torch.Tensor):
            metr[k] = v.tolist()
    metr["motion_id"] = ep["motion_id"]
    metr["motion_file"] = ep["motion_file"]
    return {ep["motion_file"]: metr}


def compute_joint_pos_metrics(joint_pos, target_joint_pos):
    stats = {}
    # Next observation should match the desired target (if possible in 1 step)
    stats["mpjpe_l"] = torch.norm(joint_pos - target_joint_pos, dim=-1).mean(-1) * 1000

    # We compute velocity as a finite difference along the time dimension.
    vel_gt = torch.diff(target_joint_pos, dim=-2)
    vel_pred = torch.diff(joint_pos, dim=-2)
    stats["vel_dist"] = torch.norm(vel_pred - vel_gt, dim=-1).mean(-1) * 1000

    # We compute acceleration as a second finite difference along time.
    accel_gt = torch.diff(target_joint_pos, n=2, dim=-2)
    accel_pred = torch.diff(joint_pos, n=2, dim=-2)
    stats["accel_dist"] = torch.norm(accel_pred - accel_gt, dim=-1).mean(-1) * 100

    stats.update(
        distance_proximity(next_obs=joint_pos, tracking_target=target_joint_pos, prefix="")
    )
    stats.update(
        emd_numpy(next_obs=joint_pos, tracking_target=target_joint_pos, prefix="")
    )
    return stats
