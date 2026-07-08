"""Reward relabeling utilities used by MJLab reward inference.

This module is intentionally self-contained and does not import the legacy G1 gym environment. It only reuses the local MuJoCo reward functions.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import inspect
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any

import mujoco
import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.buffers.trajectory import TrajectoryDictBufferMultiDim
from humanoidverse.envs.g1_env_helper import rewards as g1_rewards
from humanoidverse.envs.g1_env_helper.rewards import RewardFunction


def get_next(field: str, data: Any):
    if "next" in data and field in data["next"]:
        return data["next"][field]
    if f"next_{field}" in data:
        return data[f"next_{field}"]
    raise ValueError(f"No next of {field} found in data.")


def to_torch(x: np.ndarray | torch.Tensor, device: torch.device | str, dtype: torch.dtype):
    if len(x.shape) == 1:
        x = x[None, ...]
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, device=device, dtype=dtype)
    return x.to(device=device, dtype=dtype)


def make_reward_from_name(name: str | None) -> RewardFunction:
    for _class_name, reward_cls in inspect.getmembers(g1_rewards, inspect.isclass):
        if not issubclass(reward_cls, RewardFunction) or inspect.isabstract(reward_cls):
            continue
        reward_obj = reward_cls.reward_from_name(name)
        if reward_obj is not None:
            return reward_obj
    raise ValueError(f"Unknown reward name: {name}")


@dataclasses.dataclass(kw_only=True)
class BaseMjlabRewardWrapper:
    model: Any
    numpy_output: bool = True
    _dtype: torch.dtype = dataclasses.field(default_factory=lambda: torch.float32)

    def act(
        self,
        obs: torch.Tensor | np.ndarray,
        z: torch.Tensor | np.ndarray,
        mean: bool = True,
    ) -> torch.Tensor:
        obs = tree_map(lambda x: to_torch(x, device=self.device, dtype=self._dtype), obs)
        z = to_torch(z, device=self.device, dtype=self._dtype)
        if self.numpy_output:
            return self.unwrapped_model.act(obs, z, mean).float().cpu().detach().numpy()
        return self.unwrapped_model.act(obs, z, mean)

    @property
    def device(self) -> Any:
        return self.unwrapped_model.device

    @property
    def unwrapped_model(self):
        if hasattr(self.model, "unwrapped_model"):
            return self.model.unwrapped_model
        return self.model

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __deepcopy__(self, memo):
        return type(self)(model=copy.deepcopy(self.model, memo), numpy_output=self.numpy_output, _dtype=copy.deepcopy(self._dtype))

    def __getstate__(self):
        return {
            "model": self.model,
            "numpy_output": self.numpy_output,
            "_dtype": self._dtype,
        }

    def __setstate__(self, state):
        self.model = state["model"]
        self.numpy_output = state["numpy_output"]
        self._dtype = state["_dtype"]


@dataclasses.dataclass(kw_only=True)
class RewardWrapperHV(BaseMjlabRewardWrapper):
    inference_dataset: Any
    num_samples_per_inference: int
    inference_function: str
    max_workers: int
    process_executor: bool = False
    process_context: str = "spawn"
    env_model: str | mujoco.MjModel = "humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml"

    def reward_inference(self, task: str) -> torch.Tensor:
        if isinstance(self.env_model, str):
            self.env_model = mujoco.MjModel.from_xml_path(self.env_model)

        if isinstance(self.inference_dataset, TrajectoryDictBufferMultiDim):
            if "qpos" not in self.inference_dataset.output_key_tp1:
                self.inference_dataset.output_key_tp1.append("qpos")
            if "qvel" not in self.inference_dataset.output_key_tp1:
                self.inference_dataset.output_key_tp1.append("qvel")

        if self.num_samples_per_inference >= self.inference_dataset.size() and hasattr(self.inference_dataset, "get_full_buffer"):
            data = self.inference_dataset.get_full_buffer()
        else:
            data = self.inference_dataset.sample(self.num_samples_per_inference)

        qpos = get_next("qpos", data)
        qvel = get_next("qvel", data)
        action = data["action"]
        if isinstance(qpos, torch.Tensor):
            qpos = qpos.cpu().detach().numpy()
            qvel = qvel.cpu().detach().numpy()
            action = action.cpu().detach().numpy()

        rewards = relabel(
            self.env_model,
            qpos,
            qvel,
            action,
            make_reward_from_name(task),
            max_workers=self.max_workers,
            process_executor=self.process_executor,
            process_context=self.process_context,
        )

        td = {"reward": torch.tensor(rewards, dtype=torch.float32, device=self.device)}
        if "B" in data:
            td["B_vect"] = data["B"]
        else:
            td["next_obs"] = get_next("observation", data)
        inference_fn = getattr(self.model, self.inference_function, None)
        if inference_fn is None:
            raise AttributeError(f"Model does not define {self.inference_function!r}")
        return inference_fn(**td).reshape(1, -1)

    def __deepcopy__(self, memo):
        return type(self)(
            model=copy.deepcopy(self.model, memo),
            numpy_output=self.numpy_output,
            _dtype=copy.deepcopy(self._dtype),
            inference_dataset=copy.deepcopy(self.inference_dataset),
            num_samples_per_inference=self.num_samples_per_inference,
            inference_function=self.inference_function,
            max_workers=self.max_workers,
            process_executor=self.process_executor,
            process_context=self.process_context,
            env_model=copy.deepcopy(self.env_model, memo),
        )

    def __getstate__(self):
        return {
            "model": self.model,
            "numpy_output": self.numpy_output,
            "_dtype": self._dtype,
            "inference_dataset": self.inference_dataset,
            "num_samples_per_inference": self.num_samples_per_inference,
            "inference_function": self.inference_function,
            "max_workers": self.max_workers,
            "process_executor": self.process_executor,
            "process_context": self.process_context,
            "env_model": self.env_model,
        }

    def __setstate__(self, state):
        self.model = state["model"]
        self.numpy_output = state["numpy_output"]
        self._dtype = state["_dtype"]
        self.inference_dataset = state["inference_dataset"]
        self.num_samples_per_inference = state["num_samples_per_inference"]
        self.inference_function = state["inference_function"]
        self.max_workers = state["max_workers"]
        self.process_executor = state["process_executor"]
        self.process_context = state["process_context"]
        self.env_model = state["env_model"]


def _relabel_worker(
    x,
    model: mujoco.MjModel,
    reward_fn: RewardFunction,
):
    qpos, qvel, action = x
    assert len(qpos.shape) > 1
    assert qvel.shape[0] == qpos.shape[0]
    assert qvel.shape[0] == action.shape[0]
    rewards = np.zeros((qpos.shape[0], 1))
    for i in range(qpos.shape[0]):
        rewards[i] = reward_fn(model, qpos[i], qvel[i], action[i])
    return rewards


def relabel(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    action: np.ndarray,
    reward_fn: RewardFunction,
    max_workers: int = 5,
    process_executor: bool = False,
    process_context: str = "spawn",
):
    chunk_size = int(np.ceil(qpos.shape[0] / max_workers))
    args = [(qpos[i : i + chunk_size], qvel[i : i + chunk_size], action[i : i + chunk_size]) for i in range(0, qpos.shape[0], chunk_size)]
    if max_workers == 1:
        result = [_relabel_worker(args[0], model=model, reward_fn=reward_fn)]
    elif process_executor:
        import multiprocessing

        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=multiprocessing.get_context(process_context),
        ) as exe:
            f = functools.partial(_relabel_worker, model=model, reward_fn=reward_fn)
            result = exe.map(f, args)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            f = functools.partial(_relabel_worker, model=model, reward_fn=reward_fn)
            result = exe.map(f, args)

    return np.concatenate([r for r in result])
