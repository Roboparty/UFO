import os
import copy
import torch
from torch import nn
import numpy as np
import random

from typing import Any, List, Dict
from termcolor import colored
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from humanoidverse.utils.reference_observations import reference_base_ang_vel

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def pre_process_config(config) -> None:
    
    # compute observation_dim
    # config.robot.policy_obs_dim = -1
    # config.robot.critic_obs_dim = -1
    
    obs_dim_dict = dict()
    _obs_key_list = config.env.config.obs.obs_dict
    _aux_obs_key_list = config.env.config.obs.obs_auxiliary
    
    assert set(config.env.config.obs.noise_scales.keys()) == set(config.env.config.obs.obs_scales.keys())

    # convert obs_dims to list of dicts
    # import ipdb; ipdb.set_trace()
    import omegaconf
    from omegaconf import ListConfig
    if isinstance(config.env.config.obs.obs_dims, ListConfig):
        each_dict_obs_dims = {k: v for d in config.env.config.obs.obs_dims for k, v in d.items()}
        config.env.config.obs.obs_dims = each_dict_obs_dims
    logger.info(f"obs_dims: {config.env.config.obs.obs_dims}")
    auxiliary_obs_dims = {}
    for aux_obs_key, aux_config in _aux_obs_key_list.items():
        auxiliary_obs_dims[aux_obs_key] = 0
        for _key, _num in aux_config.items():
            try:
                assert _key in config.env.config.obs.obs_dims.keys()
                auxiliary_obs_dims[aux_obs_key] += config.env.config.obs.obs_dims[_key] * _num
            except:
                # import ipdb; ipdb.set_trace()
                logger.warning(f"aux_obs_key: {aux_obs_key} not found in obs_dims")
    logger.info(f"auxiliary_obs_dims: {auxiliary_obs_dims}")
    for obs_key, obs_config in _obs_key_list.items():
        obs_dim_dict[obs_key] = 0
        for key in obs_config:
            if key.endswith("_raw"): key = key[:-4]
            if key in config.env.config.obs.obs_dims.keys(): 
                obs_dim_dict[obs_key] += config.env.config.obs.obs_dims[key]
                logger.info(f"{obs_key}: {key} has dim: {config.env.config.obs.obs_dims[key]}")
            else:
                obs_dim_dict[obs_key] += auxiliary_obs_dims[key]
                logger.info(f"{obs_key}: {key} has dim: {auxiliary_obs_dims[key]}")
    config.robot.algo_obs_dim_dict = obs_dim_dict
    
    OmegaConf.set_struct(config.env.config.obs.obs_dims, False)
    config.env.config.obs.obs_dims.update(auxiliary_obs_dims) # ZL: adding auxiliary obs dims to obs_dims
    
    logger.info(f"algo_obs_dim_dict: {config.robot.algo_obs_dim_dict}")

    # compute action_dim for ppo
    # for agent in config.algo.config.network_dict.keys():
    #     for network in config.algo.config.network_dict[agent].keys():
    #         output_dim = config.algo.config.network_dict[agent][network].output_dim
    #         if output_dim == "action_dim":
    #             config.algo.config.network_dict[agent][network].output_dim = config.env.config.robot.actions_dim
                
    # print the config
    # logger.debug(f"PPO CONFIG")
    # logger.debug(f"{config.algo.config.module_dict}")
    # logger.debug(f"{config.algo.config.network_dict}")

def parse_observation(cls: Any, 
                      key_list: List, 
                      buf_dict: Dict, 
                      obs_scales: Dict, 
                      noise_scales: Dict,
                      current_noise_curriculum_value: Any,
                      use_noise: bool = True) -> None:
    """ Parse observations for the legged_robot_base class
    """
    # TOOD: Parse observations for manipulation tasks

    for obs_key in key_list:
        if use_noise:
            obs_noise = noise_scales[obs_key] * current_noise_curriculum_value
        else:
            obs_noise = 0.
        
        # print(f"obs_key: {obs_key}, obs_noise: {obs_noise}")
        
        actor_obs = getattr(cls, f"_get_obs_{obs_key}")().clone()
        obs_scale = obs_scales[obs_key]
        
       
        buf_dict[obs_key] = (actor_obs + (torch.rand_like(actor_obs)* 2. - 1.) * obs_noise) * obs_scale


def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else: 
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)

class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


def get_backward_observation(env, motion_id, use_root_height_obs: bool = False, velocity_multiplier: float = 1.0) -> torch.Tensor:
    from humanoidverse.utils.torch_utils import quat_rotate_inverse
    from humanoidverse.envs.motion_observations import compute_humanoid_observations_max, compute_humanoid_observations_max_with_contact
    
    motion_times = torch.arange(int(np.ceil((env._motion_lib._motion_lengths[motion_id]/env.dt).cpu()))).to(env.device) * env.dt
    
    # get blend motion state
    motion_state = env._motion_lib.get_motion_state(motion_id, motion_times)

    ref_body_pos = motion_state["rg_pos_t"]
    ref_body_rots = motion_state["rg_rot_t"]
    ref_body_vels = motion_state["body_vel_t"] * velocity_multiplier
    ref_body_angular_vels = motion_state["body_ang_vel_t"] * velocity_multiplier
    ref_dof_pos = motion_state["dof_pos"] - env.default_dof_pos[0]
    ref_dof_vel = motion_state["dof_vel"] * velocity_multiplier

    # construct observation
    if env.use_contact_in_obs_max:
        contact_binary = env.foot_contact_detect(ref_body_pos, ref_body_vels)
        obs_dict = compute_humanoid_observations_max_with_contact(
            ref_body_pos,
            ref_body_rots,
            ref_body_vels,
            ref_body_angular_vels,
            local_root_obs=True,
            root_height_obs=use_root_height_obs,
            contact_binary=contact_binary
        )
    else:
        obs_dict = compute_humanoid_observations_max(
            ref_body_pos,
            ref_body_rots,
            ref_body_vels,
            ref_body_angular_vels,
            local_root_obs=True,
            root_height_obs=use_root_height_obs,
        )
    max_local_self_obs = torch.cat([v for v in obs_dict.values()], dim=-1)

    if env.config.obs.use_obs_filter:
        base_quat = ref_body_rots[:, 0]  # root orientation
        ref_ang_vel = reference_base_ang_vel(env, base_quat, ref_body_angular_vels[:, 0])
        projected_gravity = quat_rotate_inverse(
            base_quat,
            env.gravity_vec[0:1].repeat(max_local_self_obs.shape[0], 1),
            w_last=True
        )
        bogus_actions = torch.zeros_like(ref_dof_pos)

        bogus_history_actor = torch.cat([bogus_actions, ref_ang_vel, ref_dof_pos, ref_dof_vel, projected_gravity], dim=-1).repeat(1, 4)
        ref_dict = {
            "actions": bogus_actions,
            "ref_ang_vel": ref_ang_vel,
            "ref_dof_pos": ref_dof_pos,
            "ref_dof_vel": ref_dof_vel,
            "dof_pos": motion_state["dof_pos"],
            "fake_history": bogus_history_actor,
            "max_local_self_obs": max_local_self_obs,
            "projected_gravity": projected_gravity,
            "ref_body_pos": ref_body_pos,
            "ref_body_rots": ref_body_rots,
            "ref_body_vels": ref_body_vels,
            "ref_body_angular_vels": ref_body_angular_vels
        }
        state = torch.cat([ref_dof_pos,
                    ref_dof_vel,
                    projected_gravity,
                    ref_ang_vel], dim=-1)
        last_action = bogus_actions
        # TODO get obs from raw obs instead of the obs
        ufo_obs = {
            "state": state,
            "last_action": last_action,
            "privileged_state": max_local_self_obs
        }
        return ufo_obs, ref_dict
    else:
        ref_dict = {
            "max_local_self_obs": max_local_self_obs,
        }
        return max_local_self_obs, ref_dict


_SUPPORTED_POLICY_ACTOR_INPUT_KEYS = ("state", "last_action", "history_actor")


def _as_actor_input_keys(raw_keys: Any) -> list[str]:
    if isinstance(raw_keys, str):
        keys = [raw_keys]
    else:
        try:
            keys = [str(key) for key in raw_keys]
        except TypeError as exc:
            raise TypeError("cfg.archi.actor.input_filter.key must be a string or sequence of strings") from exc
    if not keys:
        raise ValueError("cfg.archi.actor.input_filter.key must contain at least one key")
    unsupported = [key for key in keys if key not in _SUPPORTED_POLICY_ACTOR_INPUT_KEYS]
    if unsupported:
        raise ValueError(
            "Unsupported actor input key(s) for policy ONNX export: "
            f"{unsupported}. Supported keys: {list(_SUPPORTED_POLICY_ACTOR_INPUT_KEYS)}"
        )
    return keys


def _get_policy_actor_input_keys(inference_model: nn.Module) -> list[str]:
    try:
        raw_keys = inference_model.cfg.archi.actor.input_filter.key
    except AttributeError as exc:
        raise AttributeError("inference_model.cfg.archi.actor.input_filter.key is required for policy ONNX export") from exc
    return _as_actor_input_keys(raw_keys)


def _obs_space_mapping(inference_model: nn.Module) -> Any:
    if not hasattr(inference_model, "obs_space"):
        raise AttributeError("inference_model must expose obs_space for policy ONNX export")
    obs_space = inference_model.obs_space
    if hasattr(obs_space, "spaces"):
        return obs_space.spaces
    if isinstance(obs_space, dict):
        return obs_space
    raise TypeError(
        "inference_model.obs_space must be a Dict-like space with a .spaces mapping "
        f"or a dict, got {type(obs_space)}"
    )


def _require_1d_box_like_space(space: Any, key: str) -> int:
    shape = getattr(space, "shape", None)
    if shape is None:
        raise TypeError(f"obs_space['{key}'] must be a 1D Box-like space with shape, got {type(space)}")
    shape = tuple(shape)
    if len(shape) != 1:
        raise ValueError(f"obs_space['{key}'] must be 1D, got shape={shape}")
    return int(shape[0])


def _actor_input_dims_from_obs_space(inference_model: nn.Module, actor_input_keys: list[str]) -> dict[str, int]:
    spaces = _obs_space_mapping(inference_model)
    dims: dict[str, int] = {}
    for key in actor_input_keys:
        if key not in spaces:
            raise KeyError(f"Actor input key '{key}' is missing from inference_model.obs_space")
        dims[key] = _require_1d_box_like_space(spaces[key], key)
    return dims


def _actor_filter_output_dim(inference_model: nn.Module) -> int:
    try:
        output_space = inference_model._actor.input_filter.output_space
    except AttributeError as exc:
        raise AttributeError("inference_model._actor.input_filter.output_space is required for policy ONNX export") from exc
    return _require_1d_box_like_space(output_space, "_actor.input_filter.output_space")


def _policy_output_action_dim(inference_model: nn.Module) -> int:
    if not hasattr(inference_model, "action_dim"):
        raise AttributeError("inference_model.action_dim is required for policy ONNX export metadata")
    return int(inference_model.action_dim)


def _infer_policy_onnx_export_spec(inference_model: nn.Module, z_dim: int) -> dict[str, Any]:
    actor_input_keys = _get_policy_actor_input_keys(inference_model)
    actor_input_dims = _actor_input_dims_from_obs_space(inference_model, actor_input_keys)
    actor_input_dim = sum(actor_input_dims.values())
    filter_output_dim = _actor_filter_output_dim(inference_model)
    if actor_input_dim != filter_output_dim:
        raise ValueError(
            "Actor input dims inferred from obs_space do not match _actor.input_filter.output_space: "
            f"sum(actor_input_dims)={actor_input_dim}, output_space_dim={filter_output_dim}, "
            f"actor_input_dims={actor_input_dims}"
        )
    return {
        "actor_input_keys": actor_input_keys,
        "actor_input_dims": actor_input_dims,
        "z_dim": int(z_dim),
        "actor_input_dim": actor_input_dim,
        "actor_obs_dim": actor_input_dim + int(z_dim),
        "output_action_dim": _policy_output_action_dim(inference_model),
    }


def export_meta_policy_as_onnx(
    inference_model,
    path,
    exported_policy_name,
    example_obs_dict: dict[str, torch.Tensor] | None = None,
    z_dim: int | None = None,
    *,
    batch_size: int = 1,
    opset_version: int = 13,
):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, exported_policy_name)
    if z_dim is None:
        try:
            z_dim = int(inference_model.cfg.archi.z_dim)
        except AttributeError as exc:
            raise AttributeError("z_dim must be provided or available at inference_model.cfg.archi.z_dim") from exc
    z_dim = int(z_dim)
    if z_dim <= 0:
        raise ValueError(f"z_dim must be positive, got {z_dim}")
    export_spec = _infer_policy_onnx_export_spec(inference_model, z_dim)
    inference_model = inference_model.eval()
    actor = copy.deepcopy(inference_model).to("cpu")

    class PPOWrapper(nn.Module):
        def __init__(self, actor, actor_input_keys, actor_input_dims, z_dim):
            """
            model: The original PyTorch model.
            input_keys: List of input names as keys for the input dictionary.
            """
            super(PPOWrapper, self).__init__()
            self.actor = actor
            self.actor_input_keys = list(actor_input_keys)
            self.actor_input_dims = dict(actor_input_dims)
            self.z_dim = int(z_dim)

        def forward(self, actor_obs):
            """
            Dynamically creates a dictionary from the input keys and args.
            """
            actor_obs_without_z = actor_obs[:, :-self.z_dim]
            ctx = actor_obs[:, -self.z_dim:]
            actor_dict = {}
            cursor = 0
            for key in self.actor_input_keys:
                dim = self.actor_input_dims[key]
                actor_dict[key] = actor_obs_without_z[:, cursor : cursor + dim]
                cursor += dim

            return self.actor.act(actor_dict, ctx)

    wrapper = PPOWrapper(
        actor,
        actor_input_keys=export_spec["actor_input_keys"],
        actor_input_dims=export_spec["actor_input_dims"],
        z_dim=export_spec["z_dim"],
    )
    if example_obs_dict is None:
        example_input_list = torch.randn(int(batch_size), export_spec["actor_obs_dim"], dtype=torch.float32)
    else:
        if "actor_obs" not in example_obs_dict:
            raise KeyError("example_obs_dict must contain 'actor_obs'")
        example_input_list = example_obs_dict["actor_obs"]
        if int(example_input_list.shape[-1]) != export_spec["actor_obs_dim"]:
            raise ValueError(
                "example_obs_dict['actor_obs'] has wrong last dimension: "
                f"expected {export_spec['actor_obs_dim']}, got {int(example_input_list.shape[-1])}"
            )
    torch.onnx.export(
        wrapper,
        example_input_list,  # Pass x1 and x2 as separate inputs
        path,
        verbose=False,
        input_names=["actor_obs"],  # Specify the input names
        output_names=["action"],  # Name the output
        dynamic_axes={
            "actor_obs": {0: "batch"},
            "action": {0: "batch"},
        },
        opset_version=int(opset_version),  # Specify the opset version, if needed
    )
    return export_spec
