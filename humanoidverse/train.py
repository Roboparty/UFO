"""UFO training entrypoint.

UFO provides FB and TLDR unsupervised RL presets for humanoid control.
Defaults are kept in this file; command-line arguments can override them.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path


def _ensure_compile_cache(cache_root: str | Path | None = None) -> None:
    cache_dir = os.environ.get("UFO_CACHE_DIR") or os.environ.get("BFMZERO_MJLAB_CACHE_DIR")
    root = Path(cache_dir or cache_root or Path.cwd() / "cache").expanduser()
    for key, subdir in {
        "TMPDIR": "tmp",
        "TEMP": "tmp",
        "TMP": "tmp",
        "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
        "TRITON_CACHE_DIR": "triton",
        "CUDA_CACHE_PATH": "cuda",
        "WARP_CACHE_PATH": "warp",
    }.items():
        path = root / subdir
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(key, str(path))


_ensure_compile_cache()

DEFAULT_AGENT = "fb"
DEFAULT_NUM_ENVS = 1024
DEFAULT_NUM_ENV_STEPS = 192000000
DEFAULT_CHECKPOINT_EVERY_STEPS = 3200000
DEFAULT_DATA_PATH = "humanoidverse/data/lafan_29dof_10s-clipped.pkl"
DEFAULT_WORK_DIR = "runs/ufo"
DEFAULT_BUFFER_SIZE = 5120000
DEFAULT_UPDATE_Z_EVERY_STEP = 100
DEFAULT_WANDB_PROJECT = "ufo-humanoid"

from humanoidverse.agents.envs.humanoidverse_mjlab import G1_MJLAB_MJCF_PATH, HumanoidVerseMjlabConfig
from humanoidverse.agents.evaluations.humanoidverse_mjlab import HumanoidVerseMjlabTrackingEvaluationConfig
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig, FBcprAuxAgentTrainConfig
from humanoidverse.agents.fb_cpr_aux.model import FBcprAuxModelArchiConfig, FBcprAuxModelConfig
from humanoidverse.agents.gcr_rl_dist_aux.model import GcrRlDistAuxModelArchiConfig, GcrRlDistAuxModelConfig
from humanoidverse.agents.tldr_dist_aux.agent import TldrDistAuxAgentConfig, TldrDistAuxAgentTrainConfig
from humanoidverse.agents.nn_filters import DictInputFilterConfig
from humanoidverse.agents.nn_models import (
    ActorArchiConfig,
    BackwardArchiConfig,
    DiscriminatorArchiConfig,
    ForwardArchiConfig,
    RewardNormalizerConfig,
)
from humanoidverse.agents.normalizers import BatchNormNormalizerConfig, ObsNormalizerConfig
from humanoidverse.training.workspace import TrainConfig


def build_bfm_zero_agent(
    device: str = "cuda",
    compile: bool = True,
    update_z_every_step: int = DEFAULT_UPDATE_Z_EVERY_STEP,
    lr_scale: float = 1.0,
    clip_grad_norm: float = 0.0,
    cartwheel_aux_safe: bool = False,
) -> FBcprAuxAgentConfig:
    if cartwheel_aux_safe:
        aux_rewards = [
            "penalty_torques",
            "penalty_action_rate",
            "limits_dof_pos",
            "limits_torque",
        ]
        aux_rewards_scaling = {
            "penalty_action_rate": -0.03,
            "limits_dof_pos": -10.0,
            "penalty_torques": 0.0,
            "limits_torque": 0.0,
        }
    else:
        aux_rewards = [
            "penalty_torques",
            "penalty_action_rate",
            "limits_dof_pos",
            "limits_torque",
            "penalty_undesired_contact",
            "penalty_feet_ori",
            "penalty_ankle_roll",
            "penalty_slippage",
        ]
        aux_rewards_scaling = {
            "penalty_action_rate": -0.1,
            "penalty_feet_ori": -0.4,
            "penalty_ankle_roll": -4.0,
            "limits_dof_pos": -10.0,
            "penalty_slippage": -2.0,
            "penalty_undesired_contact": -1.0,
            "penalty_torques": 0.0,
            "limits_torque": 0.0,
        }

    return FBcprAuxAgentConfig(
        name="FBcprAuxAgent",
        model=FBcprAuxModelConfig(
            name="FBcprAuxModel",
            device=device,
            archi=FBcprAuxModelArchiConfig(
                name="FBcprAuxModelArchiConfig",
                z_dim=256,
                norm_z=True,
                f=ForwardArchiConfig(
                    name="ForwardArchi",
                    hidden_dim=2048,
                    model="residual",
                    hidden_layers=6,
                    embedding_layers=2,
                    num_parallel=2,
                    ensemble_mode="batch",
                    input_filter=DictInputFilterConfig(
                        name="DictInputFilterConfig", key=["state", "privileged_state", "last_action", "history_actor"]
                    ),
                ),
                b=BackwardArchiConfig(
                    name="BackwardArchi",
                    hidden_dim=256,
                    hidden_layers=1,
                    norm=True,
                    input_filter=DictInputFilterConfig(name="DictInputFilterConfig", key=["state", "privileged_state"]),
                ),
                actor=ActorArchiConfig(
                    name="actor",
                    model="residual",
                    hidden_dim=2048,
                    hidden_layers=6,
                    embedding_layers=2,
                    input_filter=DictInputFilterConfig(name="DictInputFilterConfig", key=["state", "last_action", "history_actor"]),
                ),
                critic=ForwardArchiConfig(
                    name="ForwardArchi",
                    hidden_dim=2048,
                    model="residual",
                    hidden_layers=6,
                    embedding_layers=2,
                    num_parallel=2,
                    ensemble_mode="batch",
                    input_filter=DictInputFilterConfig(
                        name="DictInputFilterConfig", key=["state", "privileged_state", "last_action", "history_actor"]
                    ),
                ),
                discriminator=DiscriminatorArchiConfig(
                    name="DiscriminatorArchi",
                    hidden_dim=1024,
                    hidden_layers=3,
                    input_filter=DictInputFilterConfig(name="DictInputFilterConfig", key=["state", "privileged_state"]),
                ),
                aux_critic=ForwardArchiConfig(
                    name="ForwardArchi",
                    hidden_dim=2048,
                    model="residual",
                    hidden_layers=6,
                    embedding_layers=2,
                    num_parallel=2,
                    ensemble_mode="batch",
                    input_filter=DictInputFilterConfig(
                        name="DictInputFilterConfig", key=["state", "privileged_state", "last_action", "history_actor"]
                    ),
                ),
            ),
            obs_normalizer=ObsNormalizerConfig(
                name="ObsNormalizerConfig",
                normalizers={
                    "state": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
                    "privileged_state": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
                    "last_action": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
                    "history_actor": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
                },
                allow_mismatching_keys=True,
            ),
            inference_batch_size=500000,
            seq_length=8,
            actor_std=0.05,
            amp=False,
            norm_aux_reward=RewardNormalizerConfig(name="RewardNormalizer", translate=False, scale=True),
        ),
        train=FBcprAuxAgentTrainConfig(
            name="FBcprAuxAgentTrainConfig",
            lr_f=0.0003 * lr_scale,
            lr_b=1e-05 * lr_scale,
            lr_actor=0.0003 * lr_scale,
            weight_decay=0.0,
            clip_grad_norm=clip_grad_norm,
            fb_target_tau=0.01,
            ortho_coef=100.0,
            train_goal_ratio=0.2,
            fb_pessimism_penalty=0.0,
            actor_pessimism_penalty=0.5,
            stddev_clip=0.3,
            q_loss_coef=0.0,
            batch_size=1024,
            discount=0.98,
            use_mix_rollout=True,
            update_z_every_step=int(update_z_every_step),
            z_buffer_size=8192,
            rollout_expert_trajectories=True,
            rollout_expert_trajectories_length=250,
            rollout_expert_trajectories_percentage=0.5,
            lr_discriminator=1e-05 * lr_scale,
            lr_critic=0.0003 * lr_scale,
            critic_target_tau=0.005,
            critic_pessimism_penalty=0.5,
            reg_coeff=0.05,
            scale_reg=True,
            expert_asm_ratio=0.6,
            relabel_ratio=0.8,
            grad_penalty_discriminator=10.0,
            weight_decay_discriminator=0.0,
            lr_aux_critic=0.0003 * lr_scale,
            reg_coeff_aux=0.02,
            aux_critic_pessimism_penalty=0.5,
        ),
        aux_rewards=aux_rewards,
        aux_rewards_scaling=aux_rewards_scaling,
        cudagraphs=False,
        compile=compile,
    )


def build_tldr_dist_aux_agent(device: str = "cuda", compile: bool = True) -> TldrDistAuxAgentConfig:
    return TldrDistAuxAgentConfig(
        name="TldrDistAuxAgent",
        model=GcrRlDistAuxModelConfig(
            name="GcrRlDistAuxModel",
            device=device,
            archi=GcrRlDistAuxModelArchiConfig(
                name="GcrRlDistAuxModelArchiConfig",
                z_dim=256,
                norm_z=True,
                goal_encoder=BackwardArchiConfig(
                    name="BackwardArchi",
                    hidden_dim=256,
                    hidden_layers=1,
                    norm=True,
                    input_filter=DictInputFilterConfig(name="DictInputFilterConfig", key=["state", "privileged_state"]),
                ),
                actor=ActorArchiConfig(
                    name="actor",
                    model="residual",
                    hidden_dim=2048,
                    hidden_layers=6,
                    embedding_layers=2,
                    input_filter=DictInputFilterConfig(name="DictInputFilterConfig", key=["state", "last_action", "history_actor"]),
                ),
                critic=ForwardArchiConfig(
                    name="ForwardArchi",
                    hidden_dim=2048,
                    model="residual",
                    hidden_layers=6,
                    embedding_layers=2,
                    num_parallel=2,
                    ensemble_mode="batch",
                    input_filter=DictInputFilterConfig(
                        name="DictInputFilterConfig", key=["state", "privileged_state", "last_action", "history_actor"]
                    ),
                ),
                discriminator=DiscriminatorArchiConfig(
                    name="DiscriminatorArchi",
                    hidden_dim=1024,
                    hidden_layers=3,
                    input_filter=DictInputFilterConfig(name="DictInputFilterConfig", key=["state", "privileged_state"]),
                ),
                aux_critic=ForwardArchiConfig(
                    name="ForwardArchi",
                    hidden_dim=2048,
                    model="residual",
                    hidden_layers=6,
                    embedding_layers=2,
                    num_parallel=2,
                    ensemble_mode="batch",
                    input_filter=DictInputFilterConfig(
                        name="DictInputFilterConfig", key=["state", "privileged_state", "last_action", "history_actor"]
                    ),
                ),
            ),
            obs_normalizer=ObsNormalizerConfig(
                name="ObsNormalizerConfig",
                normalizers={
                    "state": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
                    "privileged_state": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
                    "last_action": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
                    "history_actor": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
                },
                allow_mismatching_keys=True,
            ),
            inference_batch_size=500000,
            seq_length=8,
            actor_std=0.05,
            amp=False,
            norm_aux_reward=RewardNormalizerConfig(name="RewardNormalizer", translate=False, scale=True),
        ),
        train=TldrDistAuxAgentTrainConfig(
            name="TldrDistAuxAgentTrainConfig",
            train_goal_ratio=0.2,
            expert_asm_ratio=0.6,
            relabel_ratio=0.4,
            lr_goal_encoder=8e-7,
            lr_actor=5e-5,
            lr_critic=5e-5,
            lr_discriminator=2e-6,
            lr_aux_critic=2e-5,
            lr_dual_lam=4e-5,
            weight_decay=0.0,
            weight_decay_discriminator=0.0,
            clip_grad_norm=0.0,
            batch_size=1024,
            discount=0.98,
            stddev_clip=0.3,
            actor_pessimism_penalty=0.5,
            critic_pessimism_penalty=0.5,
            aux_critic_pessimism_penalty=0.5,
            critic_target_tau=0.005,
            use_mix_rollout=True,
            update_z_every_step=10,
            z_buffer_size=8192,
            rollout_expert_trajectories=True,
            rollout_expert_trajectories_length=250,
            rollout_expert_trajectories_percentage=0.5,
            grad_penalty_discriminator=10.0,
            use_tldr_pretrain=True,
            tldr_pretrain_env_steps=200000,
            tldr_te_during_rl=True,
            freeze_goal_encoder_after_pretrain=False,
            dual_reg=True,
            dual_lam_init=3000.0,
            dual_slack=1.0,
            tldr_softplus_scale=500.0,
            tldr_softplus_beta=0.01,
            tldr_reward_scale=1.0,
            goal_encoder_lr_schedule="none",
            goal_encoder_lr_schedule_steps=0,
            goal_encoder_lr_min=1e-6,
            reg_coeff_disc=0.03,
            reg_coeff_aux=0.01,
            disc_reward_coef=0.5,
            scale_reg=True,
        ),
        aux_rewards=[
            "penalty_torques",
            "penalty_action_rate",
            "limits_dof_pos",
            "limits_torque",
            "penalty_undesired_contact",
            "penalty_feet_ori",
            "penalty_ankle_roll",
            "penalty_slippage",
        ],
        aux_rewards_scaling={
            "penalty_action_rate": -0.1,
            "penalty_feet_ori": -0.4,
            "penalty_ankle_roll": -4.0,
            "limits_dof_pos": -10.0,
            "penalty_slippage": -2.0,
            "penalty_undesired_contact": -1.0,
            "penalty_torques": 0.0,
            "limits_torque": 0.0,
        },
        cudagraphs=False,
        compile=compile,
    )


def build_ufo_mjlab_config(
    *,
    device: str,
    work_dir: str,
    num_envs: int,
    num_env_steps: int,
    seed: int,
    use_wandb: bool,
    wandb_run_name: str | None,
    checkpoint_every_steps: int = 9600000,
    distributed_rank: int = 0,
    distributed_world_size: int = 1,
    disable_eval_prioritization: bool = False,
    smoke: bool = False,
    agent: str = DEFAULT_AGENT,
    data_path: str | list[str] | None = None,
    data_mix_weights: list[float] | None = None,
    update_z_every_step: int = DEFAULT_UPDATE_Z_EVERY_STEP,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
    disable_dr: bool = False,
    disable_obs_noise: bool = False,
    lr_scale: float = 1.0,
    clip_grad_norm: float = 0.0,
    cartwheel_aux_safe: bool = False,
) -> TrainConfig:
    evaluations = []
    run_eval_and_prioritization = not smoke and not disable_eval_prioritization
    distributed_sync = distributed_world_size > 1
    if run_eval_and_prioritization:
        evaluations = [
            HumanoidVerseMjlabTrackingEvaluationConfig(
                name="HumanoidVerseMjlabTrackingEvaluationConfig",
                generate_videos=False,
                videos_dir="videos",
                video_name_prefix="unknown_agent",
                name_in_logs="humanoidverse_tracking_eval",
                env=None,
                num_envs=num_envs,
                n_episodes_per_motion=1,
            )
        ]
    agent_device = "cuda" if device.startswith("cuda") else "cpu"
    agent_presets = {
        "fb": {
            "agent_cfg": build_bfm_zero_agent(
                device=agent_device,
                compile=not distributed_sync,
                update_z_every_step=update_z_every_step,
                lr_scale=lr_scale,
                clip_grad_norm=clip_grad_norm,
                cartwheel_aux_safe=cartwheel_aux_safe,
            ),
            "wandb_group": "ufo_fb",
            "wandb_project": DEFAULT_WANDB_PROJECT,
            "train_runtime": {
                "log_every_updates": 384000,
                "update_agent_every": 1024,
                "num_seed_steps": 10240,
                "num_agent_updates": 16,
                "checkpoint_buffer": True,
                "use_trajectory_buffer": True,
                "buffer_size": 5120000,
                "eval_every_steps": 9600000,
            },
        },
        "tldr": {
            "agent_cfg": build_tldr_dist_aux_agent(device=agent_device, compile=not distributed_sync),
            "wandb_group": "ufo_tldr",
            "wandb_project": DEFAULT_WANDB_PROJECT,
            "train_runtime": {
                "log_every_updates": 384000,
                "update_agent_every": 1024,
                "num_seed_steps": 10240,
                "num_agent_updates": 16,
                "checkpoint_buffer": True,
                "use_trajectory_buffer": True,
                "buffer_size": 5120000,
                "eval_every_steps": 9600000,
            },
        },
    }
    selected = agent_presets[agent]
    agent_cfg = selected["agent_cfg"]
    wandb_group = selected["wandb_group"]
    wandb_project = selected["wandb_project"]
    train_runtime = selected["train_runtime"]
    hydra_overrides = [
        "robot=g1/g1_29dof_hard_waist",
        "robot.control.action_scale=0.25",
        "robot.control.action_clip_value=5.0",
        "robot.control.normalize_action_to=5.0",
        "env.config.lie_down_init=True",
        "env.config.lie_down_init_prob=0.3",
    ]
    if cartwheel_aux_safe:
        hydra_overrides.extend(
            [
                "rewards.reward_scales.penalty_undesired_contact=0.0",
                "rewards.reward_scales.penalty_feet_ori=0.0",
                "rewards.reward_scales.feet_heading_alignment=0.0",
                "rewards.reward_scales.penalty_slippage=0.0",
                "rewards.reward_scales.penalty_ankle_roll=0.0",
                "rewards.reward_scales.penalty_action_rate=-0.1",
            ]
        )

    return TrainConfig(
        name="TrainConfig",
        agent=agent_cfg,
        motions="",
        motions_root="",
        env=HumanoidVerseMjlabConfig(
            name="humanoidverse_mjlab",
            device=device,
            lafan_tail_path=data_path or DEFAULT_DATA_PATH,
            data_mix_weights=data_mix_weights,
            mjcf_path=G1_MJLAB_MJCF_PATH,
            max_episode_length_s=None,
            disable_obs_noise=disable_obs_noise,
            disable_domain_randomization=disable_dr,
            relative_config_path="exp/bfm_zero/bfm_zero",
            include_last_action=True,
            hydra_overrides=hydra_overrides,
            context_length=None,
            include_history_actor=True,
            include_history_noaction=False,
            root_height_obs=True,
            auto_reset=False,
            seed=seed,
        ),
        work_dir=work_dir,
        seed=seed,
        online_parallel_envs=num_envs,
        log_every_updates=train_runtime["log_every_updates"],
        num_env_steps=num_env_steps,
        update_agent_every=train_runtime["update_agent_every"],
        num_seed_steps=train_runtime["num_seed_steps"],
        num_agent_updates=train_runtime["num_agent_updates"],
        checkpoint_every_steps=checkpoint_every_steps,
        checkpoint_buffer=train_runtime["checkpoint_buffer"],
        prioritization=run_eval_and_prioritization,
        prioritization_min_val=0.5,
        prioritization_max_val=2.0,
        prioritization_scale=2.0,
        prioritization_mode="exp",
        use_trajectory_buffer=train_runtime["use_trajectory_buffer"],
        buffer_size=int(buffer_size),
        use_wandb=use_wandb,
        wandb_ename=os.environ.get("WANDB_ENTITY", "xuewangusst-1"),
        wandb_gname=wandb_group,
        wandb_pname=wandb_project,
        wandb_run_name=wandb_run_name or f"ufo_{agent}",
        load_expert_data_from_motion_lib=True,
        buffer_device="cuda" if device.startswith("cuda") else "cpu",
        disable_tqdm=True,
        evaluations=evaluations,
        eval_every_steps=train_runtime["eval_every_steps"],
        distributed_rank=distributed_rank,
        distributed_world_size=distributed_world_size,
        rank0_only_writes=True,
        checkpoint_rank_buffers=True,
        distributed_sync=distributed_sync,
        distributed_global_steps=True,
        distributed_average_metrics=True,
        tags={"backend": "mjlab", "agent": agent, "distributed_rank": distributed_rank, "distributed_world_size": distributed_world_size},
    )


def _select_device_and_rank(seed: int) -> tuple[str, int, int, int]:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible == "":
        try:
            import torch

            if torch.cuda.is_available():
                os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
                return "cuda:0", 0, 0, 1
        except Exception:
            pass
        return "cpu", 0, 0, 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    return f"cuda:{local_rank}", local_rank, rank, world_size


def _init_distributed(local_rank: int, world_size: int) -> None:
    if world_size <= 1:
        return
    from datetime import timedelta

    import torch
    import torch.distributed as dist

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        init_kwargs = {
            "backend": "nccl",
            "init_method": "env://",
            "timeout": timedelta(hours=2),
        }
        try:
            init_kwargs["device_id"] = torch.device(f"cuda:{local_rank}")
            dist.init_process_group(**init_kwargs)
        except TypeError:
            init_kwargs.pop("device_id", None)
            dist.init_process_group(**init_kwargs)


def run_train(args: argparse.Namespace, log_dir: Path) -> None:
    device, _local_rank, rank, world_size = _select_device_and_rank(args.seed)
    _init_distributed(_local_rank, world_size)
    seed = args.seed + rank
    cfg = build_ufo_mjlab_config(
        device=device,
        work_dir=str(log_dir),
        num_envs=args.num_envs,
        num_env_steps=args.num_env_steps,
        seed=seed,
        use_wandb=bool(args.use_wandb and rank == 0),
        wandb_run_name=args.wandb_run_name,
        checkpoint_every_steps=args.checkpoint_every_steps,
        distributed_rank=rank,
        distributed_world_size=world_size,
        disable_eval_prioritization=bool(args.disable_eval_prioritization),
        smoke=bool(args.smoke),
        agent=args.agent,
        data_path=args.data_path,
        data_mix_weights=args.data_mix_weights,
        update_z_every_step=args.update_z_every_step,
        buffer_size=args.buffer_size,
        disable_dr=bool(args.disable_dr),
        disable_obs_noise=bool(args.disable_obs_noise),
        lr_scale=args.lr_scale,
        clip_grad_norm=args.clip_grad_norm,
        cartwheel_aux_safe=bool(args.cartwheel_aux_safe),
    )
    print(
        "[INFO] UFO train: "
        f"agent={args.agent}, device={device}, rank={rank}/{world_size}, seed={seed}, work_dir={log_dir}, "
        f"mjcf_path={cfg.env.mjcf_path}, data_path={cfg.env.lafan_tail_path}, data_mix_weights={cfg.env.data_mix_weights}, "
        f"num_envs_per_rank={args.num_envs}, global_parallel_envs={args.num_envs * world_size}, "
        f"num_env_steps_global={args.num_env_steps}, buffer_size_per_rank={cfg.buffer_size}, "
        f"cartwheel_aux_safe={args.cartwheel_aux_safe}, lr_scale={args.lr_scale}, clip_grad_norm={args.clip_grad_norm}, "
        f"disable_dr={cfg.env.disable_domain_randomization}, disable_obs_noise={cfg.env.disable_obs_noise}, "
        f"compile={cfg.agent.compile}",
        flush=True,
    )
    try:
        workspace = cfg.build()
        workspace.train()
    finally:
        if world_size > 1:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()


def launch(args: argparse.Namespace) -> None:
    log_dir = Path(args.work_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    _ensure_compile_cache()
    if args.gpu_ids in (None, "single"):
        run_train(args, log_dir)
        return

    existing_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if args.gpu_ids == "all":
        import torch

        num_gpus = torch.cuda.device_count()
        selected_gpus = None
    else:
        requested = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
        if existing_visible:
            visible = [x.strip() for x in existing_visible.split(",") if x.strip()]
            selected_gpus = [visible[i] for i in requested]
        else:
            selected_gpus = [str(i) for i in requested]
        num_gpus = len(selected_gpus)
    if num_gpus <= 1:
        if selected_gpus is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected_gpus)
        run_train(args, log_dir)
        return

    import torchrunx

    logging.basicConfig(level=logging.INFO)
    if selected_gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected_gpus)
    os.environ.setdefault("TORCHRUNX_LOG_DIR", str(log_dir / "torchrunx"))
    torchrunx.Launcher(
        hostnames=["localhost"],
        workers_per_host=num_gpus,
        backend=None,
        copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY
        + (
            "MUJOCO*",
            "UFO_CACHE_DIR",
            "BFMZERO_MJLAB_CACHE_DIR",
            "UV_CACHE_DIR",
            "PYTHONPYCACHEPREFIX",
            "TMPDIR",
            "TEMP",
            "TMP",
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
            "CUDA_CACHE_PATH",
            "WARP_CACHE_PATH",
        ),
    ).run(run_train, args, log_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UFO.")
    parser.add_argument("--agent", default=DEFAULT_AGENT, choices=["fb", "tldr"], help="Training agent preset: fb (default) or tldr.")
    parser.add_argument("--gpu-ids", default="single", help="'single', 'all', or a comma-separated GPU id list relative to CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--num-env-steps", type=int, default=DEFAULT_NUM_ENV_STEPS)
    parser.add_argument("--checkpoint-every-steps", type=int, default=DEFAULT_CHECKPOINT_EVERY_STEPS)
    parser.add_argument(
        "--data-path",
        nargs="+",
        default=None,
        help="One or more motion data pickle files. Multiple files require --data-mix-weights to fix source ratios.",
    )
    parser.add_argument(
        "--data-mix-weights",
        type=float,
        nargs="+",
        default=None,
        help="Source-level sampling weights for multiple --data-path entries, e.g. 0.95 0.05.",
    )
    parser.add_argument("--update-z-every-step", type=int, default=DEFAULT_UPDATE_Z_EVERY_STEP)
    parser.add_argument("--buffer-size", type=int, default=DEFAULT_BUFFER_SIZE, help="Replay capacity per rank/GPU.")
    parser.add_argument("--disable-dr", action="store_true", help="Disable domain randomization for training.")
    parser.add_argument("--disable-obs-noise", action="store_true", help="Disable observation noise for training.")
    parser.add_argument("--lr-scale", type=float, default=1.0, help="Scale FB learning rates. TLDR preset ignores this value.")
    parser.add_argument("--clip-grad-norm", type=float, default=0.0, help="Enable FB actor/FB gradient clipping when > 0.")
    parser.add_argument(
        "--cartwheel-aux-safe",
        action="store_true",
        help="Use a cartwheel-safe FB auxiliary reward set: remove locomotion contact/foot-shape penalties and reduce action-rate penalty.",
    )
    parser.add_argument("--seed", type=int, default=4728)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--disable-eval-prioritization",
        action="store_true",
        help="Validation/debug only: skip tracking eval and expert prioritization without changing default training behavior.",
    )
    parser.add_argument("--smoke", action="store_true", help="Short local smoke settings: 16 envs, 2048 env steps, no W&B.")
    args = parser.parse_args()
    if args.smoke:
        args.num_envs = min(args.num_envs, 16)
        args.num_env_steps = min(args.num_env_steps, 2048)
        args.use_wandb = False
    if args.data_path is not None:
        data_path_count = len(args.data_path)
        if args.data_mix_weights is not None:
            if len(args.data_mix_weights) != data_path_count:
                raise ValueError("--data-mix-weights length must match --data-path length")
            if any(w < 0 for w in args.data_mix_weights) or sum(args.data_mix_weights) <= 0:
                raise ValueError("--data-mix-weights must be non-negative and sum to a positive value")
            weight_sum = float(sum(args.data_mix_weights))
            args.data_mix_weights = [float(w) / weight_sum for w in args.data_mix_weights]
        elif data_path_count > 1:
            args.data_mix_weights = [1.0 / data_path_count] * data_path_count
        if data_path_count == 1:
            args.data_path = args.data_path[0]
            args.data_mix_weights = None

    if args.update_z_every_step <= 0:
        raise ValueError("--update-z-every-step must be positive")
    if args.buffer_size <= 0:
        raise ValueError("--buffer-size must be positive")
    if args.lr_scale <= 0:
        raise ValueError("--lr-scale must be positive")
    if args.clip_grad_norm < 0:
        raise ValueError("--clip-grad-norm must be non-negative")
    if args.cartwheel_aux_safe and args.agent != "fb":
        raise ValueError("--cartwheel-aux-safe is only supported with --agent fb")
    return args


def main() -> None:
    launch(parse_args())


if __name__ == "__main__":
    main()
