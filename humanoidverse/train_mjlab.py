"""BFM-Zero training entrypoint for the MJLab backend.

This file keeps the original FBcprAux agent, model, replay and prioritization
settings from ``train_bfm_zero()`` and swaps only the environment config.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path


def _ensure_compile_cache(cache_root: str | Path | None = None) -> None:
    root = Path(os.environ.get("BFMZERO_MJLAB_CACHE_DIR", cache_root or Path.cwd() / "cache")).expanduser()
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

from humanoidverse.agents.envs.humanoidverse_mjlab import G1_MJLAB_MJCF_PATH, HumanoidVerseMjlabConfig
from humanoidverse.agents.evaluations.humanoidverse_mjlab import HumanoidVerseMjlabTrackingEvaluationConfig
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig, FBcprAuxAgentTrainConfig
from humanoidverse.agents.fb_cpr_aux.model import FBcprAuxModelArchiConfig, FBcprAuxModelConfig
from humanoidverse.agents.nn_filters import DictInputFilterConfig
from humanoidverse.agents.nn_models import (
    ActorArchiConfig,
    BackwardArchiConfig,
    DiscriminatorArchiConfig,
    ForwardArchiConfig,
    RewardNormalizerConfig,
)
from humanoidverse.agents.normalizers import BatchNormNormalizerConfig, ObsNormalizerConfig
from humanoidverse.train import TrainConfig


def build_bfm_zero_agent(device: str = "cuda", compile: bool = True) -> FBcprAuxAgentConfig:
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
            lr_f=0.0003,
            lr_b=1e-05,
            lr_actor=0.0003,
            weight_decay=0.0,
            clip_grad_norm=0.0,
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
            update_z_every_step=100,
            z_buffer_size=8192,
            rollout_expert_trajectories=True,
            rollout_expert_trajectories_length=250,
            rollout_expert_trajectories_percentage=0.5,
            lr_discriminator=1e-05,
            lr_critic=0.0003,
            critic_target_tau=0.005,
            critic_pessimism_penalty=0.5,
            reg_coeff=0.05,
            scale_reg=True,
            expert_asm_ratio=0.6,
            relabel_ratio=0.8,
            grad_penalty_discriminator=10.0,
            weight_decay_discriminator=0.0,
            lr_aux_critic=0.0003,
            reg_coeff_aux=0.02,
            aux_critic_pessimism_penalty=0.5,
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


def build_bfm_zero_mjlab_config(
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
    return TrainConfig(
        name="TrainConfig",
        agent=build_bfm_zero_agent(device="cuda" if device.startswith("cuda") else "cpu", compile=not distributed_sync),
        motions="",
        motions_root="",
        env=HumanoidVerseMjlabConfig(
            name="humanoidverse_mjlab",
            device=device,
            lafan_tail_path="humanoidverse/data/lafan_29dof_10s-clipped.pkl",
            mjcf_path=G1_MJLAB_MJCF_PATH,
            max_episode_length_s=None,
            disable_obs_noise=False,
            disable_domain_randomization=False,
            relative_config_path="exp/bfm_zero/bfm_zero",
            include_last_action=True,
            hydra_overrides=[
                "robot=g1/g1_29dof_hard_waist",
                "robot.control.action_scale=0.25",
                "robot.control.action_clip_value=5.0",
                "robot.control.normalize_action_to=5.0",
                "env.config.lie_down_init=True",
                "env.config.lie_down_init_prob=0.3",
            ],
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
        log_every_updates=384000,
        num_env_steps=num_env_steps,
        update_agent_every=1024,
        num_seed_steps=10240,
        num_agent_updates=16,
        checkpoint_every_steps=checkpoint_every_steps,
        checkpoint_buffer=True,
        prioritization=run_eval_and_prioritization,
        prioritization_min_val=0.5,
        prioritization_max_val=2.0,
        prioritization_scale=2.0,
        prioritization_mode="exp",
        use_trajectory_buffer=True,
        buffer_size=5120000,
        use_wandb=use_wandb,
        wandb_ename="xuewangusst-1",
        wandb_gname="g1_lafan_mjlab",
        wandb_pname="bfmzero-g1",
        wandb_run_name=wandb_run_name or "g1_lafan_mjlab1024env",
        load_expert_data_from_motion_lib=True,
        buffer_device="cuda" if device.startswith("cuda") else "cpu",
        disable_tqdm=True,
        evaluations=evaluations,
        eval_every_steps=9600000,
        distributed_rank=distributed_rank,
        distributed_world_size=distributed_world_size,
        rank0_only_writes=True,
        checkpoint_rank_buffers=True,
        distributed_sync=distributed_sync,
        distributed_global_steps=True,
        distributed_average_metrics=True,
        tags={"backend": "mjlab", "distributed_rank": distributed_rank, "distributed_world_size": distributed_world_size},
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
    cfg = build_bfm_zero_mjlab_config(
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
    )
    print(
        "[INFO] BFM-Zero MJLab train: "
        f"device={device}, rank={rank}/{world_size}, seed={seed}, work_dir={log_dir}, "
        f"mjcf_path={cfg.env.mjcf_path}, "
        f"num_envs_per_rank={args.num_envs}, global_parallel_envs={args.num_envs * world_size}, "
        f"num_env_steps_global={args.num_env_steps}, compile={cfg.agent.compile}",
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
    parser = argparse.ArgumentParser(description="Train BFM-Zero with the MJLab backend.")
    parser.add_argument("--gpu-ids", default="single", help="'single', 'all', or a comma-separated GPU id list relative to CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--work-dir", default="/data/xue/bfmzero-mjlab")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--num-env-steps", type=int, default=384000000)
    parser.add_argument("--checkpoint-every-steps", type=int, default=9600000)
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
    return args


def main() -> None:
    launch(parse_args())


if __name__ == "__main__":
    main()
