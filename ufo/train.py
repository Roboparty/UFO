"""UFO training entrypoint.

UFO provides FB and TeCH unsupervised RL presets for humanoid control.
Algorithm defaults live with their configs; command-line arguments can override them.
"""

from __future__ import annotations

import argparse
import os
from copy import deepcopy
from pathlib import Path


def _ensure_compile_cache() -> None:
    cache_dir = os.environ.get("UFO_CACHE_DIR")
    root = Path(cache_dir or Path.cwd() / "cache").expanduser()
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
DEFAULT_BUFFER_SIZE = 5120000
DEFAULT_WANDB_PROJECT = "ufo-humanoid"

from ufo.agents.fb_cpr_aux.config import CARTWHEEL_SAFE_REWARDS, build_fb_agent
from ufo.agents.fb_cpr_aux.config import DEFAULT_UPDATE_Z_EVERY_STEP as DEFAULT_FB_UPDATE_Z_EVERY_STEP
from ufo.agents.fb_cpr_aux.config import TRAIN_RUNTIME as FB_TRAIN_RUNTIME
from ufo.agents.tech.config import DEFAULT_UPDATE_Z_EVERY_STEP as DEFAULT_TECH_UPDATE_Z_EVERY_STEP
from ufo.agents.tech.config import TRAIN_RUNTIME as TECH_TRAIN_RUNTIME
from ufo.agents.tech.config import build_tech_agent
from ufo.envs.config import MJLabEnvConfig
from ufo.motion.manifest import prepare_motion_manifest
from ufo.tools.robot_config import (
    BUILTIN_ROBOTS,
    builtin_robot_config,
    default_robot_data,
    load_robot_training_spec,
    select_robot_config,
)
from ufo.training.config import DEFAULT_BUFFER_EVERY_STEPS, DEFAULT_CHECKPOINT_EVERY_STEPS, TrainConfig
from ufo.training.evaluation import TrackingEvaluationConfig


def canonical_agent_name(agent: str) -> str:
    if agent not in ("fb", "tech"):
        raise ValueError(f"Unsupported agent preset: {agent}. Supported presets: fb, tech")
    return agent


def _default_update_z_every_step(agent: str) -> int:
    canonical = canonical_agent_name(agent)
    return DEFAULT_TECH_UPDATE_Z_EVERY_STEP if canonical == "tech" else DEFAULT_FB_UPDATE_Z_EVERY_STEP


def build_mjlab_config(
    *,
    device: str,
    work_dir: str,
    num_envs: int,
    num_env_steps: int,
    seed: int,
    use_wandb: bool,
    wandb_run_name: str | None,
    checkpoint_every_steps: int = DEFAULT_CHECKPOINT_EVERY_STEPS,
    buffer_every_steps: int = DEFAULT_BUFFER_EVERY_STEPS,
    distributed_rank: int = 0,
    distributed_world_size: int = 1,
    disable_eval_prioritization: bool = False,
    smoke: bool = False,
    agent: str = DEFAULT_AGENT,
    data_path: str | list[str] | None = None,
    data_mix_weights: list[float] | None = None,
    update_z_every_step: int | None = None,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
    disable_dr: bool = False,
    disable_obs_noise: bool = False,
    lr_scale: float = 1.0,
    clip_grad_norm: float = 0.0,
    cartwheel_aux_safe: bool = False,
    num_agent_updates: int | None = None,
    robot_config: str | Path | None = None,
    network_width: int | None = None,
    shared_critic: bool | None = None,
) -> TrainConfig:
    agent = canonical_agent_name(agent)
    if (network_width is not None or shared_critic is not None) and agent != "fb":
        raise ValueError("Network width and shared critic are currently supported for FB")
    robot_training = load_robot_training_spec(select_robot_config(robot_config))
    evaluations = []
    run_eval_and_prioritization = not smoke and not disable_eval_prioritization
    if run_eval_and_prioritization:
        evaluations = [
            TrackingEvaluationConfig(
                name="TrackingEvaluationConfig",
                name_in_logs="ufo_tracking_eval",
                n_episodes_per_motion=1,
            )
        ]
    agent_device = "cuda" if device.startswith("cuda") else "cpu"
    resolved_update_z_every_step = _default_update_z_every_step(agent) if update_z_every_step is None else int(update_z_every_step)
    if agent == "fb":
        agent_cfg = build_fb_agent(
            device=agent_device,
            update_z_every_step=resolved_update_z_every_step,
            lr_scale=lr_scale,
            clip_grad_norm=clip_grad_norm,
            network_width=network_width,
            shared_critic=shared_critic,
            overrides=robot_training.fb_agent_overrides,
            cartwheel_aux_safe=cartwheel_aux_safe,
        )
        train_runtime = dict(FB_TRAIN_RUNTIME)
    else:
        agent_cfg = build_tech_agent(device=agent_device, update_z_every_step=resolved_update_z_every_step)
        if robot_training.task_overrides.get("rewards", {}).get("set_reward") == "roboparty":
            scales = robot_training.task_overrides["rewards"]["reward_scales"]
            scales = {key: value for key, value in scales.items() if value != 0.0}
            agent_cfg = agent_cfg.model_copy(update={"aux_rewards": list(scales), "aux_rewards_scaling": scales})
        train_runtime = dict(TECH_TRAIN_RUNTIME)
    if num_agent_updates is not None:
        if num_agent_updates <= 0:
            raise ValueError("num_agent_updates must be positive")
        train_runtime["num_agent_updates"] = int(num_agent_updates)
    task_overrides = deepcopy(robot_training.task_overrides)
    if cartwheel_aux_safe:
        if agent != "fb":
            raise ValueError("--cartwheel-aux-safe is only supported with --agent fb")
        task_overrides.setdefault("rewards", {}).setdefault("reward_scales", {}).update(CARTWHEEL_SAFE_REWARDS["task"])

    # Robot baselines tune scheduling; explicit CLI run lengths stay authoritative.
    train_runtime["prioritization_scale"] = 2.0
    train_runtime.update({
        key: value for key, value in robot_training.train_overrides.items()
        if key in {"eval_every_steps", "eval_at_start", "log_every_steps", "prioritization_scale"}
    })
    return TrainConfig(
        name="TrainConfig",
        agent=agent_cfg,
        env=MJLabEnvConfig(
            name="mjlab",
            device=device,
            motion_data_path=data_path or default_robot_data(robot_training.config_path),
            data_mix_weights=data_mix_weights,
            mjcf_path=robot_training.robot.xml_path,
            robot_config_path=str(robot_training.config_path),
            robot_training=robot_training.to_env_dict(),
            task_overrides=task_overrides,
            max_episode_length_s=None,
            disable_obs_noise=disable_obs_noise,
            disable_domain_randomization=disable_dr,
            include_last_action=True,
            include_history_actor=True,
            root_height_obs=True,
            auto_reset=False,
            seed=seed,
        ),
        work_dir=work_dir,
        seed=seed,
        online_parallel_envs=num_envs,
        num_env_steps=num_env_steps,
        checkpoint_every_steps=checkpoint_every_steps,
        buffer_every_steps=buffer_every_steps,
        prioritization=run_eval_and_prioritization,
        prioritization_min_val=0.5,
        prioritization_max_val=2.0,
        buffer_size=int(buffer_size),
        use_wandb=use_wandb,
        wandb_ename=os.environ.get("WANDB_ENTITY"),
        wandb_gname=f"ufo_{agent}",
        wandb_pname=os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT),
        wandb_run_name=wandb_run_name or f"ufo_{agent}",
        buffer_device=agent_device,
        disable_tqdm=True,
        evaluations=evaluations,
        distributed_rank=distributed_rank,
        distributed_world_size=distributed_world_size,
        tags={"backend": "mjlab", "agent": agent, "distributed_rank": distributed_rank, "distributed_world_size": distributed_world_size},
        **train_runtime,
    )


def _select_device_and_rank() -> tuple[str, int, int, int]:
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
    device, _local_rank, rank, world_size = _select_device_and_rank()
    _init_distributed(_local_rank, world_size)
    seed = args.seed + rank
    cfg = build_mjlab_config(
        device=device,
        work_dir=str(log_dir),
        num_envs=args.num_envs,
        num_env_steps=args.num_env_steps,
        seed=seed,
        use_wandb=bool(args.use_wandb and rank == 0),
        wandb_run_name=args.wandb_run_name,
        checkpoint_every_steps=args.checkpoint_every_steps,
        buffer_every_steps=args.buffer_every_steps,
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
        num_agent_updates=args.num_agent_updates,
        robot_config=args.robot_config,
        network_width=args.network_width,
        shared_critic=args.shared_critic,
    )
    print(
        "[INFO] UFO train: "
        f"agent={args.agent}, device={device}, rank={rank}/{world_size}, seed={seed}, work_dir={log_dir}, "
        f"robot_config={cfg.env.robot_config_path}, mjcf_path={cfg.env.mjcf_path}, "
        f"data_path={cfg.env.motion_data_path}, data_mix_weights={cfg.env.data_mix_weights}, "
        f"num_envs_per_rank={args.num_envs}, global_parallel_envs={args.num_envs * world_size}, "
        f"num_env_steps_global={args.num_env_steps}, buffer_size_per_rank={cfg.buffer_size}, "
        f"num_agent_updates={cfg.num_agent_updates}, update_agent_every_local={cfg.update_agent_every}, "
        f"cartwheel_aux_safe={args.cartwheel_aux_safe}, lr_scale={args.lr_scale}, clip_grad_norm={args.clip_grad_norm}, "
        f"disable_dr={cfg.env.disable_domain_randomization}, disable_obs_noise={cfg.env.disable_obs_noise}, "
        f"precision={'BF16 AMP (FP32 parameters)' if getattr(cfg.agent.train, 'use_bf16_amp', False) else 'FP32/TF32'}, "
        "compile=network_forwards, cuda_optimizer=fused_Adam, gradient_sync=manual_128MiB, "
        f"shared_critic={getattr(cfg.agent.model.archi, 'shared_critic', False)}, "
        f"actor_width={cfg.agent.model.archi.actor.hidden_dim}",
        flush=True,
    )
    try:
        trainer = cfg.build()
        trainer.train()
    finally:
        if world_size > 1:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()


def launch(args: argparse.Namespace) -> None:
    paths = [args.data_path] if isinstance(args.data_path, str) else args.data_path
    for path in paths:
        if not Path(path).expanduser().exists():
            raise FileNotFoundError(
                f"Missing motion data: {path}. For G1 run: bash scripts/download_data.sh g1_lafan. "
                "For RP1 run: uv run python -m ufo.tools.rp1_motion_import --help. "
                "Custom robots need --data-path or --data-manifest."
            )
    log_dir = Path(args.work_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
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

    if selected_gpus is None and not existing_visible:
        selected_gpus = [str(i) for i in range(num_gpus)]
    if selected_gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected_gpus)
    # Reuse the parsed namespace: manifests are already resolved into dataset paths.
    args.gpu_ids = "single"
    from torch.distributed.run import LaunchConfig, elastic_launch

    config = LaunchConfig(
        min_nodes=1, max_nodes=1, nproc_per_node=num_gpus,
        rdzv_backend="c10d", rdzv_endpoint="localhost:0",
        run_id=str(os.getpid()), start_method="spawn",
    )
    elastic_launch(config, run_train)(args, log_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UFO.")
    parser.add_argument(
        "--agent",
        default=DEFAULT_AGENT,
        choices=["fb", "tech"],
        help="Training agent preset: fb or tech.",
    )
    parser.add_argument(
        "--gpu-ids", default="single", help="'single', 'all', or a comma-separated GPU id list relative to CUDA_VISIBLE_DEVICES."
    )
    parser.add_argument("--work-dir", default=None, help="Run directory; defaults to runs/<robot>_<agent>.")
    robot_args = parser.add_mutually_exclusive_group()
    robot_args.add_argument("--robot", choices=tuple(BUILTIN_ROBOTS), help="Built-in robot: g1, rp1 or rpo (default: g1).")
    robot_args.add_argument(
        "--robot-config",
        type=Path,
        default=None,
        help=(
            "Robot YAML used for training metadata. Defaults to configs/robots/g1_29dof.yaml. "
            "If omitted and --data-manifest declares robot_config, the manifest robot config is used."
        ),
    )
    parser.add_argument(
        "--network-width", type=int, choices=[1024, 1536, 2048], default=None, help="FB actor, forward map and critic width."
    )
    parser.add_argument("--shared-critic", action=argparse.BooleanOptionalAction, default=None,
                        help="Override the robot baseline's shared/independent FB critics.")
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--num-env-steps", type=int, default=None)
    parser.add_argument(
        "--checkpoint-every-steps", type=int, default=None,
        help="Model checkpoint interval in global environment steps.",
    )
    parser.add_argument(
        "--buffer-every-steps", type=int, default=DEFAULT_BUFFER_EVERY_STEPS,
        help="Rank-0 replay snapshot interval in global environment steps.",
    )
    parser.add_argument(
        "--data-path",
        nargs="+",
        default=None,
        help="One or more motion data pickle files. Multiple files use equal source weights unless --data-mix-weights is provided.",
    )
    parser.add_argument(
        "--data-mix-weights",
        type=float,
        nargs="+",
        default=None,
        help="Source-level sampling weights for multiple --data-path entries, e.g. 0.95 0.05.",
    )
    parser.add_argument(
        "--data-manifest",
        type=Path,
        default=None,
        help="YAML manifest describing weighted motion data sources. Cannot be combined with --data-path.",
    )
    parser.add_argument(
        "--rebuild-motion-cache",
        action="store_true",
        help="Rebuild manifest-generated motion pkl caches instead of reusing existing cache files.",
    )
    parser.add_argument(
        "--update-z-every-step",
        type=int,
        default=None,
        help="Override latent update interval. Defaults to 100 for FB and 10 for TeCH.",
    )
    parser.add_argument("--buffer-size", type=int, default=DEFAULT_BUFFER_SIZE, help="Replay capacity per rank/GPU.")
    parser.add_argument(
        "--num-agent-updates",
        type=int,
        default=None,
        help=(
            "Override optimizer updates per update trigger. For fair env-scaling ablations, use 32 with "
            "2048 envs/GPU and 64 with 4096 envs/GPU to match the 1024 envs/GPU update density."
        ),
    )
    parser.add_argument("--disable-dr", action="store_true", help="Disable domain randomization for training.")
    parser.add_argument("--disable-obs-noise", action="store_true", help="Disable observation noise for training.")
    parser.add_argument("--lr-scale", type=float, default=1.0, help="Scale FB learning rates. TeCH preset ignores this value.")
    parser.add_argument("--clip-grad-norm", type=float, default=0.0, help="Enable FB actor/FB gradient clipping when > 0.")
    parser.add_argument(
        "--cartwheel-aux-safe",
        action="store_true",
        help="Use a cartwheel-safe FB auxiliary reward set: remove locomotion contact/foot-shape penalties and reduce action-rate penalty.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--disable-eval-prioritization",
        action="store_true",
        help="Validation/debug only: skip tracking eval and expert prioritization without changing default training behavior.",
    )
    parser.add_argument("--smoke", action="store_true", help="Short local smoke settings: 16 envs, 2048 env steps, no W&B.")
    args = parser.parse_args()
    args.agent = canonical_agent_name(args.agent)
    if args.update_z_every_step is None:
        args.update_z_every_step = _default_update_z_every_step(args.agent)
    if args.smoke:
        args.num_envs = min(args.num_envs, 16)
        args.use_wandb = False
    manifest_robot_config = None
    if args.data_manifest is not None:
        if args.data_path is not None:
            parser.error("--data-manifest and --data-path cannot be used together")
        manifest_data = prepare_motion_manifest(args.data_manifest, rebuild_cache=bool(args.rebuild_motion_cache))
        args.data_path = manifest_data.train_data_paths
        args.data_mix_weights = manifest_data.train_data_weights
        manifest_robot_config = manifest_data.robot_config_path
    elif args.data_path is not None:
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

    if args.robot is not None:
        args.robot_config = builtin_robot_config(args.robot)
    args.robot_config = select_robot_config(args.robot_config, manifest_robot_config)
    if args.data_path is None:
        args.data_path = default_robot_data(args.robot_config)
    if args.work_dir is None:
        robot_name = next((name for name in BUILTIN_ROBOTS if args.robot_config == builtin_robot_config(name)), args.robot_config.stem)
        args.work_dir = f"runs/{robot_name}_{args.agent}"
    baseline = load_robot_training_spec(args.robot_config).train_overrides
    allowed = {"seed", "num_env_steps", "checkpoint_every_steps", "eval_every_steps", "eval_at_start", "log_every_steps", "prioritization_scale"}
    if set(baseline) - allowed:
        raise ValueError(f"Unsupported training.train_overrides: {sorted(set(baseline) - allowed)}")
    for key, default in {"seed": 4728, "num_env_steps": DEFAULT_NUM_ENV_STEPS,
                         "checkpoint_every_steps": DEFAULT_CHECKPOINT_EVERY_STEPS}.items():
        if getattr(args, key) is None:
            setattr(args, key, baseline.get(key, default))
    if args.smoke:
        args.num_env_steps = min(args.num_env_steps, 2048)

    if args.checkpoint_every_steps <= 0:
        raise ValueError("--checkpoint-every-steps must be positive")
    if args.buffer_every_steps <= 0:
        raise ValueError("--buffer-every-steps must be positive")
    if args.update_z_every_step <= 0:
        raise ValueError("--update-z-every-step must be positive")
    if args.buffer_size <= 0:
        raise ValueError("--buffer-size must be positive")
    if args.num_agent_updates is not None and args.num_agent_updates <= 0:
        raise ValueError("--num-agent-updates must be positive")
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
