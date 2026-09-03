"""UFO training entrypoint.

UFO provides FB and TeCH unsupervised RL presets for humanoid control.
Defaults are kept in this file; command-line arguments can override them.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf


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
# A zero-sized dedicated collector selects the original UFO topology: the
# behavior prior consumes main-terrain replay. Positive values remain available
# as an explicit canonical-plane ablation.
DEFAULT_PRIOR_PLANE_ENVS = 0
DEFAULT_NUM_ENV_STEPS = 192000000
DEFAULT_CHECKPOINT_EVERY_STEPS = 9600000
DEFAULT_TRACKING_EVAL_EVERY_STEPS = 3200000
DEFAULT_SAME_Z_EVAL_EVERY_STEPS = 0
DEFAULT_DATA_PATH = "humanoidverse/data/lafan_29dof_10s-clipped.pkl"
DEFAULT_WORK_DIR = "runs/ufo"
DEFAULT_BUFFER_SIZE = 5120000
DEFAULT_FB_UPDATE_Z_EVERY_STEP = 100
DEFAULT_TECH_UPDATE_Z_EVERY_STEP = 10
DEFAULT_UPDATE_Z_EVERY_STEP = DEFAULT_FB_UPDATE_Z_EVERY_STEP
DEFAULT_WANDB_PROJECT = "ufo-humanoid"
DEFAULT_ROBOT_CONFIG = "configs/robots/g1_29dof.yaml"

AGENT_ALIASES = {
    "fb": "fb",
    "fb_depth": "fb_depth",
    "fb_terrain": "fb_terrain",
    "tech": "tech",
    "tldr": "tech",
}

from humanoidverse.agents.envs.humanoidverse_mjlab import HumanoidVerseMjlabConfig
from humanoidverse.agents.evaluations.humanoidverse_mjlab import HumanoidVerseMjlabTrackingEvaluationConfig
from humanoidverse.agents.evaluations.same_z_terrain import SameZTerrainEvaluationConfig
from humanoidverse.agents.presets import build_agent_preset
from humanoidverse.terrains.coverage import validate_motion_terrain_coverage
from humanoidverse.terrains.rp1_simple import RP1_PATCH_SIZE, RP1_TERRAIN_BORDER_WIDTH
from humanoidverse.training.workspace import TrainConfig
from humanoidverse.utils.motion_data import prepare_motion_manifest
from humanoidverse.utils.robot_spec import assert_robot_configs_compatible, load_robot_training_spec, resolve_robot_config_path


def _resolve_training_robot_config(
    cli_robot_config: str | Path | None,
    manifest_robot_config: str | Path | None,
) -> Path:
    if cli_robot_config is not None and manifest_robot_config is not None:
        return assert_robot_configs_compatible(cli_robot_config, manifest_robot_config)
    if cli_robot_config is not None:
        return resolve_robot_config_path(cli_robot_config)
    if manifest_robot_config is not None:
        return resolve_robot_config_path(manifest_robot_config)
    return resolve_robot_config_path(DEFAULT_ROBOT_CONFIG)


def canonical_agent_name(agent: str) -> str:
    try:
        return AGENT_ALIASES[agent]
    except KeyError as exc:
        supported = ", ".join(sorted(AGENT_ALIASES))
        raise ValueError(f"Unsupported agent preset: {agent}. Supported presets: {supported}") from exc


def _default_update_z_every_step(agent: str) -> int:
    canonical = canonical_agent_name(agent)
    return DEFAULT_TECH_UPDATE_Z_EVERY_STEP if canonical == "tech" else DEFAULT_FB_UPDATE_Z_EVERY_STEP


def build_ufo_mjlab_config(
    *,
    device: str,
    work_dir: str,
    num_envs: int,
    num_env_steps: int,
    seed: int,
    use_wandb: bool,
    wandb_run_name: str | None,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    checkpoint_every_steps: int = DEFAULT_CHECKPOINT_EVERY_STEPS,
    tracking_eval_every_steps: int = DEFAULT_TRACKING_EVAL_EVERY_STEPS,
    same_z_eval_every_steps: int = DEFAULT_SAME_Z_EVAL_EVERY_STEPS,
    distributed_rank: int = 0,
    distributed_world_size: int = 1,
    disable_eval_prioritization: bool = False,
    smoke: bool = False,
    agent: str = DEFAULT_AGENT,
    data_path: str | list[str] | None = None,
    data_mix_weights: list[float] | None = None,
    update_z_every_step: int | None = None,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
    prior_plane_envs: int | None = None,
    disable_dr: bool = False,
    disable_obs_noise: bool = False,
    lr_scale: float = 1.0,
    clip_grad_norm: float = 0.0,
    resume_replay_warmup_steps: int = 0,
    cartwheel_aux_safe: bool = False,
    heading_context: bool = True,
    heading_reg_coeff: float = 0.0,
    behavior_prior: bool = True,
    selective_prior: bool = False,
    num_agent_updates: int | None = None,
    robot_config: str | Path | None = None,
    terrain_mode: str | None = None,
    cache_expert_buffer: bool = True,
    rebuild_expert_buffer_cache: bool = False,
    gradient_sync: str = "auto",
    ddp_bucket_cap_mb: float = 25.0,
) -> TrainConfig:
    agent = canonical_agent_name(agent)
    if heading_reg_coeff < 0.0:
        raise ValueError("heading_reg_coeff must be non-negative")
    if heading_reg_coeff > 0.0 and not heading_context:
        raise ValueError("heading_reg_coeff requires heading_context=True")
    if heading_reg_coeff > 0.0 and agent != "fb_depth":
        raise ValueError("BehaviorContext heading is only supported by fb_depth")
    if not behavior_prior and prior_plane_envs not in (None, 0):
        raise ValueError("behavior_prior=False requires prior_plane_envs=0")
    if selective_prior and not behavior_prior:
        raise ValueError("selective_prior=True requires behavior_prior=True")
    if selective_prior and agent != "fb_depth":
        raise ValueError("selective online prior is currently supported only by fb_depth")
    if selective_prior and not heading_context:
        raise ValueError("selective online prior requires BehaviorContext metadata")
    if selective_prior and prior_plane_envs not in (None, 0):
        raise ValueError("selective online prior cannot use canonical-plane collectors")
    if selective_prior:
        prior_plane_envs = 0
    if not behavior_prior:
        prior_plane_envs = 0
    elif prior_plane_envs is None:
        prior_plane_envs = DEFAULT_PRIOR_PLANE_ENVS if agent == "fb_depth" else 0
    prior_plane_envs = int(prior_plane_envs)
    if prior_plane_envs < 0:
        raise ValueError("prior_plane_envs must be non-negative")
    if prior_plane_envs > 0 and agent != "fb_depth":
        raise ValueError("canonical-plane prior collection is supported only by fb_depth")
    for cadence_name, cadence_steps in (
        ("checkpoint_every_steps", checkpoint_every_steps),
        ("tracking_eval_every_steps", tracking_eval_every_steps),
    ):
        if int(cadence_steps) <= 0:
            raise ValueError(f"{cadence_name} must be positive")
    if smoke and prior_plane_envs > 0:
        prior_plane_envs = min(prior_plane_envs, max(2, num_envs // 8))
    if gradient_sync == "auto":
        gradient_sync = "ddp" if distributed_world_size > 1 and agent in {"fb", "fb_terrain", "fb_depth"} else "manual"
    if gradient_sync not in {"manual", "ddp"}:
        raise ValueError(f"Unsupported gradient synchronization mode: {gradient_sync!r}")
    if gradient_sync == "ddp" and distributed_world_size <= 1:
        raise ValueError("gradient_sync='ddp' requires more than one distributed rank")
    robot_training = load_robot_training_spec(robot_config or DEFAULT_ROBOT_CONFIG)
    try:
        raw_robot_config = OmegaConf.to_container(OmegaConf.load(robot_training.config_path), resolve=True)
        metadata = raw_robot_config.get("metadata") if isinstance(raw_robot_config, dict) else None
        if isinstance(metadata, dict) and metadata.get("review_status") == "draft":
            print(
                "WARNING: Robot config is auto-generated draft. Review semantics, default pose, PD gains, "
                "actuator parameters, contact bodies, and reward/termination-related fields before formal training.",
                flush=True,
            )
    except Exception as exc:
        print(f"WARNING: Could not inspect robot config metadata for draft status: {exc}", flush=True)
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
                every_steps=int(tracking_eval_every_steps),
                env=None,
                num_envs=num_envs,
                n_episodes_per_motion=1,
            )
        ]
        if agent == "fb_depth" and int(same_z_eval_every_steps) > 0:
            evaluations.append(
                SameZTerrainEvaluationConfig(
                    name="SameZTerrainEvaluationConfig",
                    generate_videos=False,
                    name_in_logs="same_z_terrain_eval",
                    every_steps=int(same_z_eval_every_steps),
                    seed=seed,
                )
            )
    agent_device = "cuda" if device.startswith("cuda") else "cpu"
    resolved_update_z_every_step = _default_update_z_every_step(agent) if update_z_every_step is None else int(update_z_every_step)
    selected = build_agent_preset(
        agent=agent,
        device=agent_device,
        compile=not distributed_sync,
        update_z_every_step=resolved_update_z_every_step,
        lr_scale=lr_scale,
        clip_grad_norm=clip_grad_norm,
        cartwheel_aux_safe=cartwheel_aux_safe,
        heading_context=bool(heading_context and agent == "fb_depth"),
        heading_reg_coeff=float(heading_reg_coeff),
        behavior_prior=bool(behavior_prior),
        selective_prior=bool(selective_prior),
        wandb_project=wandb_project,
    )
    agent_cfg = selected["agent_cfg"]
    wandb_group = selected["wandb_group"]
    wandb_project = selected["wandb_project"]
    train_runtime = dict(selected["train_runtime"])
    if smoke:
        train_runtime.update(
            {
                "log_every_updates": 1024,
                "update_agent_every": 1024,
                "num_seed_steps": 1024,
                "num_agent_updates": 1,
                "checkpoint_buffer": False,
            }
        )
    if num_agent_updates is not None:
        if num_agent_updates <= 0:
            raise ValueError("num_agent_updates must be positive")
        train_runtime["num_agent_updates"] = int(num_agent_updates)
    hydra_overrides = [
        f"robot={robot_training.hydra_robot}",
        f"robot.control.action_scale={robot_training.action_scale}",
        f"robot.control.action_clip_value={robot_training.action_clip_value}",
        f"robot.control.normalize_action_to={robot_training.normalize_action_to}",
        *robot_training.hydra_overrides,
    ]
    if agent in {"fb_terrain", "fb_depth"}:
        terrain_mode = terrain_mode or ("rp1_simple" if agent == "fb_depth" else "mixed")
        hydra_overrides.extend(
            [
                "terrain=terrain_ufo_v0",
                f"terrain.terrain_type={terrain_mode}",
                f"terrain.seed={seed}",
                "rewards.terrain_aware_auxiliary=true",
                "rewards.reward_scales.penalty_undesired_contact=0.0",
                "rewards.reward_scales.penalty_feet_ori=0.0",
            ]
        )
        if agent == "fb_depth":
            hydra_overrides.append("terrain.direct_depth.enabled=true")
    elif terrain_mode is not None:
        raise ValueError("--terrain-mode is only valid with --agent fb_terrain or fb_depth")
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

    main_buffer_size = min(int(buffer_size), 4096) if smoke else int(buffer_size)
    replay_time_slots = max(main_buffer_size // max(int(num_envs), 1), 2)
    prior_buffer_size = prior_plane_envs * replay_time_slots

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
            mjcf_path=robot_training.robot.xml_path,
            robot_config_path=str(robot_training.config_path),
            robot_training=robot_training.to_env_dict(),
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
        prior_plane_envs=prior_plane_envs,
        log_every_updates=train_runtime["log_every_updates"],
        num_env_steps=num_env_steps,
        update_agent_every=train_runtime["update_agent_every"],
        num_seed_steps=train_runtime["num_seed_steps"],
        resume_replay_warmup_steps=int(resume_replay_warmup_steps),
        num_agent_updates=train_runtime["num_agent_updates"],
        checkpoint_every_steps=checkpoint_every_steps,
        checkpoint_buffer=train_runtime["checkpoint_buffer"],
        prioritization=run_eval_and_prioritization,
        prioritization_min_val=0.5,
        prioritization_max_val=2.0,
        prioritization_scale=2.0,
        prioritization_mode="exp",
        use_trajectory_buffer=train_runtime["use_trajectory_buffer"],
        buffer_size=main_buffer_size,
        prior_buffer_size=prior_buffer_size,
        use_wandb=use_wandb,
        wandb_ename=os.environ.get("WANDB_ENTITY"),
        wandb_gname=wandb_group,
        wandb_pname=wandb_project,
        wandb_run_name=wandb_run_name or f"ufo_{agent}",
        load_expert_data_from_motion_lib=True,
        buffer_device="cuda" if device.startswith("cuda") else "cpu",
        cache_expert_buffer=bool(cache_expert_buffer),
        rebuild_expert_buffer_cache=bool(rebuild_expert_buffer_cache),
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
        distributed_gradient_sync=gradient_sync,
        ddp_bucket_cap_mb=float(ddp_bucket_cap_mb),
        tags={
            "backend": "mjlab",
            "agent": agent,
            "distributed_rank": distributed_rank,
            "distributed_world_size": distributed_world_size,
            "direct_depth_replay_storage_dtype": "compact_uint8" if agent == "fb_depth" else None,
        },
    )


def _select_device_and_rank(seed: int) -> tuple[str, int, int, int]:
    if "LOCAL_RANK" in os.environ or int(os.environ.get("WORLD_SIZE", "1")) > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
        return f"cuda:{local_rank}", local_rank, rank, world_size

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
    local_rank = 0
    rank = 0
    world_size = 1
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
        wandb_project=args.wandb_project,
        checkpoint_every_steps=args.checkpoint_every_steps,
        tracking_eval_every_steps=args.tracking_eval_every_steps,
        same_z_eval_every_steps=args.same_z_eval_every_steps,
        distributed_rank=rank,
        distributed_world_size=world_size,
        disable_eval_prioritization=bool(args.disable_eval_prioritization),
        smoke=bool(args.smoke),
        agent=args.agent,
        data_path=args.data_path,
        data_mix_weights=args.data_mix_weights,
        update_z_every_step=args.update_z_every_step,
        buffer_size=args.buffer_size,
        prior_plane_envs=args.prior_plane_envs,
        disable_dr=bool(args.disable_dr),
        disable_obs_noise=bool(args.disable_obs_noise),
        lr_scale=args.lr_scale,
        clip_grad_norm=args.clip_grad_norm,
        resume_replay_warmup_steps=args.resume_replay_warmup_steps,
        cartwheel_aux_safe=bool(args.cartwheel_aux_safe),
        heading_context=bool(args.heading_context),
        heading_reg_coeff=float(args.heading_reg_coeff),
        behavior_prior=bool(args.behavior_prior),
        selective_prior=bool(args.selective_prior),
        num_agent_updates=args.num_agent_updates,
        robot_config=args.robot_config,
        terrain_mode=args.terrain_mode,
        cache_expert_buffer=bool(args.expert_buffer_cache),
        rebuild_expert_buffer_cache=bool(args.rebuild_expert_buffer_cache),
        gradient_sync=args.gradient_sync,
        ddp_bucket_cap_mb=args.ddp_bucket_cap_mb,
    )
    behavior_prior_enabled = bool(getattr(cfg.agent.train, "behavior_prior_enabled", True))
    behavior_prior_source = (
        "disabled"
        if not behavior_prior_enabled
        else (
            "selective_main"
            if bool(getattr(cfg.agent.train, "selective_prior_enabled", False))
            else ("plane" if cfg.prior_plane_envs > 0 else "main")
        )
    )
    print(
        "[INFO] UFO train: "
        f"agent={args.agent}, device={device}, rank={rank}/{world_size}, seed={seed}, work_dir={log_dir}, "
        f"robot_config={cfg.env.robot_config_path}, mjcf_path={cfg.env.mjcf_path}, "
        f"data_path={cfg.env.lafan_tail_path}, data_mix_weights={cfg.env.data_mix_weights}, "
        f"num_envs_per_rank={args.num_envs}, global_parallel_envs={args.num_envs * world_size}, "
        f"prior_plane_envs_per_rank={cfg.prior_plane_envs}, prior_buffer_size_per_rank={cfg.prior_buffer_size}, "
        f"num_env_steps_global={args.num_env_steps}, buffer_size_per_rank={cfg.buffer_size}, "
        f"num_agent_updates={cfg.num_agent_updates}, update_agent_every_local={cfg.update_agent_every}, "
        f"cartwheel_aux_safe={args.cartwheel_aux_safe}, lr_scale={args.lr_scale}, clip_grad_norm={args.clip_grad_norm}, "
        f"heading_context={cfg.agent.model.heading_context_enabled}, "
        f"heading_reg_coeff={getattr(cfg.agent.train, 'reg_coeff_heading', 0.0):g}, "
        f"behavior_prior_enabled={behavior_prior_enabled}, "
        f"selective_prior_enabled={getattr(cfg.agent.train, 'selective_prior_enabled', False)}, "
        f"behavior_prior_source={behavior_prior_source}, "
        f"discriminator_loss={getattr(cfg.agent.train, 'discriminator_loss', 'bce')}, "
        f"discriminator_reward={getattr(cfg.agent.train, 'discriminator_reward', 'log_odds')}, "
        f"resume_replay_warmup_steps={cfg.resume_replay_warmup_steps}, "
        f"disable_dr={cfg.env.disable_domain_randomization}, disable_obs_noise={cfg.env.disable_obs_noise}, "
        f"terrain_mode={args.terrain_mode}, compile={cfg.agent.compile}, "
        f"expert_buffer_cache={cfg.cache_expert_buffer}, gradient_sync={cfg.distributed_gradient_sync}, "
        f"ddp_bucket_cap_mb={cfg.ddp_bucket_cap_mb:g}",
        flush=True,
    )
    training_completed = False
    try:
        workspace = cfg.build()
        workspace.train()
        training_completed = True
    finally:
        if world_size > 1:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                if training_completed:
                    dist.destroy_process_group()
                elif rank == 0:
                    # A rank-local exception can leave peers inside a collective.
                    # Calling destroy_process_group() from only the failed rank
                    # then hides the original traceback behind a second deadlock.
                    print(
                        "[WARNING] Skipping coordinated process-group teardown after a training failure; "
                        "the distributed launcher will terminate peer ranks.",
                        file=sys.stderr,
                        flush=True,
                    )


def launch(args: argparse.Namespace) -> None:
    log_dir = Path(args.work_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    _ensure_compile_cache()
    if args.agent in {"fb_terrain", "fb_depth"} and args.terrain_mode != "plane":
        terrain_cfg = OmegaConf.load(Path(__file__).parent / "config/terrain/terrain_ufo_v0.yaml").terrain
        patch_size = (
            (RP1_PATCH_SIZE, RP1_PATCH_SIZE)
            if args.terrain_mode == "rp1_simple"
            else tuple(float(value) for value in terrain_cfg.patch_size)
        )
        border_width = RP1_TERRAIN_BORDER_WIDTH if args.terrain_mode == "rp1_simple" else float(terrain_cfg.border_width)
        sensor_radius = math.hypot(
            max(abs(float(terrain_cfg.terrain_priv.x_min)), abs(float(terrain_cfg.terrain_priv.x_max))),
            max(abs(float(terrain_cfg.terrain_priv.y_min)), abs(float(terrain_cfg.terrain_priv.y_max))),
        )
        report = validate_motion_terrain_coverage(
            args.data_path or DEFAULT_DATA_PATH,
            patch_size=patch_size,
            sensor_radius=sensor_radius,
            policy_margin=float(terrain_cfg.coverage.policy_margin),
            # Tiles are traversable. The worst assigned origin is at the
            # center of an outer tile, with the global border beyond it.
            safe_radius=(min(patch_size) / 2.0 + border_width),
        )
        print(
            "[INFO] terrain coverage preflight passed: "
            f"max_excursion={report.max_excursion:.3f}m motion={report.motion_key!r}, "
            f"sensor_radius={report.sensor_radius:.3f}m, policy_margin={report.policy_margin:.3f}m, "
            f"required={report.required_radius:.3f}m < connected_safe_radius={report.patch_safe_radius:.3f}m",
            flush=True,
        )
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
            "UFO_DATA_DIR",
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
    parser.add_argument(
        "--agent",
        default=DEFAULT_AGENT,
        choices=["fb", "fb_terrain", "fb_depth", "tech", "tldr"],
        help="Training agent preset. fb_depth uses the RP1-compatible uint8 temporal-depth branch.",
    )
    parser.add_argument(
        "--terrain-mode",
        choices=["flat", "slope", "stairs", "stairs_up", "stairs_down", "rough", "platforms", "mixed", "rp1_simple"],
        default=None,
        help="Physical terrain (defaults: fb_terrain=mixed, fb_depth=rp1_simple).",
    )
    parser.add_argument(
        "--gpu-ids", default="single", help="'single', 'all', or a comma-separated GPU id list relative to CUDA_VISIBLE_DEVICES."
    )
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=None,
        help=(
            "Robot YAML used for training metadata. Defaults to configs/robots/g1_29dof.yaml. "
            "If omitted and --data-manifest declares robot_config, the manifest robot config is used."
        ),
    )
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--num-env-steps", type=int, default=DEFAULT_NUM_ENV_STEPS)
    parser.add_argument("--checkpoint-every-steps", type=int, default=DEFAULT_CHECKPOINT_EVERY_STEPS)
    parser.add_argument(
        "--tracking-eval-every-steps",
        type=int,
        default=DEFAULT_TRACKING_EVAL_EVERY_STEPS,
        help="Flat tracking/EMD evaluation cadence in global environment steps.",
    )
    parser.add_argument(
        "--same-z-eval-every-steps",
        type=int,
        default=DEFAULT_SAME_Z_EVAL_EVERY_STEPS,
        help="Same-z terrain evaluation cadence in global environment steps. Set <=0 to disable.",
    )
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
        "--prior-plane-envs",
        type=int,
        default=None,
        help=(
            "Optional canonical-plane policy environments per rank for the fb_depth D/Q_D stream. "
            "The default 0 uses original-UFO main-terrain replay; positive values select the "
            "canonical-plane ablation."
        ),
    )
    parser.add_argument(
        "--behavior-prior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable the discriminator/Q_D behavior-prior branch. "
            "With --prior-plane-envs 0 it uses main-terrain replay; use "
            "--no-behavior-prior for the formal no-D ablation."
        ),
    )
    parser.add_argument(
        "--selective-prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use expert-anchored selective online prior expansion. Policy samples are admitted "
            "by a slow D-independent verifier; UNKNOWN samples are excluded from D/Q_D/Actor-D."
        ),
    )
    parser.add_argument(
        "--expert-buffer-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse a content-addressed expert trajectory and full-FK cache.",
    )
    parser.add_argument(
        "--rebuild-expert-buffer-cache",
        action="store_true",
        help="Force rank 0 to atomically rebuild the expert/full-FK cache.",
    )
    parser.add_argument(
        "--gradient-sync",
        choices=["auto", "ddp", "manual"],
        default="auto",
        help="Distributed gradient synchronization; auto selects DDP overlap for FB agents.",
    )
    parser.add_argument(
        "--ddp-bucket-cap-mb",
        type=float,
        default=25.0,
        help="DDP gradient bucket capacity in MiB.",
    )
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
        "--resume-replay-warmup-steps",
        type=int,
        default=0,
        help=(
            "Recovery-only local env steps to collect with a loaded policy before optimizer updates. "
            "Use only when a resumed checkpoint intentionally has no replay buffer."
        ),
    )
    parser.add_argument(
        "--cartwheel-aux-safe",
        action="store_true",
        help="Use a cartwheel-safe FB auxiliary reward set: remove locomotion contact/foot-shape penalties and reduce action-rate penalty.",
    )
    parser.add_argument(
        "--heading-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable z-companion BehaviorContext heading observations for fb_depth. Use --no-heading-context for the legacy architecture."
        ),
    )
    parser.add_argument(
        "--heading-reg-coeff",
        type=float,
        default=0.0,
        help=("Independent Q_H Actor coefficient. Zero runs the observation-only ablation and completely skips Q_H optimization."),
    )
    parser.add_argument("--seed", type=int, default=4728)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--disable-eval-prioritization",
        action="store_true",
        help="Validation/debug only: skip tracking eval and expert prioritization without changing default training behavior.",
    )
    parser.add_argument("--smoke", action="store_true", help="Short local smoke settings: 16 envs, 2048 env steps, no W&B.")
    args = parser.parse_args()
    raw_agent = args.agent
    args.agent = canonical_agent_name(args.agent)
    if raw_agent == "tldr":
        print("WARNING: agent=tldr is deprecated; use agent=tech instead.", file=sys.stderr, flush=True)
    if args.update_z_every_step is None:
        args.update_z_every_step = _default_update_z_every_step(args.agent)
    if args.smoke:
        args.num_envs = min(args.num_envs, 16)
        args.num_env_steps = min(args.num_env_steps, 2048)
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

    args.robot_config = _resolve_training_robot_config(args.robot_config, manifest_robot_config)

    if args.update_z_every_step <= 0:
        raise ValueError("--update-z-every-step must be positive")
    if args.buffer_size <= 0:
        raise ValueError("--buffer-size must be positive")
    if args.prior_plane_envs is not None and args.prior_plane_envs < 0:
        raise ValueError("--prior-plane-envs must be non-negative")
    if args.prior_plane_envs not in (None, 0) and args.agent != "fb_depth":
        raise ValueError("--prior-plane-envs is only supported with --agent fb_depth")
    if not args.behavior_prior and args.prior_plane_envs not in (None, 0):
        raise ValueError("--no-behavior-prior requires --prior-plane-envs 0")
    if not args.behavior_prior and args.agent != "fb_depth":
        raise ValueError("--no-behavior-prior is currently defined only for --agent fb_depth")
    if args.selective_prior and not args.behavior_prior:
        raise ValueError("--selective-prior requires --behavior-prior")
    if args.selective_prior and args.agent != "fb_depth":
        raise ValueError("--selective-prior is currently supported only with --agent fb_depth")
    if args.selective_prior and args.prior_plane_envs not in (None, 0):
        raise ValueError("--selective-prior cannot be combined with --prior-plane-envs")
    if args.selective_prior and not args.heading_context:
        raise ValueError("--selective-prior requires --heading-context")
    if args.num_agent_updates is not None and args.num_agent_updates <= 0:
        raise ValueError("--num-agent-updates must be positive")
    if args.lr_scale <= 0:
        raise ValueError("--lr-scale must be positive")
    if args.clip_grad_norm < 0:
        raise ValueError("--clip-grad-norm must be non-negative")
    if args.resume_replay_warmup_steps < 0:
        raise ValueError("--resume-replay-warmup-steps must be non-negative")
    if args.cartwheel_aux_safe and args.agent not in {"fb", "fb_terrain", "fb_depth"}:
        raise ValueError("--cartwheel-aux-safe is only supported with --agent fb, fb_terrain, or fb_depth")
    if args.heading_reg_coeff < 0.0:
        raise ValueError("--heading-reg-coeff must be non-negative")
    if args.heading_reg_coeff > 0.0 and not args.heading_context:
        raise ValueError("--heading-reg-coeff requires --heading-context")
    if (args.heading_context or args.heading_reg_coeff > 0.0) and args.agent != "fb_depth":
        # The parser default is convenient for fb_depth; silently disable it
        # for legacy agent presets unless a positive Q_H coefficient was asked.
        if args.heading_reg_coeff > 0.0:
            raise ValueError("BehaviorContext heading is only supported with --agent fb_depth")
        args.heading_context = False
    if args.agent in {"fb_terrain", "fb_depth"} and args.terrain_mode is None:
        args.terrain_mode = "rp1_simple" if args.agent == "fb_depth" else "mixed"
    if args.agent not in {"fb_terrain", "fb_depth"} and args.terrain_mode is not None:
        raise ValueError("--terrain-mode is only valid with --agent fb_terrain or fb_depth")
    return args


def main() -> None:
    launch(parse_args())


if __name__ == "__main__":
    main()
