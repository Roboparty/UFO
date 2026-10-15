#!/usr/bin/env python3
"""No-actuation policy/task/model preflight for UFO onboard deployment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


sys.path.insert(0, str(ROOT))
from utils.policy import resolve_context_path, validate_policy_config, load_context, select_contexts, validate_context, validate_task_config

from utils.policy import load_yaml as _read_yaml, load_robot_config
from scripts.onboard.deploy_helpers import Reporter


class _DummyStateProcessor:
    def __init__(self, num_dof: int) -> None:
        self.num_dof = int(num_dof)
        self.root_ang_vel_b = np.zeros(3, dtype=np.float32)
        self.root_quat_b = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.joint_pos = np.zeros(self.num_dof, dtype=np.float32)
        self.joint_vel = np.zeros(self.num_dof, dtype=np.float32)


class _DummyEnv:
    def __init__(self, num_dof: int, num_actions: int) -> None:
        self.num_actions = int(num_actions)
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.state_processor = _DummyStateProcessor(num_dof)


def _compute_obs_dim(policy_config: dict[str, Any], reporter: Reporter) -> int | None:
    sys.path.insert(0, str(ROOT))
    try:
        from rl_policy.observations import ObsGroup
    except Exception as exc:
        reporter.fail(f"failed to import observations: {exc.__class__.__name__}: {exc}")
        return None

    joint_names = list(policy_config.get("isaac_joint_names", []))
    policy_joint_names = list(policy_config.get("policy_joint_names", []))
    env = _DummyEnv(num_dof=len(joint_names), num_actions=len(policy_joint_names))

    obs_cfg = policy_config.get("observation")
    if not isinstance(obs_cfg, dict) or "policy" not in obs_cfg:
        reporter.fail("policy_config['observation']['policy'] is missing")
        return None

    group_dims: dict[str, int] = {}
    for group_name, obs_items in obs_cfg.items():
        try:
            group = ObsGroup(group_name, obs_items, env)
            value = group.compute()
        except Exception as exc:
            reporter.fail(f"failed to compute observation group {group_name}: {exc.__class__.__name__}: {exc}")
            return None

        if value.ndim != 1 or not np.all(np.isfinite(value)):
            reporter.fail(f"observation group {group_name} produced invalid shape/value: {value.shape}")
            return None
        group_dims[group_name] = int(value.shape[0])
        reporter.info(f"obs_group {group_name}: dim={value.shape[0]}")

    policy_dim = group_dims.get("policy")
    if policy_dim is None:
        reporter.fail("policy observation group was not computed")
        return None
    reporter.ok(f"policy observation dim: {policy_dim}")
    return int(policy_dim)


def _load_context_dim(exp_config: dict[str, Any], model_path: Path, reporter: Reporter) -> int | None:
    task_type = str(exp_config.get("type", ""))
    if task_type not in {"tracking", "reward", "goal"}:
        reporter.fail(f"unsupported task type={task_type!r}")
        return None

    ctx_source = str(exp_config.get("ctx_source", "pkl")).lower()
    if task_type == "tracking" and ctx_source == "zmq":
        ctx_dim = 256
        reporter.ok("tracking context source is ZMQ")
        reporter.info(f"ctx_zmq_addr: {exp_config.get('ctx_zmq_addr', 'tcp://127.0.0.1:28711')}")
        reporter.info(f"ctx_zmq_timeout_ms: {exp_config.get('ctx_zmq_timeout_ms', 200)}")
        reporter.info(f"ctx_norm_ref: {exp_config.get('ctx_norm_ref', 16.0)}")
        return ctx_dim

    if ctx_source != "pkl":
        reporter.fail(f"unsupported ctx_source={ctx_source!r}")
        return None

    try:
        ctx_path = resolve_context_path(
            model_path, exp_config, default_ctx_dir=f"{task_type}_inference",
            fallback_ctx_dirs=(f"{task_type}_inference_mjlab",),
        )
        ctx = load_context(ctx_path)
        dimension = validate_context(ctx)
        if task_type != "tracking":
            select_contexts(ctx, exp_config)
    except Exception as exc:
        reporter.fail(f"failed to load {task_type} context: {exc.__class__.__name__}: {exc}")
        return None
    reporter.ok(f"{task_type} context loads: {ctx_path}")
    return int(dimension)


def _check_joint_config(robot_config: dict[str, Any], policy_config: dict[str, Any], reporter: Reporter) -> None:
    sys.path.insert(0, str(ROOT))
    try:
        from utils.strings import resolve_matching_names_values, unitree_joint_names
    except Exception as exc:
        reporter.fail(f"failed to import joint helpers: {exc.__class__.__name__}: {exc}")
        return

    isaac_joint_names = list(policy_config.get("isaac_joint_names", []))
    policy_joint_names = list(policy_config.get("policy_joint_names", []))
    if len(isaac_joint_names) != 29:
        reporter.fail(f"isaac_joint_names length is {len(isaac_joint_names)}, expected 29")
    else:
        reporter.ok("isaac_joint_names length is 29")
    if len(policy_joint_names) != 29:
        reporter.fail(f"policy_joint_names length is {len(policy_joint_names)}, expected 29")
    else:
        reporter.ok("policy_joint_names length is 29")

    missing = [name for name in isaac_joint_names if name not in unitree_joint_names]
    if missing:
        reporter.fail(f"isaac joints missing from Unitree order: {missing}")
    else:
        reporter.ok("all isaac joints map to Unitree order")

    for key in [
        "default_joint_pos",
        "action_scale",
        "joint_kp",
        "joint_kd",
    ]:
        try:
            resolve_matching_names_values(policy_config[key], isaac_joint_names, preserve_order=True, strict=False)
            reporter.ok(f"policy_config {key} resolves against isaac joints")
        except Exception as exc:
            reporter.fail(f"policy_config {key} failed to resolve: {exc.__class__.__name__}: {exc}")

    for key in ["joint_pos_lower_limit", "joint_pos_upper_limit", "joint_velocity_limit"]:
        try:
            _, _, values = resolve_matching_names_values(robot_config[key], isaac_joint_names, preserve_order=True, strict=False)
            values_arr = np.asarray(values, dtype=np.float64)
            if not np.all(np.isfinite(values_arr)):
                reporter.fail(f"robot_config {key} contains non-finite values")
            else:
                reporter.ok(f"robot_config {key} resolves against isaac joints")
        except Exception as exc:
            reporter.fail(f"robot_config {key} failed to resolve: {exc.__class__.__name__}: {exc}")


def _check_onnx(model_path: Path, obs_dim: int, ctx_dim: int, policy_config: dict[str, Any], reporter: Reporter) -> None:
    import onnxruntime as ort

    providers = policy_config.get("onnx_providers", ["CPUExecutionProvider"])
    if isinstance(providers, str):
        providers = [providers]
    try:
        sess = ort.InferenceSession(str(model_path), providers=providers)
    except Exception as exc:
        reporter.fail(f"failed to load policy ONNX: {exc.__class__.__name__}: {exc}")
        return

    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    if len(inputs) != 1:
        reporter.fail(f"policy ONNX input count is {len(inputs)}, expected 1")
        return
    if len(outputs) != 1:
        reporter.fail(f"policy ONNX output count is {len(outputs)}, expected 1")
        return

    in_shape = list(inputs[0].shape)
    out_shape = list(outputs[0].shape)
    expected_actor_dim = int(obs_dim) + int(ctx_dim)
    reporter.info(f"policy ONNX providers: {sess.get_providers()}")
    reporter.info(f"policy ONNX input: {(inputs[0].name, in_shape, inputs[0].type)}")
    reporter.info(f"policy ONNX output: {(outputs[0].name, out_shape, outputs[0].type)}")

    if len(in_shape) != 2 or in_shape[-1] != expected_actor_dim:
        reporter.fail(f"actor_obs dim mismatch: onnx={in_shape}, expected [1,{expected_actor_dim}]")
    else:
        reporter.ok(f"actor_obs dim matches obs+ctx: {obs_dim}+{ctx_dim}={expected_actor_dim}")

    num_actions = len(policy_config.get("policy_joint_names", []))
    if len(out_shape) != 2 or out_shape[-1] != num_actions:
        reporter.fail(f"action output dim mismatch: onnx={out_shape}, expected [1,{num_actions}]")
    else:
        reporter.ok(f"action output dim matches policy joints: {num_actions}")

    try:
        sample = np.zeros((1, expected_actor_dim), dtype=np.float32)
        action = sess.run([outputs[0].name], {inputs[0].name: sample})[0]
        if action.shape != (1, num_actions) or not np.all(np.isfinite(action)):
            reporter.fail(f"sample ONNX inference returned invalid action: shape={action.shape}")
        else:
            reporter.ok("sample ONNX inference returns finite action")
    except Exception as exc:
        reporter.fail(f"sample ONNX inference failed: {exc.__class__.__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="No-actuation UFO policy/task/model preflight")
    parser.add_argument("--robot-config", default="config/robot/g1_real.yaml")
    parser.add_argument("--policy-config", help="Defaults to the model's .deploy.yaml")
    parser.add_argument("--task", default="config/tasks/tracking.yaml")
    parser.add_argument("--model-path", default="model/g1_policy/exported/FBcprAuxModel.onnx")
    args = parser.parse_args()

    reporter = Reporter()
    reporter.info("NO-ACTUATION: does not instantiate UFODeployPolicy, G1Interface, StateProcessor, or CommandSender")

    robot_config_path = ROOT / args.robot_config
    task_path = ROOT / args.task
    model_path = ROOT / args.model_path
    policy_config_path = ROOT / args.policy_config if args.policy_config else model_path.with_suffix(".deploy.yaml")

    try:
        robot_config = load_robot_config(robot_config_path)
        policy_config = _read_yaml(policy_config_path)
        exp_config = _read_yaml(task_path)
        validate_task_config(exp_config, robot_config["ROBOT_TYPE"])
    except Exception as exc:
        reporter.fail(f"failed to load config: {exc.__class__.__name__}: {exc}")
        reporter.info(f"summary: {reporter.failures} failure(s), {reporter.warnings} warning(s)")
        return 1

    reporter.ok(f"robot config loads: {robot_config_path}")
    reporter.ok(f"policy config loads: {policy_config_path}")
    reporter.ok(f"task config loads: {task_path}")

    if not model_path.is_file():
        reporter.fail(f"policy model missing: {model_path}")
        reporter.info(f"summary: {reporter.failures} failure(s), {reporter.warnings} warning(s)")
        return 1
    reporter.ok(f"policy model exists: {model_path}")

    try:
        validate_policy_config(model_path, policy_config)
        reporter.ok("policy config matches exported training contract")
    except (ValueError, KeyError, TypeError) as exc:
        reporter.fail(str(exc))

    _check_joint_config(robot_config, policy_config, reporter)
    obs_dim = _compute_obs_dim(policy_config, reporter)
    ctx_dim = _load_context_dim(exp_config, model_path, reporter)

    if obs_dim is not None and ctx_dim is not None:
        _check_onnx(model_path, obs_dim, ctx_dim, policy_config, reporter)

    if reporter.failures:
        reporter.info(f"summary: {reporter.failures} failure(s), {reporter.warnings} warning(s)")
        return 1
    reporter.info(f"summary: all checks passed, {reporter.warnings} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
