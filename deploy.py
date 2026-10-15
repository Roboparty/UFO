#!/usr/bin/env python3
"""Run a G1 policy in MuJoCo or onboard, with one task selector."""

import argparse
import os
from pathlib import Path

from utils.policy import load_robot_config, load_yaml, validate_task_config

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("sim2sim", "sim2real"))
    parser.add_argument("--task", required=True, choices=("tracking", "goal", "reward", "teleop"))
    parser.add_argument("--model", type=Path, default=ROOT / "model/g1_policy/exported/FBcprAuxModel.onnx")
    parser.add_argument("--policy-config", type=Path, help="Defaults to the model's .deploy.yaml")
    parser.add_argument("--task-config", type=Path, help="Override config/tasks/<task>.yaml")
    parser.add_argument("--robot-config", type=Path, help="Override the G1 config for the selected mode")
    parser.add_argument("--interface", default=os.environ.get("G1_INTERFACE"), help="Onboard DDS network interface")
    parser.add_argument("--allow-default-route-interface", action="store_true")
    parser.add_argument("--pico-control", action="store_true", help="Debug: also let PICO buttons control policy state")
    parser.add_argument("--pico-control-addr", default="tcp://127.0.0.1:28704")
    args = parser.parse_args(argv)

    if args.mode == "sim2real" and os.environ.get("UFO_REAL_ROBOT_OK") != "1":
        parser.error("Real robot control requires UFO_REAL_ROBOT_OK=1")
    if args.mode == "sim2sim" and (args.interface or args.allow_default_route_interface):
        parser.error("--interface / G1_INTERFACE applies only to sim2real")
    model_path = args.model.expanduser().resolve()
    robot_name = "g1_real" if args.mode == "sim2real" else "g1"
    robot_config = load_robot_config(args.robot_config or ROOT / f"config/robot/{robot_name}.yaml")
    expected_robot = "g1_real" if args.mode == "sim2real" else "g1_29dof"
    if robot_config.get("ROBOT_TYPE") != expected_robot:
        parser.error(f"{args.mode} requires ROBOT_TYPE={expected_robot}")
    task_config = load_yaml(args.task_config or ROOT / f"config/tasks/{args.task}.yaml")
    expected_type = "tracking" if args.task == "teleop" else args.task
    if task_config.get("type") != expected_type:
        parser.error(f"--task {args.task} requires task config type={expected_type}")
    if expected_type == "tracking":
        expected_source = "zmq" if args.task == "teleop" else "pkl"
        if task_config.get("ctx_source", "pkl") != expected_source:
            parser.error(f"--task {args.task} requires ctx_source={expected_source}")
    validate_task_config(task_config, expected_robot)
    policy_config = load_yaml(args.policy_config or model_path.with_suffix(".deploy.yaml"))

    if args.mode == "sim2real":
        from scripts.onboard.interface_config import validate_interface
        from scripts.onboard.deploy_helpers import Reporter

        interface = args.interface or robot_config.get("INTERFACE", "")
        if not validate_interface(interface, Reporter(), allow_default_route_interface=args.allow_default_route_interface):
            parser.error("Invalid onboard DDS interface")
        robot_config["INTERFACE"] = interface

    # The runtime validates model metadata before opening the robot interface.
    from rl_policy.ufo_policy import UFODeployPolicy

    policy = UFODeployPolicy(
        robot_config=robot_config, policy_config=policy_config,
        exp_config=task_config, model_path=str(model_path),
        pico_control=args.pico_control, pico_control_addr=args.pico_control_addr,
    )
    policy.run()


if __name__ == "__main__":
    main()
