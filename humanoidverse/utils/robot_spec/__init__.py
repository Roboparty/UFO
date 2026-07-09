"""Robot specification helpers for robot-aware motion data."""

from humanoidverse.utils.robot_spec.mujoco_parser import load_robot_spec
from humanoidverse.utils.robot_spec.robot_spec import RobotSpec
from humanoidverse.utils.robot_spec.training_spec import (
    RobotTrainingSpec,
    assert_robot_configs_compatible,
    load_robot_training_spec,
    resolve_robot_config_path,
)

__all__ = [
    "RobotSpec",
    "RobotTrainingSpec",
    "assert_robot_configs_compatible",
    "load_robot_spec",
    "load_robot_training_spec",
    "resolve_robot_config_path",
]
