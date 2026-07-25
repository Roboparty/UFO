"""Vendored motion_tracking PICO/XRobot to G1 qpos retargeting.

The package initializer intentionally avoids importing MuJoCo/Mink-dependent runtime
modules, so command-line help and lightweight metadata checks can run before teleop
dependencies are installed.
"""

from .helper import ParsedXRobotMotionSnapshot, default_controller_buttons, parse_xrobot_motion_snapshot
from .params import XR_BODY_JOINT_NAMES

__all__ = [
    "ParsedXRobotMotionSnapshot",
    "XR_BODY_JOINT_NAMES",
    "default_controller_buttons",
    "parse_xrobot_motion_snapshot",
]
