import abc
import dataclasses
import re
from typing import Optional

import mujoco
import numpy as np
from dm_control.utils import rewards

COORD_TO_INDEX = {"x": 0, "y": 1, "z": 2}
ALIGNMENT_BOUNDS = {"x": (-0.1, 0.1), "z": (0.9, float("inf")), "y": (-0.1, 0.1)}

GRAVITY_VECTOR = np.array([0, 0, -1], dtype=np.float32)

REWARD_LIMITS = {
    # "l": [0, 0.5, 0.2], # get down to the knew
    "l": [0.6, 0.8, 0.2],
    "m": [1.0, float("inf"), 0.1],
    # "h": [1.3, float("inf"), 0.1],  # TODO this won't work, hard to find positions with hands above 1.3
}


def add_visual_arrow(renderer, point1, point2, rgba):
    """Adds an arrow to an mjvScene."""
    if renderer is None:
        return
    if not isinstance(rgba, np.ndarray):
        rgba = np.array(rgba).astype(np.float32)
    if not isinstance(point1, np.ndarray):
        point1 = np.array(point1).astype(np.float32)
    if not isinstance(point2, np.ndarray):
        point2 = np.array(point2).astype(np.float32)
    scene = renderer.scene
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        rgba.astype(np.float32),
    )
    mujoco.mjv_connector(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_ARROW,
        0.02,
        point1,
        point2,
    )


def add_arrow_from_xpos_to_direction(renderer, xpos, vector, rgba):
    point1 = xpos
    point2 = xpos + vector
    add_visual_arrow(renderer, point1, point2, rgba)


def rot2eul(R: np.ndarray):
    beta = -np.arcsin(R[2, 0])
    alpha = np.arctan2(R[2, 1] / np.cos(beta), R[2, 2] / np.cos(beta))
    gamma = np.arctan2(R[1, 0] / np.cos(beta), R[0, 0] / np.cos(beta))
    return np.array((alpha, beta, gamma))


def get_xpos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert index > -1
    xpos = data.xpos[index].copy()
    return xpos


def get_xmat(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert index > -1
    xmat = data.xmat[index].reshape((3, 3)).copy()
    return xmat


def get_torso_upright(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    _index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    assert _index > -1
    _upright = data.xmat[_index][-2]
    return _upright


def get_center_of_mass_linvel(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    chest_subtree_linvel_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_link_subtreelinvel")  # in global coordinate
    start = model.sensor_adr[chest_subtree_linvel_index]
    end = start + model.sensor_dim[chest_subtree_linvel_index]
    center_of_mass_velocity = data.sensordata[start:end].copy()
    return center_of_mass_velocity


def get_sensor_data(model: mujoco.MjModel, data: mujoco.MjData, name: str):
    chest_gyro_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)  # in global coordinate
    assert chest_gyro_index > -1
    start = model.sensor_adr[chest_gyro_index]
    end = start + model.sensor_dim[chest_gyro_index]
    sensord = data.sensordata[start:end].copy()
    return sensord


def quat_rotate_inverse_numpy(q, v):
    # shape = q.shape
    # q_w corresponds to the scalar part of the quaternion
    q_w = q[:, 0]
    # q_vec corresponds to the vector part of the quaternion
    q_vec = q[:, 1:]

    # Calculate a
    a = v * (2.0 * q_w**2 - 1.0)[:, np.newaxis]

    # Calculate b
    b = np.cross(q_vec, v) * q_w[:, np.newaxis] * 2.0

    # Calculate c
    dot_product = np.sum(q_vec * v, axis=1, keepdims=True)
    c = q_vec * dot_product * 2.0

    return a - b + c


class RewardFunction(abc.ABC):
    @abc.abstractmethod
    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float: ...

    @staticmethod
    @abc.abstractmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]: ...

    def __call__(
        self,
        model: mujoco.MjModel,
        qpos: np.ndarray,
        qvel: np.ndarray,
        ctrl: np.ndarray,
    ):
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.ctrl[:] = ctrl
        mujoco.mj_forward(model, data)
        return self.compute(model, data)


@dataclasses.dataclass
class ZeroReward(RewardFunction):
    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        return 0.0

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        if name.lower() in ["none", "zero", "rewardfree"]:
            return ZeroReward()
        return None


# @dataclasses.dataclass
# class StandPose(RewardFunction):
#     desired_pose: np.ndarray
#     pelvis_stand_height: float = 0.8
#     head_stand_height: float = 1.2

#     def compute(
#         self,
#         model: mujoco.MjModel,
#         data: mujoco.MjData,
#     ) -> float:
#         # standing_pelvis = rewards.tolerance(
#         #     data.body("pelvis").xpos[-1],
#         #     bounds=(self.pelvis_stand_height, float("inf")),
#         #     margin=self.pelvis_stand_height,
#         #     value_at_margin=0.01,
#         #     sigmoid="linear",
#         # )
#         standing_head = rewards.tolerance(
#             data.geom("head").xpos[-1],
#             bounds=(self.head_stand_height, float("inf")),
#             margin=self.head_stand_height,
#             value_at_margin=0.01,
#             sigmoid="linear",
#         )
#         # qpos_dist = rewards.tolerance(np.abs(data.qpos[7:] - self.desired_pose), margin=2, sigmoid="linear").mean()
#         # rew = (2 + qpos_dist) / 3
#         # rew = standing_pelvis * rew
#         # rew = (1.0 + rew) / 2.0
#         # return standing_head * rew

#         # distance in qpos
#         qpos_dist = rewards.tolerance(np.abs(data.qpos[7:] - self.desired_pose), margin=2, sigmoid="linear").mean()

#         # vertical position of the torso
#         pel = model.site("imu_in_pelvis").id
#         gravity = data.site_xmat[pel].reshape(3, 3).T @ GRAVITY_VECTOR
#         gravity[-1] += 1  # target gravity is [0,0,-1] so we add 1 to make it centered to 0
#         y = rewards.tolerance(gravity, margin=1, sigmoid="linear").mean()
#         rew = (1 + qpos_dist) / 2

#         return y * rew * standing_head

#     @staticmethod
#     def reward_from_name(name: str) -> Optional["RewardFunction"]:
#         if name.lower() == "stand_default":
#             return StandPose(
#                 desired_pose=np.array(
#                     [
#                         -0.1,
#                         0.0,
#                         0.0,
#                         0.3,
#                         -0.2,
#                         0.0,
#                         -0.1,
#                         0.0,
#                         0.0,
#                         0.3,
#                         -0.2,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ]
#                 )
#             )
#         else:
#             return None


@dataclasses.dataclass
class LocomotionReward(RewardFunction):
    move_speed: float = 5
    # Head height of G1 robot after "Default" reset is 1.22
    stand_height: float = 0.5
    move_angle: float = 0
    egocentric_target: bool = True
    stay_low: bool = False
    strict_velocity: bool = False
    target_yaw_velocity: float = 0.0

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        root_height = get_xpos(model, data, "pelvis")[-1]
        # head_height = data.geom("head").xpos[-1]
        # torso_upright = get_torso_upright(model, data)
        # base_quat = data.qpos[3:7].copy().reshape(1, -1)
        # v = np.array([[0, 0, -1]])
        # gravity = quat_rotate_inverse_numpy(base_quat, v).ravel()
        center_of_mass_velocity = get_center_of_mass_linvel(model, data)
        if self.move_angle is not None:
            move_angle = np.deg2rad(self.move_angle)
        if self.egocentric_target:
            pelvis_xmat = get_xmat(model, data, name="pelvis")
            euler = rot2eul(pelvis_xmat)
            move_angle = move_angle + euler[-1]

        if self.stay_low:
            standing = rewards.tolerance(
                root_height,
                bounds=(self.stand_height*0.95, self.stand_height*1.05),
                margin=self.stand_height / 2,
                value_at_margin=0.01,
                sigmoid="linear",
            )
        else:
            standing = rewards.tolerance(
                root_height,
                bounds=(self.stand_height, float("inf")),
                margin=self.stand_height,
                value_at_margin=0.01,
                sigmoid="linear",
            )
        # upright = rewards.tolerance(
        #             torso_upright,
        #             bounds=(-0.1, 0.1),
        #             margin=0.8,
        #             value_at_margin=0,
        #             sigmoid="linear",
        #         )
        # gravity_upright = rewards.tolerance(
        #     -gravity[-1],
        #     bounds=(0.9, float("inf")),
        #     sigmoid="linear",
        #     margin=1.9,
        #     value_at_margin=0,
        # )
        upvector_torso = get_sensor_data(model, data, "upvector_torso")
        cost_orientation = rewards.tolerance(
            np.sum(np.square(upvector_torso - np.array([0.073, 0.0, 1.0]))),
            bounds=(0, 0.1),
            margin=3,
            value_at_margin=0,
            sigmoid="linear",
        )
        stand_reward = standing * cost_orientation
        # small_control = rewards.tolerance(data.ctrl, margin=1, value_at_margin=0, sigmoid="quadratic").mean()
        # small_control = (4 + small_control) / 5
        small_control = 1.0
        if 0<= self.move_speed <= 0.01:
            horizontal_velocity = center_of_mass_velocity[[0, 1]]
            dont_move = rewards.tolerance(horizontal_velocity, margin=0.2).mean()
            angular_velocity = get_sensor_data(model, data, "imu-angular-velocity")
            dont_rotate = rewards.tolerance(angular_velocity, margin=0.1).mean()
            return small_control * stand_reward * dont_move * dont_rotate
        else:
            vel = center_of_mass_velocity[[0, 1]]
            if self.strict_velocity:
                target_direction = np.array([np.cos(move_angle), np.sin(move_angle)])
                lateral_direction = np.array([-target_direction[1], target_direction[0]])
                forward_velocity = target_direction.dot(vel)
                lateral_velocity = lateral_direction.dot(vel)
                forward_reward = rewards.tolerance(
                    forward_velocity,
                    bounds=(
                        self.move_speed - 0.1 * self.move_speed,
                        self.move_speed + 0.1 * self.move_speed,
                    ),
                    margin=self.move_speed / 2,
                    value_at_margin=0.5,
                    sigmoid="gaussian",
                )
                lateral_reward = rewards.tolerance(
                    lateral_velocity,
                    bounds=(-0.05, 0.05),
                    margin=0.5,
                    value_at_margin=0.1,
                    sigmoid="gaussian",
                )
                yaw_velocity = get_sensor_data(model, data, "imu-angular-velocity")[2]
                yaw_reward = rewards.tolerance(
                    yaw_velocity,
                    bounds=(self.target_yaw_velocity - 0.1, self.target_yaw_velocity + 0.1),
                    margin=0.5,
                    value_at_margin=0.1,
                    sigmoid="gaussian",
                )
                return small_control * stand_reward * forward_reward * lateral_reward * yaw_reward
            com_velocity = np.linalg.norm(vel)
            move = rewards.tolerance(
                com_velocity,
                bounds=(
                    self.move_speed - 0.1 * self.move_speed,
                    self.move_speed + 0.1 * self.move_speed,
                ),
                margin=self.move_speed / 2,
                value_at_margin=0.5,
                sigmoid="gaussian",
            )
            move = (5 * move + 1) / 6
            # move in a specific direction
            if np.isclose(com_velocity, 0.0) or move_angle is None:
                angle_reward = 1.0
            else:
                direction = vel / (com_velocity + 1e-6)
                target_direction = np.array([np.cos(move_angle), np.sin(move_angle)])
                dot = target_direction.dot(direction)
                angle_reward = (dot + 1.0) / 2.0
            reward = small_control * stand_reward * move * angle_reward
            return reward

    def render(self, renderer, model, data):
        if renderer is None:
            return
        pelvis_xpos = get_xpos(model, data, "pelvis")
        com_velocity = get_center_of_mass_linvel(model, data)
        com_velocity[2] = 0  # ignore vertical velocity

        move_angle = np.deg2rad(self.move_angle)
        if self.egocentric_target:
            pelvis_xmat = get_xmat(model, data, name="pelvis")
            euler = rot2eul(pelvis_xmat)
            move_angle = move_angle + euler[-1]
        target_direction = np.array([np.cos(move_angle), np.sin(move_angle), 0])
        target_direction = target_direction * self.move_speed

        # Visualize center of mass velocity
        add_arrow_from_xpos_to_direction(renderer, pelvis_xpos, com_velocity, (0, 1, 0, 1))

        # Visualize target direction
        add_arrow_from_xpos_to_direction(renderer, pelvis_xpos, target_direction, (1, 0, 0, 1))

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^move-ego-strict-(-?\d+\.*\d*)-(-?\d+\.*\d*)-(-?\d+\.*\d*)$"
        match = re.search(pattern, name)
        if match:
            move_angle, move_speed, target_yaw_velocity = (
                float(match.group(1)),
                float(match.group(2)),
                float(match.group(3)),
            )
            return LocomotionReward(
                move_angle=move_angle,
                move_speed=move_speed,
                strict_velocity=True,
                target_yaw_velocity=target_yaw_velocity,
            )
        pattern = r"^move-ego-(-?\d+\.*\d*)-(-?\d+\.*\d*)$"
        match = re.search(pattern, name)
        if match:
            move_angle, move_speed = float(match.group(1)), float(match.group(2))
            return LocomotionReward(move_angle=move_angle, move_speed=move_speed)
        pattern = r"^move-ego-low(-?\d+\.*\d*)-(-?\d+\.*\d*)-(-?\d+\.*\d*)$"
        match = re.search(pattern, name)
        if match:
            stand_height, move_angle, move_speed = float(match.group(1)), float(match.group(2)), float(match.group(3))
            print(stand_height, move_angle, move_speed)
            return LocomotionReward(move_angle=move_angle, move_speed=move_speed, stay_low=True, stand_height=stand_height)
        return None


@dataclasses.dataclass
class HighKneeLocomotionReward(LocomotionReward):
    """Forward locomotion reward with an extra swing-foot lift preference.

    The lift term uses the relative height difference between the two foot
    sites instead of absolute foot height. This keeps the reward focused on
    gait swing clearance and avoids rewarding a two-foot jump where both feet
    rise together.
    """

    foot_lift_height: float = 0.08
    foot_lift_weight: float = 0.5

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        base_reward = super().compute(model, data)
        left_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        right_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
        if left_site_id < 0 or right_site_id < 0:
            raise RuntimeError("HighKneeLocomotionReward requires left_foot/right_foot sites")

        left_height = float(data.site_xpos[left_site_id, 2])
        right_height = float(data.site_xpos[right_site_id, 2])
        relative_swing_height = abs(left_height - right_height)
        lift_reward = rewards.tolerance(
            relative_swing_height,
            bounds=(self.foot_lift_height, float("inf")),
            margin=max(self.foot_lift_height, 1e-6),
            value_at_margin=0.1,
            sigmoid="linear",
        )
        lift_weight = float(np.clip(self.foot_lift_weight, 0.0, 1.0))
        return base_reward * ((1.0 - lift_weight) + lift_weight * lift_reward)

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^move-ego-highknee-(-?\d+\.*\d*)-(-?\d+\.*\d*)-(-?\d+\.*\d*)$"
        match = re.search(pattern, name)
        if match:
            move_angle, move_speed, foot_lift_height = (
                float(match.group(1)),
                float(match.group(2)),
                float(match.group(3)),
            )
            return HighKneeLocomotionReward(
                move_angle=move_angle,
                move_speed=move_speed,
                foot_lift_height=foot_lift_height,
            )

        pattern = r"^move-ego-highknee-(-?\d+\.*\d*)-(-?\d+\.*\d*)-(-?\d+\.*\d*)-(-?\d+\.*\d*)$"
        match = re.search(pattern, name)
        if match:
            move_angle, move_speed, foot_lift_height, foot_lift_weight = (
                float(match.group(1)),
                float(match.group(2)),
                float(match.group(3)),
                float(match.group(4)),
            )
            return HighKneeLocomotionReward(
                move_angle=move_angle,
                move_speed=move_speed,
                foot_lift_height=foot_lift_height,
                foot_lift_weight=foot_lift_weight,
            )
        return None


@dataclasses.dataclass
class JumpReward(RewardFunction):
    jump_height: float = 1.4
    max_velocity: float = 5.0

    def compute(self, model: mujoco.MjModel, data: mujoco.MjData) -> float:
        head_height = data.geom("head").xpos[-1]
        chest_upright = get_torso_upright(model, data)
        center_of_mass_velocity = get_center_of_mass_linvel(model, data)

        jumping = rewards.tolerance(
            head_height,
            bounds=(self.jump_height, self.jump_height + 0.1),
            margin=self.jump_height,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        upright = rewards.tolerance(
            chest_upright,
            bounds=(0.9, float("inf")),
            sigmoid="linear",
            margin=1.9,
            value_at_margin=0,
        )
        up_velocity = rewards.tolerance(
            center_of_mass_velocity[-1],
            bounds=(self.max_velocity, float("inf")),
            sigmoid="linear",
            margin=self.max_velocity,
            value_at_margin=0,
        )
        reward = jumping * upright * up_velocity
        return reward

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^jump-(\d+\.*\d*)$"
        match = re.search(pattern, name)
        if match:
            jump_height = float(match.group(1))
            return JumpReward(jump_height=jump_height)
        return None

    def render(self, renderer, model, data):
        if renderer is None:
            return
        head_xpos = data.geom("head").xpos
        target_height = np.array([head_xpos[0], head_xpos[1], self.jump_height])
        current_height = np.array([head_xpos[0], head_xpos[1], head_xpos[2]])

        # Visualize target height
        color = (0, 1, 0, 1) if head_xpos[2] >= self.jump_height else (1, 0, 0, 1)
        add_visual_arrow(renderer, current_height, target_height, color)


@dataclasses.dataclass
class RotationReward(RewardFunction):
    axis: str = "x"
    target_ang_velocity: float = 5.0
    # Note: pelvis height is 0.8 exactly after reset with default pose
    stand_pelvis_height: float = 0.8

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        pelvis_height = get_xpos(model, data, name="pelvis")[-1]
        pelvis_xmat = get_xmat(model, data, name="pelvis")
        torso_rotation = pelvis_xmat[2, :].ravel()
        angular_velocity = get_sensor_data(model, data, "imu-angular-velocity")

        height_reward = rewards.tolerance(
            pelvis_height,
            bounds=(self.stand_pelvis_height, float("inf")),
            margin=self.stand_pelvis_height,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        direction = np.sign(self.target_ang_velocity)

        small_control = rewards.tolerance(data.ctrl, margin=1, value_at_margin=0, sigmoid="quadratic").mean()
        small_control = (4 + small_control) / 5
        small_control = 1

        targ_av = np.abs(self.target_ang_velocity)
        move = rewards.tolerance(
            direction * angular_velocity[COORD_TO_INDEX[self.axis]],
            bounds=(targ_av, targ_av + 5),
            margin=targ_av / 2,
            value_at_margin=0,
            sigmoid="linear",
        )

        aligned = rewards.tolerance(
            torso_rotation[COORD_TO_INDEX[self.axis]],
            bounds=ALIGNMENT_BOUNDS[self.axis],
            sigmoid="linear",
            margin=0.9,
            value_at_margin=0,
        )

        reward = move * height_reward * small_control * aligned
        return reward

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^rotate-(x|y|z)-(-?\d+\.*\d*)-(\d+\.*\d*)$"
        match = re.search(pattern, name)
        if match:
            axis, target_ang_velocity, stand_pelvis_height = (
                match.group(1),
                float(match.group(2)),
                float(match.group(3)),
            )
            return RotationReward(
                axis=axis,
                target_ang_velocity=target_ang_velocity,
                stand_pelvis_height=stand_pelvis_height,
            )
        return None


@dataclasses.dataclass
class ArmsReward(RewardFunction):
    # head height of the character in T-pose, this is used as a reference to compute whether the character is standing
    left_pose: str = "m"
    right_pose: str = "m"
    # target height for standing pose
    stand_height: float = 0.5

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        root_height = get_xpos(model, data, "pelvis")[-1]
        left_limits = REWARD_LIMITS[self.left_pose]
        right_limits = REWARD_LIMITS[self.right_pose]
        center_of_mass_velocity = get_center_of_mass_linvel(model, data)
        left_height = data.body("left_wrist_roll_link").xpos[-1]
        right_height = data.body("right_wrist_roll_link").xpos[-1]
        standing = rewards.tolerance(
            root_height,
            bounds=(self.stand_height, float("inf")),
            margin=self.stand_height,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        upvector_torso = get_sensor_data(model, data, "upvector_torso")
        cost_orientation = rewards.tolerance(
            np.sum(np.square(upvector_torso - np.array([0.073, 0.0, 1.0]))),
            bounds=(0, 0.1),
            margin=3,
            value_at_margin=0,
            sigmoid="linear",
        )
        stand_reward = standing * cost_orientation
        dont_move = rewards.tolerance(center_of_mass_velocity, margin=0.2).mean()
        angular_velocity = get_sensor_data(model, data, "imu-angular-velocity")
        dont_rotate = rewards.tolerance(angular_velocity, margin=0.1).mean()
        # dont_move = rewards.tolerance(data.qvel, margin=0.5).mean()
        # small_control = rewards.tolerance(data.ctrl, margin=1, value_at_margin=0, sigmoid="quadratic").mean()
        # small_control = (4 + small_control) / 5
        small_control = 1
        left_arm = rewards.tolerance(
            left_height,
            bounds=(left_limits[0], left_limits[1]),
            margin=left_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        left_arm = (4 * left_arm + 1) / 5
        right_arm = rewards.tolerance(
            right_height,
            bounds=(right_limits[0], right_limits[1]),
            margin=right_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        right_arm = (4 * right_arm + 1) / 5

        return small_control * stand_reward * dont_move * left_arm * right_arm * dont_rotate

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^raisearms-(l|m|h|x)-(l|m|h|x)"
        match = re.search(pattern, name)
        if match:
            left_pose, right_pose = match.group(1), match.group(2)
            return ArmsReward(left_pose=left_pose, right_pose=right_pose)
        return None


@dataclasses.dataclass
class SitOnGroundReward(RewardFunction):

    pelvis_height_th: float = 0
    constrained_knees: bool = False
    knees_not_on_ground: bool = False

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        pelvis_height = get_xpos(model, data, name="pelvis")[-1]
        left_knee_pos = get_xpos(model, data, name="left_knee_link")[-1]
        right_knee_pos = get_xpos(model, data, name="right_knee_link")[-1]
        center_of_mass_velocity = get_center_of_mass_linvel(model, data)
        upvector_torso = get_sensor_data(model, data, "upvector_torso")
        cost_orientation = rewards.tolerance(
            np.sum(np.square(upvector_torso - np.array([0.073, 0.0, 1.0]))),
            bounds=(0, 0.1),
            margin=3,
            value_at_margin=0,
            sigmoid="linear",
        )
        dont_move = rewards.tolerance(center_of_mass_velocity, margin=0.5).mean()
        angular_velocity = get_sensor_data(model, data, "imu-angular-velocity")
        dont_rotate = rewards.tolerance(angular_velocity, margin=0.1).mean()
        # small_control = rewards.tolerance(data.ctrl, margin=1, value_at_margin=0, sigmoid="quadratic").mean()
        # small_control = (4 + small_control) / 5
        small_control = 1
        pelvis_reward = rewards.tolerance(
            pelvis_height,
            bounds=(self.pelvis_height_th, self.pelvis_height_th + 0.1),
            sigmoid="linear",
            margin=0.7,
            value_at_margin=0,
        )
        knee_reward = 1
        if self.constrained_knees:
            knee_reward *= rewards.tolerance(
                left_knee_pos,
                bounds=(0, 0.1),
                sigmoid="linear",
                margin=0.7,
                value_at_margin=0,
            )
            knee_reward *= rewards.tolerance(
                right_knee_pos,
                bounds=(0, 0.1),
                sigmoid="linear",
                margin=0.7,
                value_at_margin=0,
            )
        if self.knees_not_on_ground:
            knee_reward *= rewards.tolerance(
                left_knee_pos,
                bounds=(0.2, 1),
                sigmoid="linear",
                margin=0.1,
                value_at_margin=0,
            )
            knee_reward *= rewards.tolerance(
                right_knee_pos,
                bounds=(0.2, 1),
                sigmoid="linear",
                margin=0.1,
                value_at_margin=0,
            )
        return small_control * cost_orientation * dont_move * dont_rotate * pelvis_reward * (2*knee_reward+1)/3

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        if name == "sitonground":
            pelvis_height_th = 0
            return SitOnGroundReward(pelvis_height_th=pelvis_height_th, constrained_knees=True)
        pattern = r"^crouch-(\d+\.*\d*)$"
        match = re.search(pattern, name)
        if match:
            pelvis_height_th = float(match.group(1))
            return SitOnGroundReward(pelvis_height_th=pelvis_height_th, knees_not_on_ground=True)
        return None
    

@dataclasses.dataclass
class ToTheKnee(RewardFunction):
    stand_height: float = 0.4

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        root_height = get_xpos(model, data, "pelvis")[-1]
        left_limits = [0, 0.4, 0.2]
        right_limits = [0, 0.4, 0.2]
        center_of_mass_velocity = get_center_of_mass_linvel(model, data)
        left_height = data.body("left_wrist_roll_link").xpos[-1]
        right_height = data.body("right_wrist_roll_link").xpos[-1]
        standing = rewards.tolerance(
            root_height,
            bounds=(self.stand_height, float("inf")),
            margin=self.stand_height,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        upvector_torso = get_sensor_data(model, data, "upvector_torso")
        cost_orientation = rewards.tolerance(
            np.sum(np.square(upvector_torso - np.array([0.073, 0.0, 1.0]))),
            bounds=(0, 0.1),
            margin=3,
            value_at_margin=0,
            sigmoid="linear",
        )
        stand_reward = standing * cost_orientation
        dont_move = rewards.tolerance(center_of_mass_velocity, margin=0.2).mean()
        # dont_move = rewards.tolerance(data.qvel, margin=0.5).mean()
        # small_control = rewards.tolerance(data.ctrl, margin=1, value_at_margin=0, sigmoid="quadratic").mean()
        # small_control = (4 + small_control) / 5
        small_control = 1
        left_arm = rewards.tolerance(
            left_height,
            bounds=(left_limits[0], left_limits[1]),
            margin=left_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        left_arm = (4 * left_arm + 1) / 5
        right_arm = rewards.tolerance(
            right_height,
            bounds=(right_limits[0], right_limits[1]),
            margin=right_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        right_arm = (4 * right_arm + 1) / 5
        return small_control * stand_reward * dont_move * left_arm * right_arm

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        if name.lower() == "totheknee":
            return ToTheKnee()


# @dataclasses.dataclass
# class StandPoseRelative(RewardFunction):
#     """Attempt at stand_pose but with only local info"""

#     projected_head_to_ankle_distance: float = 1.1

#     def compute(
#         self,
#         model: mujoco.MjModel,
#         data: mujoco.MjData,
#     ) -> float:
#         # Concept: head to feet distance projected on the gravity (we get both "upright" and "standing" from this)

#         # Using ankle roll links, as they are the lowest body part with a name in default robot file
#         right_ankle_link = get_xpos(model, data, "right_ankle_roll_link")
#         left_ankle_link = get_xpos(model, data, "left_ankle_roll_link")
#         average_ankle_pos = (right_ankle_link + left_ankle_link) / 2
#         head_pos = data.geom("head").xpos

#         head_to_feet = average_ankle_pos - head_pos

#         # Project to gravity direction
#         head_to_feet_projected_on_gravity = head_to_feet @ GRAVITY_VECTOR

#         reward = rewards.tolerance(
#             head_to_feet_projected_on_gravity,
#             bounds=(self.projected_head_to_ankle_distance, float("inf")),
#             margin=self.projected_head_to_ankle_distance,
#             value_at_margin=0.01,
#             sigmoid="linear",
#         )

#         return reward

#     @staticmethod
#     def reward_from_name(name: str) -> Optional["RewardFunction"]:
#         pattern = r"^stand-relative-(\d+\.*\d*)$"
#         match = re.search(pattern, name)
#         if match:
#             projected_head_to_ankle_distance = float(match.group(1))
#             return StandPoseRelative(projected_head_to_ankle_distance=projected_head_to_ankle_distance)
#         return None

#     def render(self, renderer, model, data):
#         if renderer is None:
#             return
#         right_ankle_link = get_xpos(model, data, "right_ankle_roll_link")
#         left_ankle_link = get_xpos(model, data, "left_ankle_roll_link")
#         average_ankle_pos = (right_ankle_link + left_ankle_link) / 2
#         head_pos = data.geom("head").xpos

#         head_to_feet = average_ankle_pos - head_pos
#         head_to_feet_project_dist = head_to_feet @ GRAVITY_VECTOR
#         head_to_feet_projected_on_gravity = GRAVITY_VECTOR * head_to_feet_project_dist

#         # Visualize the projected arrow
#         color = (0, 1, 0, 1) if head_to_feet_project_dist >= self.projected_head_to_ankle_distance else (1, 0, 0, 1)
#         source_pos = head_pos.copy()
#         source_pos[0] += 0.2
#         source_pos[1] += 0.2
#         add_arrow_from_xpos_to_direction(renderer, source_pos, head_to_feet_projected_on_gravity, color)


@dataclasses.dataclass
class MoveArmsReward(RewardFunction):
    move_speed: float = 5
    # Head height of G1 robot after "Default" reset is 1.22
    stand_height: float = 0.5
    move_angle: float = 0
    egocentric_target: bool = True
    low_height: float = 0.5
    stay_low: bool = False  # TODO this currently doesn't work because arm heights are expressed in global coords
    left_pose: str = "m"
    right_pose: str = "m"

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        root_height = get_xpos(model, data, "pelvis")[-1]
        center_of_mass_velocity = get_center_of_mass_linvel(model, data)
        if self.move_angle is not None:
            move_angle = np.deg2rad(self.move_angle)
        if self.egocentric_target:
            pelvis_xmat = get_xmat(model, data, name="pelvis")
            euler = rot2eul(pelvis_xmat)
            move_angle = move_angle + euler[-1]

        # STANDING HEIGHT
        if self.stay_low:
            standing = rewards.tolerance(
                root_height,
                bounds=(self.low_height / 2, self.low_height),
                margin=self.low_height / 2,
                value_at_margin=0.01,
                sigmoid="linear",
            )
        else:
            standing = rewards.tolerance(
                root_height,
                bounds=(self.stand_height, float("inf")),
                margin=self.stand_height,
                value_at_margin=0.01,
                sigmoid="linear",
            )

        # STANDING STRAIGHT
        upvector_torso = get_sensor_data(model, data, "upvector_torso")
        cost_orientation = rewards.tolerance(
            np.sum(np.square(upvector_torso - np.array([0.073, 0.0, 1.0]))),
            bounds=(0, 0.1),
            margin=3,
            value_at_margin=0,
            sigmoid="linear",
        )
        stand_reward = standing * cost_orientation

        # SMALL CONTROL
        # small_control = rewards.tolerance(data.ctrl, margin=1, value_at_margin=0, sigmoid="quadratic").mean()
        # small_control = (4 + small_control) / 5
        small_control = 1.0

        # ARM POSES
        left_limits = REWARD_LIMITS[self.left_pose]
        right_limits = REWARD_LIMITS[self.right_pose]
        left_height = data.body("left_wrist_roll_link").xpos[-1]
        right_height = data.body("right_wrist_roll_link").xpos[-1]
        left_arm = rewards.tolerance(
            left_height,
            bounds=(left_limits[0], left_limits[1]),
            margin=left_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        left_arm = (4 * left_arm + 1) / 5
        right_arm = rewards.tolerance(
            right_height,
            bounds=(right_limits[0], right_limits[1]),
            margin=right_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        right_arm = (4 * right_arm + 1) / 5

        if self.move_speed == 0:
            horizontal_velocity = center_of_mass_velocity[[0, 1]]
            dont_move = rewards.tolerance(horizontal_velocity, margin=0.2).mean()
            angular_velocity = get_sensor_data(model, data, "imu-angular-velocity")
            dont_rotate = rewards.tolerance(angular_velocity, margin=0.1).mean()
            reward = small_control * stand_reward * dont_move * dont_rotate * left_arm * right_arm
            return reward
        else:
            vel = center_of_mass_velocity[[0, 1]]
            com_velocity = np.linalg.norm(vel)
            move = rewards.tolerance(
                com_velocity,
                bounds=(
                    self.move_speed - 0.1 * self.move_speed,
                    self.move_speed + 0.1 * self.move_speed,
                ),
                margin=self.move_speed / 2,
                value_at_margin=0.5,
                sigmoid="gaussian",
            )
            move = (5 * move + 1) / 6
            # move in a specific direction
            if np.isclose(com_velocity, 0.0) or move_angle is None:
                angle_reward = 1.0
            else:
                direction = vel / (com_velocity + 1e-6)
                target_direction = np.array([np.cos(move_angle), np.sin(move_angle)])
                dot = target_direction.dot(direction)
                angle_reward = (dot + 1.0) / 2.0
            reward = small_control * stand_reward * move * angle_reward * left_arm * right_arm
            return reward

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^move-arms-(-?\d+\.*\d*)-(-?\d+\.*\d*)-(l|m|h|x)-(l|m|h|x)$"
        match = re.search(pattern, name)
        if match:
            move_angle, move_speed, left_pose, right_pose = float(match.group(1)), float(match.group(2)), match.group(3), match.group(4)
            return MoveArmsReward(move_angle=move_angle, move_speed=move_speed, left_pose=left_pose, right_pose=right_pose)
        pattern = r"^move-ego-low-(-?\d+\.*\d*)-(-?\d+\.*\d*)-(l|m|h|x)-(l|m|h|x)$"
        match = re.search(pattern, name)
        if match:
            move_angle, move_speed, left_pose, right_pose = float(match.group(1)), float(match.group(2)), match.group(3), match.group(4)
            return MoveArmsReward(move_angle=move_angle, move_speed=move_speed, stay_low=True, left_pose=left_pose, right_pose=right_pose)
        return None


@dataclasses.dataclass
class SpinArmsReward(RewardFunction):
    axis: str = "z"
    target_ang_velocity: float = 5.0
    stand_pelvis_height: float = 0.5
    left_pose: str = "m"
    right_pose: str = "m"

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        pelvis_height = get_xpos(model, data, name="pelvis")[-1]
        pelvis_xmat = get_xmat(model, data, name="pelvis")
        torso_rotation = pelvis_xmat[2, :].ravel()
        angular_velocity = get_sensor_data(model, data, "imu-angular-velocity")

        # PELVIS HEIGHT
        height_reward = rewards.tolerance(
            pelvis_height,
            bounds=(self.stand_pelvis_height, float("inf")),
            margin=self.stand_pelvis_height,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        direction = np.sign(self.target_ang_velocity)

        # SMALL CONTROL
        # small_control = rewards.tolerance(data.ctrl, margin=1, value_at_margin=0, sigmoid="quadratic").mean()
        # small_control = (4 + small_control) / 5
        small_control = 1

        # SPINNING
        targ_av = np.abs(self.target_ang_velocity)
        move = rewards.tolerance(
            direction * angular_velocity[COORD_TO_INDEX[self.axis]],
            bounds=(targ_av, targ_av + 5),
            margin=targ_av / 2,
            value_at_margin=0,
            sigmoid="linear",
        )

        # UPRIGHT
        aligned = rewards.tolerance(
            torso_rotation[COORD_TO_INDEX[self.axis]],
            bounds=ALIGNMENT_BOUNDS[self.axis],
            sigmoid="linear",
            margin=0.9,
            value_at_margin=0,
        )

        # ARM POSES
        left_limits = REWARD_LIMITS[self.left_pose]
        right_limits = REWARD_LIMITS[self.right_pose]
        left_height = data.body("left_wrist_roll_link").xpos[-1]
        right_height = data.body("right_wrist_roll_link").xpos[-1]
        left_arm = rewards.tolerance(
            left_height,
            bounds=(left_limits[0], left_limits[1]),
            margin=left_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        left_arm = (4 * left_arm + 1) / 5
        right_arm = rewards.tolerance(
            right_height,
            bounds=(right_limits[0], right_limits[1]),
            margin=right_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        right_arm = (4 * right_arm + 1) / 5

        reward = move * height_reward * small_control * aligned * left_arm * right_arm
        return reward

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^spin-arms-(-?\d+\.*\d*)-(l|m|h|x)-(l|m|h|x)$"
        match = re.search(pattern, name)
        if match:
            target_ang_velocity, left_pose, right_pose = (
                float(match.group(1)),
                match.group(2),
                match.group(3),
            )
            return SpinArmsReward(
                target_ang_velocity=target_ang_velocity,
                left_pose=left_pose,
                right_pose=right_pose,
            )
        return None
