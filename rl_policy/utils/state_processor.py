import numpy as np
import zmq
import time


from utils.strings import unitree_joint_names
from loguru import logger
from utils.common import PORTS, LowStateMessage

class StateProcessor:
    """Listens to the unitree sdk channels and converts observation into isaac compatible order.
    Assumes the message in the channel follows the joint order of unitree_joint_names.
    """
    def __init__(self, robot_config, dest_joint_names):
        self.robot_type = robot_config["ROBOT_TYPE"]
        self.state_timeout_s = float(robot_config.get("LOW_STATE_TIMEOUT_MS", 200)) / 1000.0
        if not np.isfinite(self.state_timeout_s) or self.state_timeout_s <= 0:
            raise ValueError("LOW_STATE_TIMEOUT_MS must be finite and positive")
        self.last_state_time = None
        self.last_state_tick = None
        # Initialize state source
        if self.robot_type == "g1_real":
            self.robot = robot_config["robot"]
        else:
            supported_types = {"g1_29dof"}
            if self.robot_type not in supported_types:
                raise NotImplementedError(
                    f"Robot type {self.robot_type} is not supported by the UFO-Deploy release runtime"
                )

            self.low_state_port = robot_config.get(
                "LOW_STATE_PORT", PORTS["low_state"]
            )
            state_host = robot_config.get("LOW_STATE_HOST", "127.0.0.1")
            state_endpoint = f"tcp://{state_host}:{self.low_state_port}"

            self.zmq_context = zmq.Context.instance()
            self.low_state_socket: zmq.Socket = self.zmq_context.socket(zmq.SUB)
            self.low_state_socket.setsockopt(zmq.SUBSCRIBE, b"")
            self.low_state_socket.setsockopt(zmq.CONFLATE, 1)
            self.low_state_socket.setsockopt(zmq.RCVTIMEO, 10)
            self.low_state_socket.connect(state_endpoint)
            self.latest_low_state: LowStateMessage | None = None

        # Initialize joint mapping
        self.num_dof = len(dest_joint_names)
        self.joint_indices_in_source = [unitree_joint_names.index(name) for name in dest_joint_names]
        self.joint_names = dest_joint_names

        self.qpos = np.zeros(3 + 4 + self.num_dof)
        self.qvel = np.zeros(3 + 3 + self.num_dof)

        # create views of qpos and qvel
        self.root_pos_w = self.qpos[0:3]
        self.root_lin_vel_w = self.qvel[0:3]

        self.root_quat_b = self.qpos[3:7]
        self.root_ang_vel_b = self.qvel[3:6]

        self.joint_pos = self.qpos[7:]
        self.joint_vel = self.qvel[6:]

    def _prepare_low_state(self):
        if hasattr(self, "low_state_socket"):
            self._receive_low_state()
            if self.last_state_time is None or time.monotonic() - self.last_state_time > self.state_timeout_s:
                return False

            low_state = self.latest_low_state
            self.root_quat_b[:] = low_state.quaternion
            self.root_ang_vel_b[:] = low_state.gyroscope

            source_joint_pos = low_state.joint_positions
            source_joint_vel = low_state.joint_velocities
            for dst_idx, src_idx in enumerate(self.joint_indices_in_source):
                self.joint_pos[dst_idx] = source_joint_pos[src_idx]
                self.joint_vel[dst_idx] = source_joint_vel[src_idx]

            return True
        elif hasattr(self, "robot"):
            try:
                state = self.robot.read_low_state()
            except Exception as e:
                logger.warning(f"Failed to read G1 low state: {e}")
                return False

            if state is None:
                return False

            # The SDK returns a cached snapshot; freshness must use its receive tick.
            if not hasattr(state, "tick"):
                logger.error("G1 SDK must expose low-state tick for stale-state detection")
                return False
            if not self._valid_state(state.imu.quat, state.imu.omega, state.motor.q, state.motor.dq):
                return False
            if state.tick != self.last_state_tick:
                self.last_state_tick = state.tick
                self.last_state_time = time.monotonic()
            if self.last_state_time is None or time.monotonic() - self.last_state_time > self.state_timeout_s:
                return False

            # IMU
            self.root_quat_b[:] = state.imu.quat  # [w, x, y, z]
            self.root_ang_vel_b[:] = state.imu.omega

            # Joints
            for dst_idx, src_idx in enumerate(self.joint_indices_in_source):
                self.joint_pos[dst_idx] = state.motor.q[src_idx]
                self.joint_vel[dst_idx] = state.motor.dq[src_idx]
            return True

    def _valid_state(self, quat, omega, q, dq):
        count = max(self.joint_indices_in_source) + 1
        arrays = [np.asarray(value) for value in (quat, omega, q, dq)]
        return (
            arrays[0].shape == (4,) and arrays[1].shape == (3,)
            and arrays[2].ndim == 1 and arrays[3].ndim == 1
            and arrays[2].size >= count and arrays[3].size >= count
            and all(np.all(np.isfinite(value)) for value in arrays)
            and abs(float(np.linalg.norm(arrays[0])) - 1.0) < 0.1
        )

    def _receive_low_state(self):
        """Fetch the most recent low state message from the ZMQ socket."""
        if not hasattr(self, "low_state_socket"):
            return

        while True:
            try:
                data = self.low_state_socket.recv(flags=zmq.DONTWAIT)
            except zmq.Again:
                break
            try:
                state = LowStateMessage.from_bytes(data)
                if not self._valid_state(state.quaternion, state.gyroscope, state.joint_positions, state.joint_velocities):
                    continue
                if state.tick != self.last_state_tick:
                    self.latest_low_state = state
                    self.last_state_tick = state.tick
                    self.last_state_time = time.monotonic()
            except Exception as exc:
                logger.warning(f"Failed to decode low state message: {exc}")
