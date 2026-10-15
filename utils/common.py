#!/usr/bin/env python3
"""Binary G1 state/command messages and shared ZMQ ports."""

import numpy as np
import struct

# ZMQ Port Configuration
PORTS = {"pelvis_pose": 5555, "low_state": 5590, "low_cmd": 5591}


class LowCmdMessage:
    """Binary message containing joint-space command targets."""

    def __init__(
        self,
        q_target: np.ndarray,
        dq_target: np.ndarray,
        tau_ff: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
    ):
        arrays = [
            np.asarray(q_target, dtype=np.float32),
            np.asarray(dq_target, dtype=np.float32),
            np.asarray(tau_ff, dtype=np.float32),
            np.asarray(kp, dtype=np.float32),
            np.asarray(kd, dtype=np.float32),
        ]

        length = arrays[0].size
        if any(arr.size != length for arr in arrays[1:]):
            raise ValueError("All arrays in LowCmdMessage must have the same length")

        self.q_target = arrays[0]
        self.dq_target = arrays[1]
        self.tau_ff = arrays[2]
        self.kp = arrays[3]
        self.kd = arrays[4]

    def to_bytes(self) -> bytes:
        count = self.q_target.size
        header = struct.pack('<I', count)
        payload = b''.join(
            arr.astype(np.float32, copy=False).tobytes()
            for arr in (self.q_target, self.dq_target, self.tau_ff, self.kp, self.kd)
        )
        return header + payload

    @classmethod
    def from_bytes(cls, data: bytes) -> 'LowCmdMessage':
        if len(data) < 4:
            raise ValueError("LowCmdMessage data is too short")

        (count,) = struct.unpack('<I', data[:4])
        offset = 4
        segment_size = count * 4
        arrays = []
        for _ in range(5):
            end = offset + segment_size
            if end > len(data):
                raise ValueError("LowCmdMessage data is incomplete")
            arrays.append(np.frombuffer(data[offset:end], dtype=np.float32).copy())
            offset = end

        return cls(*arrays)


class LowStateMessage:
    """Binary message containing base and joint state information."""

    def __init__(
        self,
        quaternion: np.ndarray,
        gyroscope: np.ndarray,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        joint_torques: np.ndarray | None = None,
        tick: int = 0,
    ):
        self.quaternion = np.asarray(quaternion, dtype=np.float32)
        if self.quaternion.size != 4:
            raise ValueError("Quaternion must have exactly 4 elements")

        self.gyroscope = np.asarray(gyroscope, dtype=np.float32)
        if self.gyroscope.size != 3:
            raise ValueError("Gyroscope must have exactly 3 elements")

        self.joint_positions = np.asarray(joint_positions, dtype=np.float32)
        self.joint_velocities = np.asarray(joint_velocities, dtype=np.float32)
        if self.joint_positions.size != self.joint_velocities.size:
            raise ValueError("Joint position and velocity arrays must match in length")

        if joint_torques is not None:
            joint_torques = np.asarray(joint_torques, dtype=np.float32)
            if joint_torques.size != self.joint_positions.size:
                raise ValueError("Joint torque array must match joint positions length")
        self.joint_torques = joint_torques
        self.tick = int(tick)

    def to_bytes(self) -> bytes:
        count = self.joint_positions.size
        has_torque = 1 if self.joint_torques is not None else 0
        header = struct.pack('<III', count, self.tick, has_torque)

        payload_parts = [
            self.quaternion.astype(np.float32, copy=False).tobytes(),
            self.gyroscope.astype(np.float32, copy=False).tobytes(),
            self.joint_positions.astype(np.float32, copy=False).tobytes(),
            self.joint_velocities.astype(np.float32, copy=False).tobytes(),
        ]

        if self.joint_torques is not None:
            payload_parts.append(self.joint_torques.astype(np.float32, copy=False).tobytes())

        return header + b''.join(payload_parts)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'LowStateMessage':
        header_size = struct.calcsize('<III')
        if len(data) < header_size + 28:  # header + quaternion(16) + gyro(12)
            raise ValueError("LowStateMessage data is too short")

        count, tick, has_torque = struct.unpack('<III', data[:header_size])
        offset = header_size

        quat_end = offset + 16
        gyro_end = quat_end + 12
        quaternion = np.frombuffer(data[offset:quat_end], dtype=np.float32).copy()
        gyroscope = np.frombuffer(data[quat_end:gyro_end], dtype=np.float32).copy()

        segment_size = count * 4
        pos_end = gyro_end + segment_size
        vel_end = pos_end + segment_size
        if vel_end > len(data):
            raise ValueError("LowStateMessage joint data is incomplete")

        joint_positions = np.frombuffer(data[gyro_end:pos_end], dtype=np.float32).copy()
        joint_velocities = np.frombuffer(data[pos_end:vel_end], dtype=np.float32).copy()

        joint_torques = None
        if has_torque:
            torque_end = vel_end + segment_size
            if torque_end > len(data):
                raise ValueError("LowStateMessage torque data is incomplete")
            joint_torques = np.frombuffer(data[vel_end:torque_end], dtype=np.float32).copy()

        return cls(
            quaternion=quaternion,
            gyroscope=gyroscope,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            joint_torques=joint_torques,
            tick=tick,
        )
