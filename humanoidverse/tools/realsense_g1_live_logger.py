#!/usr/bin/env python3
"""Read-only synchronized D435i + G1 logger for perception bring-up.

This program never imports an Actor or a motor-command API.  It records the
native D435i Z16 stream and three read-only G1 state streams, maps their clocks
onto host monotonic time, interpolates state at each camera exposure, and saves
both the raw streams and synchronized poses in one NPZ.

The file deliberately remains Python 3.8 compatible because it runs on the G1
Jetson rather than in the training environment.
"""

from __future__ import print_function

import argparse
import json
import math
import os
import signal
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "ufo.realsense_g1_live.v2"
G1_JOINT_COUNT = 29
WAIST_YAW_INDEX = 12
WAIST_ROLL_INDEX = 13
WAIST_PITCH_INDEX = 14
TORSO_ORIGIN_IN_WAIST_YAW = np.asarray([-0.0039635, 0.0, 0.044], dtype=np.float64)
DEFAULT_DEPTH_WIDTH = 480
DEFAULT_DEPTH_HEIGHT = 270
DEFAULT_DEPTH_FPS = 60
INTENDED_RUNTIME_RATE_HZ = 50.0
INSTINCT_ONBOARD_COMMIT = "b6965a4202d8dceaf163dd95edb7742ed5d65e38"
INSTINCT_ONBOARD_ROBOT_CFG_URL = (
    "https://github.com/project-instinct/instinct_onboard/blob/"
    + INSTINCT_ONBOARD_COMMIT
    + "/instinct_onboard/robot_cfgs.py#L265-L279"
)
# p_camera_depth_link = (Z, -X, -Y) for p_optical = (X, Y, Z).
CAMERA_DEPTH_LINK_FROM_OPTICAL_MATRIX = np.asarray(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)
CAMERA_DEPTH_LINK_FROM_OPTICAL_QUAT_XYZW = np.asarray(
    [-0.5, 0.5, -0.5, 0.5],
    dtype=np.float64,
)


def _normalized_quaternion_xyzw(value, name="quaternion"):
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("{} must contain four finite values".format(name))
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise ValueError("{} must be non-zero".format(name))
    return quaternion / norm


def quaternion_multiply_xyzw(left, right):
    left = _normalized_quaternion_xyzw(left, "left quaternion")
    right = _normalized_quaternion_xyzw(right, "right quaternion")
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return _normalized_quaternion_xyzw(
        np.asarray(
            [
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
                lw * rw - lx * rx - ly * ry - lz * rz,
            ],
            dtype=np.float64,
        )
    )


def quaternion_slerp_xyzw(first, second, fraction):
    first = _normalized_quaternion_xyzw(first, "first quaternion")
    second = _normalized_quaternion_xyzw(second, "second quaternion")
    alpha = float(fraction)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("SLERP fraction must be in [0, 1]")
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return _normalized_quaternion_xyzw((1.0 - alpha) * first + alpha * second)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    return _normalized_quaternion_xyzw(
        math.sin((1.0 - alpha) * theta) / sin_theta * first
        + math.sin(alpha * theta) / sin_theta * second
    )


def _axis_quaternion_xyzw(axis, angle):
    half = 0.5 * float(angle)
    result = np.zeros(4, dtype=np.float64)
    result[int(axis)] = math.sin(half)
    result[3] = math.cos(half)
    return result


def rotate_vector_xyzw(quaternion, vector):
    x, y, z, w = _normalized_quaternion_xyzw(quaternion)
    rotation = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return rotation.dot(np.asarray(vector, dtype=np.float64))


def heading_quaternion_xyzw(quaternion):
    x, y, z, w = _normalized_quaternion_xyzw(quaternion)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return _axis_quaternion_xyzw(2, yaw)


def torso_pose_from_pelvis(pelvis_position, pelvis_quaternion, waist_angles):
    """Apply the G1 pelvis -> yaw -> roll -> pitch chain used by the model."""
    waist = np.asarray(waist_angles, dtype=np.float64)
    if waist.shape != (3,) or not np.isfinite(waist).all():
        raise ValueError("waist_angles must contain yaw, roll, and pitch")
    yaw_quaternion = _axis_quaternion_xyzw(2, waist[0])
    roll_quaternion = _axis_quaternion_xyzw(0, waist[1])
    pitch_quaternion = _axis_quaternion_xyzw(1, waist[2])
    relative_orientation = quaternion_multiply_xyzw(
        quaternion_multiply_xyzw(yaw_quaternion, roll_quaternion), pitch_quaternion
    )
    torso_quaternion = quaternion_multiply_xyzw(pelvis_quaternion, relative_orientation)
    relative_position = rotate_vector_xyzw(yaw_quaternion, TORSO_ORIGIN_IN_WAIST_YAW)
    torso_position = np.asarray(pelvis_position, dtype=np.float64) + rotate_vector_xyzw(
        pelvis_quaternion, relative_position
    )
    return torso_position, torso_quaternion


def quaternion_distance_degrees(first, second):
    first = _normalized_quaternion_xyzw(first)
    second = _normalized_quaternion_xyzw(second)
    dot = min(1.0, max(0.0, abs(float(np.dot(first, second)))))
    return math.degrees(2.0 * math.acos(dot))


def instinct_nominal_torso_from_optical():
    """Compose Instinct's G1 camera-depth-link mount with optical axes.

    The upstream static transform is ``T_torso_link_from_camera_depth_link``
    and uses a robot frame (+x forward, +y left, +z up).  The terrain adapter
    consumes the standard optical frame (+x right, +y down, +z forward), so
    the returned transform is explicitly ``T_torso_link_from_camera_optical``.
    """
    angle_48 = math.radians(48.0)
    angle_half_degree = math.radians(0.5)
    translation = np.asarray(
        [
            0.04764571478 + 0.0039635 - 0.0042 * math.cos(angle_48),
            0.015,
            0.46268178553 - 0.044 + 0.0042 * math.sin(angle_48) + 0.016,
        ],
        dtype=np.float64,
    )
    upstream_wxyz = np.asarray(
        [
            math.cos(angle_half_degree / 2.0) * math.cos(angle_48 / 2.0),
            math.sin(angle_half_degree / 2.0),
            math.sin(angle_48 / 2.0),
            0.0,
        ],
        dtype=np.float64,
    )
    torso_from_depth_link_xyzw = _normalized_quaternion_xyzw(
        upstream_wxyz[[1, 2, 3, 0]],
        "Instinct torso-from-camera-depth-link quaternion",
    )
    torso_from_optical_xyzw = quaternion_multiply_xyzw(
        torso_from_depth_link_xyzw,
        CAMERA_DEPTH_LINK_FROM_OPTICAL_QUAT_XYZW,
    )
    canonical_translation = translation.tolist()
    canonical_quaternion = torso_from_optical_xyzw.tolist()
    return {
        "transform_name": "T_torso_link_from_camera_optical",
        "torso_from_camera_optical_translation_m": canonical_translation,
        "torso_from_camera_optical_quaternion_xyzw": canonical_quaternion,
        # Compatibility fields consumed by RealSenseCalibration.
        "mount_pos_torso": canonical_translation,
        "optical_quat_torso_xyzw": canonical_quaternion,
        "reference_kind": "nominal_reference",
        "source_repository": "project-instinct/instinct_onboard",
        "source_commit": INSTINCT_ONBOARD_COMMIT,
        "source_url": INSTINCT_ONBOARD_ROBOT_CFG_URL,
        "composition": "T_torso_link_from_camera_depth_link @ T_camera_depth_link_from_camera_optical",
        "torso_from_camera_depth_link_translation_m": canonical_translation,
        "torso_from_camera_depth_link_quaternion_wxyz": upstream_wxyz.tolist(),
        "camera_depth_link_from_camera_optical_matrix": CAMERA_DEPTH_LINK_FROM_OPTICAL_MATRIX.tolist(),
        "optical_axis_mapping": "(X, Y, Z)_optical -> (Z, -X, -Y)_camera_depth_link",
    }


def _unique_sorted_series(times_ns, values, name):
    times = np.asarray(times_ns, dtype=np.int64)
    values = np.asarray(values)
    if times.ndim != 1 or values.shape[0] != times.shape[0] or times.size < 2:
        raise RuntimeError("{} needs at least two timestamped samples".format(name))
    order = np.argsort(times, kind="stable")
    times = times[order]
    values = values[order]
    keep = np.ones(times.size, dtype=bool)
    keep[:-1] = times[:-1] != times[1:]
    times = times[keep]
    values = values[keep]
    if times.size < 2:
        raise RuntimeError("{} timestamps are not advancing".format(name))
    return times, values


def interpolate_series(times_ns, values, targets_ns, quaternion=False):
    times, values = _unique_sorted_series(times_ns, values, "interpolation stream")
    targets = np.asarray(targets_ns, dtype=np.int64)
    output = []
    nearest_dt_ns = []
    bracket_span_ns = []
    for target in targets:
        right = int(np.searchsorted(times, target, side="left"))
        if right < times.size and times[right] == target:
            left = right
        else:
            left = right - 1
        if left < 0 or right >= times.size:
            raise RuntimeError(
                "state stream does not bracket camera time {} (range {}..{})".format(
                    int(target), int(times[0]), int(times[-1])
                )
            )
        if left == right:
            result = values[left]
            span = 0
            nearest = 0
        else:
            denominator = int(times[right] - times[left])
            fraction = float(int(target - times[left])) / float(denominator)
            if quaternion:
                result = quaternion_slerp_xyzw(values[left], values[right], fraction)
            else:
                result = (1.0 - fraction) * values[left] + fraction * values[right]
            span = denominator
            left_dt = int(times[left] - target)
            right_dt = int(times[right] - target)
            nearest = left_dt if abs(left_dt) <= abs(right_dt) else right_dt
        output.append(result)
        nearest_dt_ns.append(nearest)
        bracket_span_ns.append(span)
    return (
        np.asarray(output),
        np.asarray(nearest_dt_ns, dtype=np.int64),
        np.asarray(bracket_span_ns, dtype=np.int64),
    )


def map_camera_timestamps_to_monotonic(
    camera_timestamp_ms,
    timestamp_domains,
    receive_monotonic_ns,
    receive_realtime_ns,
):
    camera_ms = np.asarray(camera_timestamp_ms, dtype=np.float64)
    receive_mono = np.asarray(receive_monotonic_ns, dtype=np.int64)
    receive_real = np.asarray(receive_realtime_ns, dtype=np.int64)
    domains = [str(value) for value in timestamp_domains]
    if camera_ms.ndim != 1 or camera_ms.size < 2:
        raise RuntimeError("camera clock mapping needs at least two frames")
    if len(domains) != camera_ms.size or receive_mono.shape != camera_ms.shape or receive_real.shape != camera_ms.shape:
        raise RuntimeError("camera clock arrays have inconsistent lengths")
    if len(set(domains)) != 1:
        raise RuntimeError("camera timestamp domain changed during capture: {}".format(sorted(set(domains))))
    if not np.isfinite(camera_ms).all() or np.any(np.diff(camera_ms) <= 0.0):
        raise RuntimeError("camera timestamps must be finite and strictly increasing")
    if np.any(np.diff(receive_mono) <= 0):
        raise RuntimeError("camera host receive timestamps must be strictly increasing")

    domain = domains[0].lower()
    camera_ns = np.rint(camera_ms * 1.0e6).astype(np.int64)
    host_offset_ns = int(np.median(receive_mono - receive_real))
    if "global_time" in domain or "system_time" in domain:
        target_ns = camera_ns + host_offset_ns
        receive_residual_ns = receive_mono - target_ns
        mapping = {
            "mode": "camera_realtime_plus_host_monotonic_offset",
            "domain": domains[0],
            "host_monotonic_minus_realtime_ns": host_offset_ns,
            "affine_slope": 1.0,
            "receive_residual_median_ns": int(np.median(receive_residual_ns)),
            "receive_residual_p95_ns": int(np.percentile(np.abs(receive_residual_ns), 95)),
        }
        if mapping["receive_residual_p95_ns"] > 100000000:
            raise RuntimeError("camera global timestamp differs from host receive time by more than 100 ms")
        return target_ns, mapping

    centered_camera = camera_ns.astype(np.float64) - float(camera_ns[0])
    centered_receive = receive_mono.astype(np.float64) - float(receive_mono[0])
    slope = float(np.dot(centered_camera, centered_receive) / np.dot(centered_camera, centered_camera))
    if not 0.99 <= slope <= 1.01:
        raise RuntimeError("camera-to-host clock slope is implausible: {:.9f}".format(slope))
    intercept = float(np.median(receive_mono.astype(np.float64) - slope * camera_ns.astype(np.float64)))
    target_ns = np.rint(slope * camera_ns.astype(np.float64) + intercept).astype(np.int64)
    residual_ns = receive_mono - target_ns
    mapping = {
        "mode": "camera_clock_to_host_receive_affine",
        "domain": domains[0],
        "host_monotonic_minus_realtime_ns": host_offset_ns,
        "affine_slope": slope,
        "affine_intercept_ns": intercept,
        "receive_residual_median_ns": int(np.median(residual_ns)),
        "receive_residual_p95_ns": int(np.percentile(np.abs(residual_ns), 95)),
        "warning": "constant camera transport latency remains in the fitted intercept",
    }
    return target_ns, mapping


def runtime_frame_schedule(timestamp_s, runtime_rate_hz=INTENDED_RUNTIME_RATE_HZ):
    """Select the latest captured frame at each fixed-rate runtime tick."""
    timestamps = np.asarray(timestamp_s, dtype=np.float64)
    rate_hz = float(runtime_rate_hz)
    if timestamps.ndim != 1 or timestamps.size < 2:
        raise RuntimeError("runtime frame scheduling needs at least two camera timestamps")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0.0):
        raise RuntimeError("runtime frame scheduling requires strictly increasing finite timestamps")
    if not np.isfinite(rate_hz) or rate_hz <= 0.0:
        raise ValueError("runtime_rate_hz must be positive and finite")
    duration_s = float(timestamps[-1] - timestamps[0])
    tick_count = int(math.floor(duration_s * rate_hz + 1.0e-9)) + 1
    tick_timestamps = timestamps[0] + np.arange(tick_count, dtype=np.float64) / rate_hz
    frame_indices = np.searchsorted(timestamps, tick_timestamps, side="right") - 1
    if np.any(frame_indices < 0) or np.any(frame_indices >= timestamps.size):
        raise RuntimeError("internal runtime frame schedule is outside the camera capture")
    camera_age_s = tick_timestamps - timestamps[frame_indices]
    if np.any(camera_age_s < -1.0e-9):
        raise RuntimeError("runtime schedule selected a camera frame from the future")
    return frame_indices.astype(np.int64), tick_timestamps, camera_age_s


class LockedSamples(object):
    def __init__(self):
        self._lock = threading.Lock()
        self._samples = []

    def append(self, sample):
        with self._lock:
            self._samples.append(sample)

    def snapshot(self):
        with self._lock:
            return list(self._samples)

    def count(self):
        with self._lock:
            return len(self._samples)


class UnitreeStateCollector(object):
    """Read only odometry, lowstate, and torso IMU through one DDS participant."""

    def __init__(self, network_interface, odom_topic):
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
            from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowState_
        except ImportError as error:
            raise RuntimeError(
                "unitree_sdk2py is required; add the onboard SDK source and site-packages to PYTHONPATH"
            ) from error
        self.odom_samples = LockedSamples()
        self.lowstate_samples = LockedSamples()
        self.torso_imu_samples = LockedSamples()
        ChannelFactoryInitialize(0, network_interface)
        dds_odom_topic = odom_topic if odom_topic.startswith("rt/") else "rt/" + odom_topic.lstrip("/")
        self.odom_subscriber = ChannelSubscriber(dds_odom_topic, Odometry_)
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.torso_imu_subscriber = ChannelSubscriber("rt/secondary_imu", IMUState_)
        self.odom_subscriber.Init(self._on_odom, 200)
        self.lowstate_subscriber.Init(self._on_lowstate, 200)
        self.torso_imu_subscriber.Init(self._on_torso_imu, 200)

    def _on_odom(self, message):
        host_monotonic_ns = time.monotonic_ns()
        host_realtime_ns = time.time_ns()
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        self.odom_samples.append(
            {
                "host_monotonic_ns": host_monotonic_ns,
                "host_realtime_ns": host_realtime_ns,
                "header_stamp_ns": int(message.header.stamp.sec) * 1000000000 + int(message.header.stamp.nanosec),
                "position": [position.x, position.y, position.z],
                "quaternion_xyzw": [orientation.x, orientation.y, orientation.z, orientation.w],
                "linear_velocity": [
                    message.twist.twist.linear.x,
                    message.twist.twist.linear.y,
                    message.twist.twist.linear.z,
                ],
                "angular_velocity": [
                    message.twist.twist.angular.x,
                    message.twist.twist.angular.y,
                    message.twist.twist.angular.z,
                ],
                "frame_id": str(message.header.frame_id),
                "child_frame_id": str(message.child_frame_id),
            }
        )

    @staticmethod
    def _imu_payload(message):
        return {
            "quaternion_wxyz": np.asarray(message.quaternion, dtype=np.float64).tolist(),
            "gyroscope": np.asarray(message.gyroscope, dtype=np.float64).tolist(),
            "accelerometer": np.asarray(message.accelerometer, dtype=np.float64).tolist(),
            "rpy": np.asarray(message.rpy, dtype=np.float64).tolist(),
            "temperature": int(message.temperature),
        }

    def _on_lowstate(self, message):
        host_monotonic_ns = time.monotonic_ns()
        host_realtime_ns = time.time_ns()
        motors = message.motor_state
        if len(motors) < G1_JOINT_COUNT:
            return
        self.lowstate_samples.append(
            {
                "host_monotonic_ns": host_monotonic_ns,
                "host_realtime_ns": host_realtime_ns,
                "tick": int(message.tick),
                "mode_pr": int(message.mode_pr),
                "mode_machine": int(message.mode_machine),
                "imu": self._imu_payload(message.imu_state),
                "joint_position": [float(motors[index].q) for index in range(G1_JOINT_COUNT)],
                "joint_velocity": [float(motors[index].dq) for index in range(G1_JOINT_COUNT)],
                "joint_tau_est": [float(motors[index].tau_est) for index in range(G1_JOINT_COUNT)],
            }
        )

    def _on_torso_imu(self, message):
        self.torso_imu_samples.append(
            {
                "host_monotonic_ns": time.monotonic_ns(),
                "host_realtime_ns": time.time_ns(),
                "imu": self._imu_payload(message),
            }
        )

    def close(self):
        self.odom_subscriber.Close()
        self.lowstate_subscriber.Close()
        self.torso_imu_subscriber.Close()


class RealSenseCapture(object):
    METADATA_NAMES = (
        "actual_exposure",
        "actual_fps",
        "backend_timestamp",
        "frame_counter",
        "frame_timestamp",
        "sensor_timestamp",
        "time_of_arrival",
        "gain_level",
        "frame_laser_power",
        "frame_emitter_mode",
        "input_width",
        "input_height",
    )
    SENSOR_OPTION_NAMES = (
        "emitter_enabled",
        "enable_auto_exposure",
        "exposure",
        "gain",
        "laser_power",
        "global_time_enabled",
        "visual_preset",
    )

    def __init__(self, width, height, fps, serial):
        try:
            import pyrealsense2 as rs
        except ImportError as error:
            raise RuntimeError("pyrealsense2 is required on the G1 host") from error
        self.rs = rs
        context = rs.context()
        candidates = []
        for device in context.query_devices():
            name = device.get_info(rs.camera_info.name)
            device_serial = device.get_info(rs.camera_info.serial_number)
            if "D435I" not in name.upper():
                continue
            if serial and device_serial != serial:
                continue
            candidates.append(device)
        if not candidates:
            raise RuntimeError("no matching D435i was found")
        if len(candidates) != 1:
            raise RuntimeError("multiple D435i devices found; pass --serial")
        device = candidates[0]
        self.serial = device.get_info(rs.camera_info.serial_number)
        self.device_name = device.get_info(rs.camera_info.name)
        self.firmware_version = device.get_info(rs.camera_info.firmware_version)
        self.usb_type = device.get_info(rs.camera_info.usb_type_descriptor)
        if not self.usb_type.startswith("3"):
            raise RuntimeError("D435i must use USB3; active link is {!r}".format(self.usb_type))

        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.pipeline = rs.pipeline()
        profile = self.pipeline.start(config)
        self.profile = profile
        self.sensor = profile.get_device().first_depth_sensor()
        stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        if stream.width() != width or stream.height() != height or stream.fps() != fps:
            self.close()
            raise RuntimeError("active D435i profile differs from the requested profile")
        intrinsics = stream.get_intrinsics()
        intrinsic_values = [intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy]
        if not np.isfinite(intrinsic_values).all() or intrinsics.fx <= 0.0 or intrinsics.fy <= 0.0:
            self.close()
            raise RuntimeError("D435i returned invalid intrinsics")
        self.width = int(stream.width())
        self.height = int(stream.height())
        self.fps = int(stream.fps())
        self.intrinsic_matrix = np.asarray(
            [[intrinsics.fx, 0.0, intrinsics.ppx], [0.0, intrinsics.fy, intrinsics.ppy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.distortion_model = str(intrinsics.model)
        self.distortion_coefficients = np.asarray(intrinsics.coeffs, dtype=np.float64)
        if not np.isfinite(self.distortion_coefficients).all() or np.max(
            np.abs(self.distortion_coefficients)
        ) > 1.0e-7:
            self.close()
            raise RuntimeError(
                "active depth stream is not rectified; the pinhole terrain runtime cannot ignore distortion"
            )
        self.depth_scale_m = float(self.sensor.get_depth_scale())
        if not np.isfinite(self.depth_scale_m) or self.depth_scale_m <= 0.0:
            self.close()
            raise RuntimeError("D435i returned invalid depth scale")
        self.sensor_options = {}
        for name in self.SENSOR_OPTION_NAMES:
            option = getattr(rs.option, name, None)
            if option is not None and self.sensor.supports(option):
                self.sensor_options[name] = float(self.sensor.get_option(option))

    def capture(self, duration_s, warmup_frames, stop_event):
        for _ in range(warmup_frames):
            if stop_event.is_set():
                break
            self.pipeline.wait_for_frames(5000)
        frames = []
        started_ns = time.monotonic_ns()
        while not stop_event.is_set():
            if frames and (time.monotonic_ns() - started_ns) * 1.0e-9 >= duration_s:
                break
            wait_start_ns = time.monotonic_ns()
            frame_set = self.pipeline.wait_for_frames(5000)
            frame = frame_set.get_depth_frame()
            receive_monotonic_ns = time.monotonic_ns()
            receive_realtime_ns = time.time_ns()
            if frame is None:
                raise RuntimeError("D435i returned no depth frame")
            depth = np.asanyarray(frame.get_data()).copy()
            if depth.dtype != np.uint16 or depth.shape != (self.height, self.width):
                raise RuntimeError("unexpected native depth shape or dtype: {} {}".format(depth.shape, depth.dtype))
            frame_number = int(frame.get_frame_number())
            if frames and frame_number <= frames[-1]["frame_number"]:
                raise RuntimeError("D435i frame number did not advance")
            metadata = []
            for name in self.METADATA_NAMES:
                key = getattr(self.rs.frame_metadata_value, name)
                metadata.append(int(frame.get_frame_metadata(key)) if frame.supports_frame_metadata(key) else -1)
            frames.append(
                {
                    "depth": depth,
                    "frame_number": frame_number,
                    "camera_timestamp_ms": float(frame.get_timestamp()),
                    "timestamp_domain": str(frame.get_frame_timestamp_domain()),
                    "wait_start_monotonic_ns": wait_start_ns,
                    "receive_monotonic_ns": receive_monotonic_ns,
                    "receive_realtime_ns": receive_realtime_ns,
                    "metadata": metadata,
                }
            )
        if len(frames) < 2:
            raise RuntimeError("capture ended before two D435i frames were received")
        return frames

    def describe(self):
        return {
            "device_name": self.device_name,
            "serial": self.serial,
            "firmware_version": self.firmware_version,
            "usb_type": self.usb_type,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "intrinsic_matrix": self.intrinsic_matrix.tolist(),
            "distortion_model": self.distortion_model,
            "distortion_coefficients": self.distortion_coefficients.tolist(),
            "depth_scale_m": self.depth_scale_m,
            "sensor_options": self.sensor_options,
        }

    def close(self):
        pipeline = getattr(self, "pipeline", None)
        if pipeline is not None:
            pipeline.stop()
            self.pipeline = None


def _wxyz_to_xyzw(quaternion):
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape[-1] != 4:
        raise ValueError("Unitree quaternion must have four values")
    return quaternion[..., [1, 2, 3, 0]]


def _validated_torso_from_optical(payload, source_path=None):
    if not isinstance(payload, dict):
        raise ValueError("torso-from-optical calibration must contain an object")
    transform_name = payload.get("transform_name", "T_torso_link_from_camera_optical")
    if transform_name != "T_torso_link_from_camera_optical":
        raise ValueError(
            "transform_name must be 'T_torso_link_from_camera_optical', got {!r}".format(
                transform_name
            )
        )
    translation_key = "torso_from_camera_optical_translation_m"
    quaternion_key = "torso_from_camera_optical_quaternion_xyzw"
    if translation_key in payload or quaternion_key in payload:
        missing = [name for name in (translation_key, quaternion_key) if name not in payload]
        if missing:
            raise ValueError("torso-from-optical JSON is missing {}".format(", ".join(missing)))
        position = np.asarray(payload[translation_key], dtype=np.float64)
        quaternion = np.asarray(payload[quaternion_key], dtype=np.float64)
    else:
        # Keep old bring-up JSONs replayable, but canonicalize their meaning.
        required = ("mount_pos_torso", "optical_quat_torso_xyzw")
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError("torso-from-optical JSON is missing {}".format(", ".join(missing)))
        position = np.asarray(payload["mount_pos_torso"], dtype=np.float64)
        quaternion = np.asarray(payload["optical_quat_torso_xyzw"], dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("T_torso_link_from_camera_optical translation must contain three finite meters")
    quaternion_norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(quaternion_norm) or abs(quaternion_norm - 1.0) > 1.0e-3:
        raise ValueError("T_torso_link_from_camera_optical xyzw quaternion must be unit length")
    quaternion = _normalized_quaternion_xyzw(
        quaternion,
        "T_torso_link_from_camera_optical xyzw quaternion",
    )
    result = dict(payload)
    result.update(
        {
            "transform_name": "T_torso_link_from_camera_optical",
            translation_key: position.tolist(),
            quaternion_key: quaternion.tolist(),
            # Compatibility fields consumed by RealSenseCalibration.
            "mount_pos_torso": position.tolist(),
            "optical_quat_torso_xyzw": quaternion.tolist(),
        }
    )
    if source_path is not None:
        result["source_path"] = str(Path(source_path).expanduser().resolve())
        result.setdefault("reference_kind", "measured_or_user_supplied")
    return result


def _load_extrinsics(path):
    payload = json.loads(Path(path).expanduser().read_text())
    return _validated_torso_from_optical(payload, source_path=path)


def _sample_arrays(samples, prefix, fields):
    payload = {}
    for output_name, sample_path, dtype in fields:
        values = []
        for sample in samples:
            value = sample
            for key in sample_path:
                value = value[key]
            values.append(value)
        payload[prefix + output_name] = np.asarray(values, dtype=dtype)
    return payload


def _odom_interpolation_times(samples):
    header_ns = np.asarray([sample["header_stamp_ns"] for sample in samples], dtype=np.int64)
    host_mono = np.asarray([sample["host_monotonic_ns"] for sample in samples], dtype=np.int64)
    host_real = np.asarray([sample["host_realtime_ns"] for sample in samples], dtype=np.int64)
    valid = header_ns > 0
    if valid.all() and np.max(np.abs(header_ns - host_real)) < 5000000000:
        offset_ns = int(np.median(host_mono - host_real))
        return header_ns + offset_ns, "ros_header_realtime_plus_host_monotonic_offset", offset_ns
    return host_mono, "host_monotonic_receive", None


def build_probe_report(frames, camera, odom_samples, lowstate_samples, torso_imu_samples):
    odom_samples = sorted(odom_samples, key=lambda sample: sample["host_monotonic_ns"])
    lowstate_samples = sorted(lowstate_samples, key=lambda sample: sample["host_monotonic_ns"])
    torso_imu_samples = sorted(torso_imu_samples, key=lambda sample: sample["host_monotonic_ns"])
    frame_ids = set(sample["frame_id"] for sample in odom_samples)
    child_frame_ids = set(sample["child_frame_id"] for sample in odom_samples)
    if frame_ids != {"odom"} or child_frame_ids != {"robot_center"}:
        raise RuntimeError(
            "expected odom -> robot_center, got {} -> {}".format(
                sorted(frame_ids), sorted(child_frame_ids)
            )
        )
    camera_target_ns, camera_clock_mapping = map_camera_timestamps_to_monotonic(
        [frame["camera_timestamp_ms"] for frame in frames],
        [frame["timestamp_domain"] for frame in frames],
        [frame["receive_monotonic_ns"] for frame in frames],
        [frame["receive_realtime_ns"] for frame in frames],
    )
    odom_times, odom_clock_mode, odom_clock_offset_ns = _odom_interpolation_times(odom_samples)
    streams = {
        "odom": (
            odom_times,
            np.asarray([sample["position"] for sample in odom_samples], dtype=np.float64),
            False,
        ),
        "lowstate": (
            np.asarray([sample["host_monotonic_ns"] for sample in lowstate_samples], dtype=np.int64),
            np.asarray([sample["joint_position"] for sample in lowstate_samples], dtype=np.float64),
            False,
        ),
        "torso_imu": (
            np.asarray([sample["host_monotonic_ns"] for sample in torso_imu_samples], dtype=np.int64),
            _wxyz_to_xyzw(
                np.asarray(
                    [sample["imu"]["quaternion_wxyz"] for sample in torso_imu_samples], dtype=np.float64
                )
            ),
            True,
        ),
    }
    synchronization = {}
    synchronized_values = {}
    for name, (times, values, quaternion) in streams.items():
        synchronized, nearest_dt_ns, bracket_span_ns = interpolate_series(
            times, values, camera_target_ns, quaternion=quaternion
        )
        synchronized_values[name] = synchronized
        synchronization[name] = {
            "nearest_abs_dt_p95_ms": float(np.percentile(np.abs(nearest_dt_ns), 95) * 1.0e-6),
            "nearest_abs_dt_max_ms": float(np.max(np.abs(nearest_dt_ns)) * 1.0e-6),
            "bracket_span_p95_ms": float(np.percentile(bracket_span_ns, 95) * 1.0e-6),
            "bracket_span_max_ms": float(np.max(bracket_span_ns) * 1.0e-6),
        }
    synchronized_robot_center_quaternion, _, _ = interpolate_series(
        odom_times,
        np.asarray([sample["quaternion_xyzw"] for sample in odom_samples], dtype=np.float64),
        camera_target_ns,
        quaternion=True,
    )
    synchronized_lowstate_imu, _, _ = interpolate_series(
        streams["lowstate"][0],
        _wxyz_to_xyzw(
            np.asarray(
                [sample["imu"]["quaternion_wxyz"] for sample in lowstate_samples], dtype=np.float64
            )
        ),
        camera_target_ns,
        quaternion=True,
    )
    fk_torso_disagreement_deg = []
    lowstate_imu_odom_disagreement_deg = []
    for robot_center_quaternion, lowstate_imu, joint_position, torso_imu in zip(
        synchronized_robot_center_quaternion,
        synchronized_lowstate_imu,
        synchronized_values["lowstate"],
        synchronized_values["torso_imu"],
    ):
        _, torso_fk_quaternion = torso_pose_from_pelvis(
            np.zeros(3),
            robot_center_quaternion,
            joint_position[[WAIST_YAW_INDEX, WAIST_ROLL_INDEX, WAIST_PITCH_INDEX]],
        )
        fk_torso_disagreement_deg.append(quaternion_distance_degrees(torso_fk_quaternion, torso_imu))
        lowstate_imu_odom_disagreement_deg.append(
            quaternion_distance_degrees(robot_center_quaternion, lowstate_imu)
        )

    def angle_summary(values):
        values = np.asarray(values, dtype=np.float64)
        return {
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }

    return {
        "mode": "read_only_probe",
        "camera": camera.describe(),
        "camera_clock_mapping": camera_clock_mapping,
        "odom_clock_mode": odom_clock_mode,
        "odom_clock_offset_ns": odom_clock_offset_ns,
        "odom_frame_ids": sorted(set(sample["frame_id"] for sample in odom_samples)),
        "odom_child_frame_ids": sorted(set(sample["child_frame_id"] for sample in odom_samples)),
        "frames": len(frames),
        "odom_samples": len(odom_samples),
        "lowstate_samples": len(lowstate_samples),
        "torso_imu_samples": len(torso_imu_samples),
        "synchronization": synchronization,
        "orientation_cross_checks_deg": {
            "nominal_torso_fk_from_robot_center_vs_secondary_imu": angle_summary(
                fk_torso_disagreement_deg
            ),
            "robot_center_odom_vs_lowstate_imu": angle_summary(
                lowstate_imu_odom_disagreement_deg
            ),
        },
        "motor_commands": "disabled_by_design",
    }


def build_capture_payload(
    frames,
    camera,
    odom_samples,
    lowstate_samples,
    torso_imu_samples,
    extrinsics,
    scene,
    max_sync_gap_ms,
):
    odom_samples = sorted(odom_samples, key=lambda sample: sample["host_monotonic_ns"])
    lowstate_samples = sorted(lowstate_samples, key=lambda sample: sample["host_monotonic_ns"])
    torso_imu_samples = sorted(torso_imu_samples, key=lambda sample: sample["host_monotonic_ns"])
    frame_ids = set(sample["frame_id"] for sample in odom_samples)
    child_frame_ids = set(sample["child_frame_id"] for sample in odom_samples)
    if frame_ids != {"odom"} or child_frame_ids != {"robot_center"}:
        raise RuntimeError(
            "expected odom -> robot_center, got {} -> {}".format(
                sorted(frame_ids), sorted(child_frame_ids)
            )
        )
    camera_timestamp_ms = np.asarray([frame["camera_timestamp_ms"] for frame in frames], dtype=np.float64)
    camera_domains = [frame["timestamp_domain"] for frame in frames]
    camera_receive_mono = np.asarray([frame["receive_monotonic_ns"] for frame in frames], dtype=np.int64)
    camera_receive_real = np.asarray([frame["receive_realtime_ns"] for frame in frames], dtype=np.int64)
    camera_target_ns, camera_clock_mapping = map_camera_timestamps_to_monotonic(
        camera_timestamp_ms,
        camera_domains,
        camera_receive_mono,
        camera_receive_real,
    )
    raw_timestamp_s = (camera_target_ns - camera_target_ns[0]).astype(np.float64) * 1.0e-9
    runtime_frame_index, runtime_tick_timestamp_s, runtime_camera_age_s = runtime_frame_schedule(
        raw_timestamp_s
    )

    odom_times, odom_clock_mode, odom_clock_offset_ns = _odom_interpolation_times(odom_samples)
    odom_positions = np.asarray([sample["position"] for sample in odom_samples], dtype=np.float64)
    odom_quaternions = np.asarray([sample["quaternion_xyzw"] for sample in odom_samples], dtype=np.float64)
    lowstate_times = np.asarray([sample["host_monotonic_ns"] for sample in lowstate_samples], dtype=np.int64)
    joint_positions = np.asarray([sample["joint_position"] for sample in lowstate_samples], dtype=np.float64)
    torso_imu_times = np.asarray([sample["host_monotonic_ns"] for sample in torso_imu_samples], dtype=np.int64)
    torso_imu_quaternions = _wxyz_to_xyzw(
        np.asarray([sample["imu"]["quaternion_wxyz"] for sample in torso_imu_samples], dtype=np.float64)
    )

    robot_center_positions, odom_position_dt, odom_position_span = interpolate_series(
        odom_times, odom_positions, camera_target_ns
    )
    robot_center_quaternions, odom_quaternion_dt, odom_quaternion_span = interpolate_series(
        odom_times, odom_quaternions, camera_target_ns, quaternion=True
    )
    synchronized_joint_positions, lowstate_dt, lowstate_span = interpolate_series(
        lowstate_times, joint_positions, camera_target_ns
    )
    synchronized_torso_imu, torso_imu_dt, torso_imu_span = interpolate_series(
        torso_imu_times, torso_imu_quaternions, camera_target_ns, quaternion=True
    )

    if not np.array_equal(odom_position_dt, odom_quaternion_dt) or not np.array_equal(
        odom_position_span, odom_quaternion_span
    ):
        raise RuntimeError("internal odometry interpolation mismatch")
    max_gap_ns = int(round(float(max_sync_gap_ms) * 1.0e6))
    sync_checks = {
        "odom": (odom_position_dt, odom_position_span),
        "lowstate": (lowstate_dt, lowstate_span),
        "torso_imu": (torso_imu_dt, torso_imu_span),
    }
    for name, (nearest_dt, bracket_span) in sync_checks.items():
        if np.max(np.abs(nearest_dt)) > max_gap_ns or np.max(bracket_span) > 2 * max_gap_ns:
            raise RuntimeError(
                "{} cannot bracket every camera exposure within {:.3f} ms".format(name, max_sync_gap_ms)
            )

    waist_angles = synchronized_joint_positions[
        :, [WAIST_YAW_INDEX, WAIST_ROLL_INDEX, WAIST_PITCH_INDEX]
    ]
    torso_positions = []
    torso_quaternions = []
    heading_quaternions = []
    torso_imu_disagreement_deg = []
    for robot_center_position, robot_center_quaternion, waist, torso_imu_quaternion in zip(
        robot_center_positions, robot_center_quaternions, waist_angles, synchronized_torso_imu
    ):
        # This nominal construction makes the current replay API usable without
        # claiming that Unitree's undocumented robot_center origin is pelvis.
        torso_position, torso_quaternion = torso_pose_from_pelvis(
            robot_center_position, robot_center_quaternion, waist
        )
        torso_positions.append(torso_position)
        torso_quaternions.append(torso_quaternion)
        heading_quaternions.append(heading_quaternion_xyzw(robot_center_quaternion))
        torso_imu_disagreement_deg.append(
            quaternion_distance_degrees(torso_quaternion, torso_imu_quaternion)
        )

    runtime_calibration = {
        "native_width": camera.width,
        "native_height": camera.height,
        "target_width": 64,
        "target_height": 36,
        "intrinsic_matrix": camera.intrinsic_matrix.reshape(-1).tolist(),
        "depth_scale_m": camera.depth_scale_m,
        "mount_pos_torso": extrinsics["mount_pos_torso"],
        "optical_quat_torso_xyzw": extrinsics["optical_quat_torso_xyzw"],
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "scene": scene,
        "created_realtime_ns": time.time_ns(),
        "camera": camera.describe(),
        "camera_clock_mapping": camera_clock_mapping,
        "odom_clock_mode": odom_clock_mode,
        "odom_clock_offset_ns": odom_clock_offset_ns,
        "pose_contract": (
            "world/odom positions and xyzw quaternions; /dog_odom robot_center is retained as "
            "an egomotion proxy, not asserted equal to pelvis; current pelvis_* replay arrays are "
            "compatibility aliases and torso pose is a nominal G1 waist yaw-roll-pitch FK"
        ),
        "odom_frame_contract": {
            "parent_frame": "odom",
            "child_frame": "robot_center",
            "usage": "relative egomotion proxy",
            "robot_center_equals_model_pelvis": None,
            "straight_translation_validation": "allowed",
            "yaw_with_unknown_lever_arm": "diagnostic_only",
        },
        "replay_pose_aliases": {
            "pelvis_pos_w": "egomotion_pos_w copied from odom->robot_center",
            "pelvis_quat_w_xyzw": "egomotion_quat_w_xyzw copied from odom->robot_center",
            "pelvis_heading_quat_w_xyzw": "heading of odom->robot_center",
        },
        "extrinsics": extrinsics,
        "intended_runtime_cadence_hz": INTENDED_RUNTIME_RATE_HZ,
        "capture_cadence_note": "all native camera frames are retained; replay uses timestamp-driven latest-frame indices",
        "runtime_frame_schedule": {
            "selection": "latest camera exposure at or before each 50 Hz runtime tick",
            "source_frames": len(frames),
            "runtime_ticks": int(runtime_frame_index.size),
            "unique_camera_frames": int(np.unique(runtime_frame_index).size),
            "camera_age_p95_ms": float(np.percentile(runtime_camera_age_s, 95) * 1.0e3),
            "camera_age_max_ms": float(np.max(runtime_camera_age_s) * 1.0e3),
        },
        "max_sync_gap_ms": float(max_sync_gap_ms),
        "runtime_calibration": runtime_calibration,
        "raw_stream_counts": {
            "depth": len(frames),
            "odom": len(odom_samples),
            "lowstate": len(lowstate_samples),
            "torso_imu": len(torso_imu_samples),
        },
        "torso_imu_fk_disagreement_deg": {
            "median": float(np.median(torso_imu_disagreement_deg)),
            "p95": float(np.percentile(torso_imu_disagreement_deg, 95)),
            "max": float(np.max(torso_imu_disagreement_deg)),
        },
    }

    payload = {
        "schema_version": np.asarray(SCHEMA_VERSION),
        "capture_metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "runtime_calibration_json": np.asarray(json.dumps(runtime_calibration, sort_keys=True)),
        "depth": np.stack([frame["depth"] for frame in frames]).astype(np.uint16, copy=False),
        "intrinsic_matrix": camera.intrinsic_matrix,
        "distortion_coefficients": camera.distortion_coefficients,
        "distortion_model": np.asarray(camera.distortion_model),
        "depth_scale_m": np.asarray(camera.depth_scale_m, dtype=np.float64),
        "frame_number": np.asarray([frame["frame_number"] for frame in frames], dtype=np.uint64),
        "camera_timestamp_ms": camera_timestamp_ms,
        "camera_timestamp_domain": np.asarray(camera_domains, dtype="U64"),
        "camera_wait_start_monotonic_ns": np.asarray(
            [frame["wait_start_monotonic_ns"] for frame in frames], dtype=np.int64
        ),
        "camera_host_monotonic_receive_ns": camera_receive_mono,
        "camera_host_realtime_receive_ns": camera_receive_real,
        "camera_target_monotonic_ns": camera_target_ns,
        "camera_metadata_names": np.asarray(RealSenseCapture.METADATA_NAMES, dtype="U64"),
        "camera_metadata": np.asarray([frame["metadata"] for frame in frames], dtype=np.int64),
        "timestamp_s": raw_timestamp_s,
        "reset_mask": np.arange(len(frames)) == 0,
        "runtime_frame_index": runtime_frame_index,
        "runtime_tick_timestamp_s": runtime_tick_timestamp_s,
        "runtime_camera_age_s": runtime_camera_age_s,
        "egomotion_pos_w": robot_center_positions.astype(np.float32),
        "egomotion_quat_w_xyzw": robot_center_quaternions.astype(np.float32),
        "egomotion_heading_quat_w_xyzw": np.asarray(heading_quaternions, dtype=np.float32),
        # Compatibility aliases for the existing RealSense replay boundary.
        "pelvis_pos_w": robot_center_positions.astype(np.float32),
        "pelvis_quat_w_xyzw": robot_center_quaternions.astype(np.float32),
        "pelvis_heading_quat_w_xyzw": np.asarray(heading_quaternions, dtype=np.float32),
        "torso_pos_w": np.asarray(torso_positions, dtype=np.float32),
        "torso_quat_w_xyzw": np.asarray(torso_quaternions, dtype=np.float32),
        "synchronized_joint_position": synchronized_joint_positions.astype(np.float32),
        "synchronized_torso_imu_quat_w_xyzw": synchronized_torso_imu.astype(np.float32),
        "torso_imu_fk_disagreement_deg": np.asarray(torso_imu_disagreement_deg, dtype=np.float32),
        "odom_sync_dt_s": odom_position_dt.astype(np.float64) * 1.0e-9,
        "odom_bracket_span_s": odom_position_span.astype(np.float64) * 1.0e-9,
        "lowstate_sync_dt_s": lowstate_dt.astype(np.float64) * 1.0e-9,
        "lowstate_bracket_span_s": lowstate_span.astype(np.float64) * 1.0e-9,
        "torso_imu_sync_dt_s": torso_imu_dt.astype(np.float64) * 1.0e-9,
        "torso_imu_bracket_span_s": torso_imu_span.astype(np.float64) * 1.0e-9,
    }
    payload.update(
        _sample_arrays(
            odom_samples,
            "raw_odom_",
            (
                ("host_monotonic_ns", ("host_monotonic_ns",), np.int64),
                ("host_realtime_ns", ("host_realtime_ns",), np.int64),
                ("header_stamp_ns", ("header_stamp_ns",), np.int64),
                ("position", ("position",), np.float64),
                ("quaternion_xyzw", ("quaternion_xyzw",), np.float64),
                ("linear_velocity", ("linear_velocity",), np.float64),
                ("angular_velocity", ("angular_velocity",), np.float64),
                ("frame_id", ("frame_id",), "U64"),
                ("child_frame_id", ("child_frame_id",), "U64"),
            ),
        )
    )
    payload.update(
        _sample_arrays(
            lowstate_samples,
            "raw_lowstate_",
            (
                ("host_monotonic_ns", ("host_monotonic_ns",), np.int64),
                ("host_realtime_ns", ("host_realtime_ns",), np.int64),
                ("tick", ("tick",), np.uint32),
                ("mode_pr", ("mode_pr",), np.uint8),
                ("mode_machine", ("mode_machine",), np.uint8),
                ("pelvis_imu_quaternion_wxyz", ("imu", "quaternion_wxyz"), np.float64),
                ("pelvis_imu_gyroscope", ("imu", "gyroscope"), np.float64),
                ("pelvis_imu_accelerometer", ("imu", "accelerometer"), np.float64),
                ("pelvis_imu_rpy", ("imu", "rpy"), np.float64),
                ("joint_position", ("joint_position",), np.float32),
                ("joint_velocity", ("joint_velocity",), np.float32),
                ("joint_tau_est", ("joint_tau_est",), np.float32),
            ),
        )
    )
    payload.update(
        _sample_arrays(
            torso_imu_samples,
            "raw_torso_imu_",
            (
                ("host_monotonic_ns", ("host_monotonic_ns",), np.int64),
                ("host_realtime_ns", ("host_realtime_ns",), np.int64),
                ("quaternion_wxyz", ("imu", "quaternion_wxyz"), np.float64),
                ("gyroscope", ("imu", "gyroscope"), np.float64),
                ("accelerometer", ("imu", "accelerometer"), np.float64),
                ("rpy", ("imu", "rpy"), np.float64),
            ),
        )
    )
    return payload, metadata, runtime_calibration


def _write_json_atomic(path, payload):
    path = Path(path)
    handle = tempfile.NamedTemporaryFile(mode="w", prefix=path.name + ".", suffix=".tmp", dir=str(path.parent), delete=False)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(handle.name, str(path))
    except Exception:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise


def write_capture(output_path, payload, metadata, runtime_calibration, overwrite):
    output = Path(output_path).expanduser()
    if output.suffix != ".npz":
        raise ValueError("--output must end in .npz")
    calibration_path = output.with_suffix(".runtime_calibration.json")
    summary_path = output.with_suffix(".summary.json")
    targets = (output, calibration_path, summary_path)
    if not overwrite:
        existing = [str(path) for path in targets if path.exists()]
        if existing:
            raise FileExistsError("refusing to overwrite {}".format(", ".join(existing)))
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(prefix=output.name + ".", suffix=".tmp.npz", dir=str(output.parent), delete=False)
    handle.close()
    try:
        np.savez_compressed(handle.name, **payload)
        os.replace(handle.name, str(output))
    except Exception:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise
    summary = dict(metadata)
    summary["output_npz"] = str(output.resolve())
    summary["runtime_calibration_json"] = str(calibration_path.resolve())
    _write_json_atomic(calibration_path, runtime_calibration)
    _write_json_atomic(summary_path, summary)
    return output, calibration_path, summary_path


def _wait_for_sources(unitree_state, timeout_s):
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        counts = (
            unitree_state.odom_samples.count(),
            unitree_state.lowstate_samples.count(),
            unitree_state.torso_imu_samples.count(),
        )
        if min(counts) >= 2:
            return counts
        time.sleep(0.02)
    raise RuntimeError(
        "state source timeout: odom={}, lowstate={}, torso_imu={}".format(
            unitree_state.odom_samples.count(),
            unitree_state.lowstate_samples.count(),
            unitree_state.torso_imu_samples.count(),
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Raw synchronized NPZ (required unless --probe)")
    extrinsics = parser.add_mutually_exclusive_group()
    extrinsics.add_argument(
        "--torso-from-optical-json",
        "--extrinsics-json",
        dest="torso_from_optical_json",
        type=Path,
        help="Measured T_torso_link_from_camera_optical JSON",
    )
    extrinsics.add_argument(
        "--use-instinct-nominal-torso-from-optical",
        action="store_true",
        help="Use the pinned Project Instinct G1 nominal mount with explicit optical-axis composition",
    )
    parser.add_argument("--scene", default="unspecified", help="flat, stair_10cm, stair_14cm, stair_18cm, or sync_motion")
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--postroll-ms", type=float, default=200.0)
    parser.add_argument("--max-sync-gap-ms", type=float, default=50.0)
    parser.add_argument("--serial", default=None)
    parser.add_argument("--width", type=int, default=DEFAULT_DEPTH_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_DEPTH_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_DEPTH_FPS)
    parser.add_argument("--network-interface", default="eth0")
    parser.add_argument("--odom-topic", default="/dog_odom")
    parser.add_argument(
        "--accept-robot-center-egomotion-proxy",
        action="store_true",
        help=(
            "Use odom->robot_center as a relative-motion proxy without claiming it equals model pelvis; "
            "yaw motion remains diagnostic until the fixed frame transform is known"
        ),
    )
    parser.add_argument("--probe", action="store_true", help="Validate read-only sources without writing a capture")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.duration_s <= 0.0 or args.warmup_frames < 0 or args.postroll_ms < 0.0 or args.max_sync_gap_ms <= 0.0:
        raise ValueError("capture durations and sync gap must be valid and positive")
    if min(args.width, args.height, args.fps) <= 0:
        raise ValueError("camera profile values must be positive")
    if not args.probe:
        if args.output is None:
            raise ValueError("--output is required for a capture")
        if args.torso_from_optical_json is None and not args.use_instinct_nominal_torso_from_optical:
            raise ValueError(
                "a capture requires --torso-from-optical-json or "
                "--use-instinct-nominal-torso-from-optical"
            )
        if not args.accept_robot_center_egomotion_proxy:
            raise ValueError(
                "pass --accept-robot-center-egomotion-proxy to record /dog_odom without asserting "
                "robot_center == pelvis"
            )
        if args.use_instinct_nominal_torso_from_optical:
            extrinsics = instinct_nominal_torso_from_optical()
        else:
            extrinsics = _load_extrinsics(args.torso_from_optical_json)
    else:
        extrinsics = None

    stop_event = threading.Event()

    def stop_handler(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    unitree_state = None
    camera = None
    try:
        unitree_state = UnitreeStateCollector(args.network_interface, args.odom_topic)
        _wait_for_sources(unitree_state, 8.0)
        camera = RealSenseCapture(args.width, args.height, args.fps, args.serial)
        frames = camera.capture(args.duration_s, args.warmup_frames, stop_event)
        camera.close()
        time.sleep(args.postroll_ms * 1.0e-3)
        odom_samples = unitree_state.odom_samples.snapshot()
        lowstate_samples = unitree_state.lowstate_samples.snapshot()
        torso_imu_samples = unitree_state.torso_imu_samples.snapshot()
        if args.probe:
            report = build_probe_report(
                frames,
                camera,
                odom_samples,
                lowstate_samples,
                torso_imu_samples,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        payload, metadata, runtime_calibration = build_capture_payload(
            frames,
            camera,
            odom_samples,
            lowstate_samples,
            torso_imu_samples,
            extrinsics,
            args.scene,
            args.max_sync_gap_ms,
        )
        paths = write_capture(args.output, payload, metadata, runtime_calibration, args.overwrite)
        print(
            json.dumps(
                {
                    "output_npz": str(paths[0]),
                    "runtime_calibration_json": str(paths[1]),
                    "summary_json": str(paths[2]),
                    "frames": len(frames),
                    "motor_commands": "disabled_by_design",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    finally:
        if camera is not None:
            camera.close()
        if unitree_state is not None:
            unitree_state.close()


if __name__ == "__main__":
    raise SystemExit(main())
