from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.realtime.realtime_z_server import (  # noqa: E402
    OnlineZInferer,
    _extract_latest_frame,
    _is_pose_stale,
)


def _valid_payload():
    return {
        "frames": [
            {
                "root_pos": [0.0, 0.0, 0.8],
                "root_quat": [1.0, 0.0, 0.0, 0.0],
                "dof_pos": [0.0] * 29,
            }
        ]
    }


def test_pose_stale_check_blocks_publish_path():
    assert _is_pose_stale(last_valid_pose_monotonic=1.0, max_pose_stale_s=0.2, now=1.21)
    assert not _is_pose_stale(last_valid_pose_monotonic=1.0, max_pose_stale_s=0.2, now=1.19)
    assert not _is_pose_stale(last_valid_pose_monotonic=1.0, max_pose_stale_s=None, now=100.0)


def test_extract_latest_frame_rejects_nonfinite_pose_values():
    payload = _valid_payload()
    payload["frames"][0]["root_pos"][0] = float("nan")
    assert _extract_latest_frame(payload) is None

    payload = _valid_payload()
    payload["frames"][0]["root_quat"][1] = float("inf")
    assert _extract_latest_frame(payload) is None

    payload = _valid_payload()
    payload["frames"][0]["dof_pos"][3] = float("-inf")
    assert _extract_latest_frame(payload) is None


def test_extract_latest_frame_accepts_and_normalizes_valid_pose():
    frame = _extract_latest_frame(_valid_payload())
    assert frame is not None
    np.testing.assert_allclose(frame.root_quat_wxyz, np.array([1.0, 0.0, 0.0, 0.0]))
    assert frame.dof_pos.shape == (29,)


def test_online_inferer_rejects_invalid_z_without_touching_last_z():
    inferer = OnlineZInferer.__new__(OnlineZInferer)
    inferer._last_invalid_z_warning = 0.0
    inferer.last_z = np.ones(256, dtype=np.float32)

    assert inferer._validate_z_output(np.zeros(255, dtype=np.float32)) is None
    np.testing.assert_array_equal(inferer.last_z, np.ones(256, dtype=np.float32))

    assert inferer._validate_z_output(np.full(256, np.nan, dtype=np.float32)) is None
    np.testing.assert_array_equal(inferer.last_z, np.ones(256, dtype=np.float32))


if __name__ == "__main__":
    test_pose_stale_check_blocks_publish_path()
    test_extract_latest_frame_rejects_nonfinite_pose_values()
    test_extract_latest_frame_accepts_and_normalizes_valid_pose()
    test_online_inferer_rejects_invalid_z_without_touching_last_z()
    print("realtime_z_server safety tests ok")
