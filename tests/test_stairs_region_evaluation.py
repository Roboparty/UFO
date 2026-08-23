from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from humanoidverse.terrain_transfer_inference import (
    _max_consecutive_stair_transitions,
    _separated_stairs_progress_metrics,
    _stairs_pre_ascent_offset,
    _stairs_progress_metrics,
)


_WATCHER_PATH = Path(__file__).parents[1] / "scripts" / "watch_stairs_region_milestones.py"
_WATCHER_SPEC = importlib.util.spec_from_file_location("watch_stairs_region_milestones", _WATCHER_PATH)
assert _WATCHER_SPEC is not None and _WATCHER_SPEC.loader is not None
_WATCHER = importlib.util.module_from_spec(_WATCHER_SPEC)
_WATCHER_SPEC.loader.exec_module(_WATCHER)
MILESTONES = _WATCHER.MILESTONES
create_snapshot = _WATCHER.create_snapshot
preserve_virtualenv_executable = _WATCHER.preserve_virtualenv_executable


def test_pre_ascent_reset_stays_inside_center_platform() -> None:
    assert _stairs_pre_ascent_offset(platform_width=1.0, edge_margin=0.35) == pytest.approx(0.15)


def test_stairs_progress_distinguishes_ascent_and_descent() -> None:
    ascent = _stairs_progress_metrics(
        ground_heights=[0.0, 0.1, 0.2, 0.4, 0.6],
        ground_clearances=[0.8] * 5,
        body_impacts=[0.1] * 4,
        step_height=0.1,
        num_steps=6,
        fall_clearance=0.45,
        min_descent_steps=3,
        max_allowed_body_impact=1.0,
    )
    assert ascent["ascent_initiated"]
    assert ascent["high_platform_reached"]
    assert ascent["ascent_success"]
    assert not ascent["descent_initiated"]

    descent = _stairs_progress_metrics(
        ground_heights=[0.6, 0.5, 0.4, 0.3],
        ground_clearances=[0.8, 0.75, 0.7, 0.7],
        body_impacts=[0.2, 0.3, 0.2],
        step_height=0.1,
        num_steps=6,
        fall_clearance=0.45,
        min_descent_steps=3,
        max_allowed_body_impact=1.0,
    )
    assert descent["descent_initiated"]
    assert descent["descending_steps_completed"] == 3
    assert descent["descent_success"]
    assert not descent["low_platform_reached"]


@pytest.mark.parametrize(
    ("impacts", "clearances"),
    [([2.0], [0.8, 0.8]), ([0.1], [0.8, 0.2])],
)
def test_descent_success_rejects_falls(
    impacts: list[float], clearances: list[float]
) -> None:
    result = _stairs_progress_metrics(
        ground_heights=[0.6, 0.2],
        ground_clearances=clearances,
        body_impacts=impacts,
        step_height=0.1,
        num_steps=6,
        fall_clearance=0.45,
        min_descent_steps=3,
        max_allowed_body_impact=1.0,
    )
    assert not result["descent_success"]


def test_separated_stairs_metrics_measure_center_to_outer_transition() -> None:
    result = _separated_stairs_progress_metrics(
        terrain="stairs_up",
        ground_heights=[-1.0, -1.0, -0.9, -0.8, -0.7],
        ground_clearances=[0.8] * 5,
        body_impacts=[0.1] * 4,
        planar_radii=[0.0, 0.3, 0.5, 0.8, 1.1],
        cumulative_planar_path=1.1,
        step_height=0.1,
        num_steps=3,
        center_width=0.8,
        fall_clearance=0.45,
        max_allowed_body_impact=1.0,
    )
    assert result["center_departed"]
    assert result["center_departure_step"] == 2
    assert result["first_transition"]
    assert result["first_transition_step"] == 2
    assert result["consecutive_steps_completed"] == 3
    assert result["outer_ground_reached"]
    assert not result["stalled_at_center"]


def test_separated_stairs_metrics_identify_center_loop() -> None:
    result = _separated_stairs_progress_metrics(
        terrain="stairs_down",
        ground_heights=[1.0] * 5,
        ground_clearances=[0.8] * 5,
        body_impacts=[0.0] * 4,
        planar_radii=[0.0, 0.2, 0.3, 0.2, 0.3],
        cumulative_planar_path=1.7,
        step_height=0.1,
        num_steps=10,
        center_width=0.8,
        fall_clearance=0.45,
        max_allowed_body_impact=1.0,
    )
    assert not result["center_departed"]
    assert result["stalled_at_center"]
    assert result["center_looped"]


def test_consecutive_stair_transitions_reject_skipped_levels() -> None:
    assert _max_consecutive_stair_transitions([0, 0, 1, 1, 2, 3]) == 3
    assert _max_consecutive_stair_transitions([0, 2, 3]) == 1


def test_milestones_use_real_checkpoint_steps_and_never_fake_final() -> None:
    assert MILESTONES == (
        ("20M", 19_202_048),
        ("40M", 38_404_096),
        ("80M", 76_808_192),
        ("120M", 115_212_288),
        ("192M", 192_020_480),
    )


def test_watcher_preserves_virtualenv_python_symlink(tmp_path: Path) -> None:
    system_python = tmp_path / "python3.10"
    system_python.write_text("")
    virtualenv_python = tmp_path / ".venv" / "bin" / "python"
    virtualenv_python.parent.mkdir(parents=True)
    virtualenv_python.symlink_to(system_python)

    assert preserve_virtualenv_executable(virtualenv_python) == virtualenv_python


def test_watcher_uses_short_repo_local_tmpdir() -> None:
    source = _WATCHER_PATH.read_text()
    assert 'tmp_dir = args.repo / "cache" / "milestone_eval_tmp"' in source
    assert '"TMPDIR": str(tmp_dir)' in source
    assert '"TMPDIR": str(milestone_dir / "tmp")' not in source


def test_evaluation_snapshot_excludes_replay_and_optimizer(tmp_path) -> None:
    work_dir = tmp_path / "run"
    checkpoint = work_dir / "checkpoint"
    model = checkpoint / "model"
    model.mkdir(parents=True)
    (work_dir / "config.json").write_text("{}")
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "init_kwargs.json").write_text("{}")
    (checkpoint / "train_status.json").write_text('{"global_time": 19202048}')
    (checkpoint / "optimizers.pth").write_bytes(b"optimizer")
    (checkpoint / "buffers").mkdir()
    (checkpoint / "buffers" / "buffer.bin").write_bytes(b"replay")
    (model / "config.json").write_text("{}")
    (model / "init_kwargs.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"weights")

    output = work_dir / "milestone_evaluations" / "20M"
    create_snapshot(work_dir, output, 19_202_048)

    assert (output / "checkpoint" / "model" / "model.safetensors").read_bytes() == b"weights"
    assert not (output / "checkpoint" / "optimizers.pth").exists()
    assert not (output / "checkpoint" / "buffers").exists()
