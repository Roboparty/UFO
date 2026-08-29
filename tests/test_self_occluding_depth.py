from __future__ import annotations

from types import SimpleNamespace

import torch

from humanoidverse.perception.depth_augmentation import (
    MetricDepthAugmentation,
    MetricDepthAugmentationConfig,
)
from humanoidverse.perception.depth_camera import DepthCameraConfig
from humanoidverse.perception.depth_preprocessing import resize_depth_with_conservative_invalid_mask
from humanoidverse.perception.self_occluding_depth import (
    SelfOcclusionDepthConfig,
    classify_synchronized_first_hits,
    dilate_self_mask,
    make_self_occlusion_camera_pair,
    self_occluding_depth_from_sensors,
)


def _sensor(distances: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(data=SimpleNamespace(distances=distances.reshape(distances.shape[0], -1)))


def test_camera_pair_is_coregistered_and_scene_keeps_robot_groups() -> None:
    camera = DepthCameraConfig(name="camera", width=5, height=3, max_range=9.0)
    terrain, scene = make_self_occlusion_camera_pair(camera, SelfOcclusionDepthConfig())

    assert terrain.name != scene.name
    assert terrain.max_range == scene.max_range == 2.0
    assert terrain.min_range == scene.min_range == 0.1
    assert terrain.include_geom_groups == (5,)
    assert scene.include_geom_groups == (2, 3, 5)
    assert terrain.intrinsics().equal(scene.intrinsics())
    assert terrain.mount_pos_torso == scene.mount_pos_torso


def test_self_occlusion_config_restores_only_semantic_metadata_fields() -> None:
    config = SelfOcclusionDepthConfig.from_metadata(
        {
            "min_ray_range_m": 0.1,
            "max_ray_range_m": 2.0,
            "hit_tolerance_m": 0.003,
            "dilation_sigma_multiplier": 2.5,
            "terrain_geom_groups": [5],
            "scene_geom_groups": [2, 3, 5],
            "camera_housing_geom_names": ["head_collision"],
            "camera_housing_mesh_names": ["head_link"],
            "camera_housing_geom_group": 4,
            "scene_camera": {"ignored": True},
            "max_dilation_radius_px": 8,
        }
    )

    assert config.hit_tolerance_m == 0.003
    assert config.dilation_sigma_multiplier == 2.5
    assert config.scene_geom_groups == (2, 3, 5)


def test_first_hit_classification_uses_tolerance_and_handles_missing_terrain() -> None:
    terrain = torch.tensor([[[1.0, 1.0, -1.0, -1.0, 1.0]]])
    scene = torch.tensor([[[1.001, 0.7, 0.6, -1.0, 1.01]]])
    terrain_mask, self_mask, far_mask, ambiguous = classify_synchronized_first_hits(
        terrain,
        scene,
        SelfOcclusionDepthConfig(hit_tolerance_m=0.002),
    )

    assert terrain_mask.tolist() == [[[True, False, False, False, False]]]
    assert self_mask.tolist() == [[[False, True, True, False, False]]]
    assert far_mask.tolist() == [[[False, False, False, True, False]]]
    assert ambiguous.tolist() == [[[False, False, False, False, True]]]


def test_dilation_radius_tracks_each_frames_sigma() -> None:
    mask = torch.zeros(2, 9, 9, dtype=torch.bool)
    mask[:, 4, 4] = True
    dilated, radii = dilate_self_mask(mask, torch.tensor([0.2, 1.0]), sigma_multiplier=3.0)

    assert radii.tolist() == [1, 3]
    assert int(dilated[0].sum()) == 9
    assert int(dilated[1].sum()) == 49


def test_self_and_far_pixels_never_enter_final_depth() -> None:
    camera = DepthCameraConfig(
        name="camera",
        width=3,
        height=3,
        horizontal_fov_deg=60.0,
        vertical_fov_deg=40.0,
        min_range=0.1,
        max_range=2.0,
    )
    terrain_range = torch.tensor([[[1.0, 1.0, 1.0], [1.0, 1.2, -1.0], [1.0, 1.0, 1.0]]])
    scene_range = torch.tensor([[[1.0, 1.0, 1.0], [1.0, 0.4, -1.0], [1.0, 1.0, 1.0]]])
    augmentation = MetricDepthAugmentation(
        MetricDepthAugmentationConfig(
            max_depth_m=2.0,
            blur_probability=1.0,
            sigma_min_px=0.0,
            sigma_max_px=0.0,
        )
    )
    frame = self_occluding_depth_from_sensors(
        _sensor(terrain_range),
        _sensor(scene_range),
        camera,
        SelfOcclusionDepthConfig(dilation_sigma_multiplier=3.0),
        augmentation,
    )

    assert frame.valid_terrain_mask[0, 1].tolist() == [True, False, False]
    assert torch.isfinite(frame.final_depth_z)[0, 1].tolist() == [True, False, False]
    assert frame.self_mask[0, 1].tolist() == [False, True, False]
    assert frame.far_or_no_hit_mask[0, 1].tolist() == [False, False, True]
    # Off-axis ray range must be converted to a smaller optical-axis depth.
    assert 0.8 < float(frame.final_depth_z[0, 1, 0]) < 1.0


def test_validity_aware_blur_does_not_fill_invalid_pixels() -> None:
    augmentation = MetricDepthAugmentation(
        MetricDepthAugmentationConfig(max_depth_m=2.0, blur_probability=1.0, sigma_min_px=1.0, sigma_max_px=1.0)
    )
    depth = torch.tensor([[[1.0, float("nan"), 2.0]]])
    valid = torch.tensor([[[True, False, True]]])
    output, output_valid, _sigma = augmentation.apply_to_valid_depth(depth, valid, sigma=torch.tensor([1.0]))

    assert output_valid.equal(valid)
    assert torch.isfinite(output).tolist() == valid.tolist()


def test_resize_keeps_any_self_covered_output_pixel_invalid() -> None:
    depth = torch.ones(1, 4, 4)
    self_mask = torch.zeros_like(depth, dtype=torch.bool)
    self_mask[0, 0, 0] = True
    depth[self_mask] = float("nan")
    resized, resized_self = resize_depth_with_conservative_invalid_mask(
        depth,
        self_mask,
        target_height=2,
        target_width=2,
    )

    assert resized_self.tolist() == [[[True, False], [False, False]]]
    assert not torch.isfinite(resized[0, 0, 0])
    assert torch.isfinite(resized[0, 0, 1:]).all()
    assert torch.isfinite(resized[0, 1]).all()
