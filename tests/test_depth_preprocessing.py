import unittest

import torch

from humanoidverse.perception.depth_preprocessing import (
    DepthCropConfig,
    crop_and_resize_depth,
    crop_and_resize_depth_with_conservative_invalid_mask,
    crop_and_scale_intrinsics,
    crop_depth_roi,
    depth_crop_candidate,
)


class DepthPreprocessingTest(unittest.TestCase):
    def test_reference_crop_scales_to_native_resolution(self):
        crop = depth_crop_candidate("top6_side4")
        self.assertEqual(crop.native_bounds(native_height=270, native_width=480), (45, 270, 30, 450))
        depth = torch.arange(270 * 480).reshape(270, 480)
        cropped = crop_depth_roi(depth, crop)
        self.assertEqual(tuple(cropped.shape), (225, 420))
        self.assertEqual(int(cropped[0, 0]), int(depth[45, 30]))

    def test_crop_then_resize_updates_principal_point_with_pixel_centers(self):
        intrinsic = torch.tensor([[400.0, 0.0, 239.5], [0.0, 300.0, 134.5], [0.0, 0.0, 1.0]])
        target = crop_and_scale_intrinsics(
            intrinsic,
            native_height=270,
            native_width=480,
            target_height=36,
            target_width=64,
            crop=DepthCropConfig(top=6, left=4, right=4),
        )
        # Native ROI is y=[45,270), x=[30,450); its optical center remains centered in x.
        torch.testing.assert_close(
            target,
            torch.tensor(
                [
                    [400.0 * 64.0 / 420.0, 0.0, 31.5],
                    [0.0, 300.0 * 36.0 / 225.0, (134.5 - 45.0 + 0.5) * 36.0 / 225.0 - 0.5],
                    [0.0, 0.0, 1.0],
                ]
            ),
        )

    def test_crop_occurs_before_resize_and_preserves_invalid_mask(self):
        depth = torch.ones((1, 36, 64))
        depth[:, :6] = 9.0
        invalid = torch.zeros_like(depth, dtype=torch.bool)
        invalid[:, 6:8, 4:8] = True
        crop = DepthCropConfig(top=6, left=4, right=4)
        resized = crop_and_resize_depth(depth, target_height=18, target_width=28, crop=crop)
        self.assertTrue(torch.allclose(resized, torch.ones_like(resized)))
        conservative, resized_invalid = crop_and_resize_depth_with_conservative_invalid_mask(
            depth,
            invalid,
            target_height=15,
            target_width=28,
            crop=crop,
        )
        self.assertTrue(resized_invalid.any())
        self.assertTrue(torch.isnan(conservative[resized_invalid]).all())

    def test_invalid_crop_is_rejected(self):
        with self.assertRaises(ValueError):
            DepthCropConfig(top=36).validate()


if __name__ == "__main__":
    unittest.main()
