"""Dense-stereo preprocessing, consistency filtering and cloud contracts."""

import numpy as np

from real.calib.calibrate_stereo import StereoRectification
from real.tracking.dense_stereo import (
    DenseStereoConfig,
    FastFoundationCandidate,
    SGBMCandidate,
    disparity_range_from_depths,
    disparity_to_cloud,
    inpaint_sponge_tags,
    left_right_validity,
    sample_cloud,
    voxel_downsample,
)
from real.tracking.eval_fast_foundation import max_disparity_for_range


def _config():
    return DenseStereoConfig(
        processing_scale=0.5,
        processing_height_multiple=32,
        tag_inpaint_dilate_px=0,
        tag_inpaint_radius_px=2,
        static_mean_window_s=0.15,
        held_out_fraction=0.25,
        min_static_window_frames=5,
        development_frame_stride=8,
        disparity_margin_px=4,
        lr_max_error_px=1.5,
        point_sample_count=4,
        voxel_size_m=0.01,
        workspace_z_m=(-1.0, 10.0),
        depth_mad_scale=4.0,
        depth_mad_floor_m=0.01,
        min_mask_area_px=10,
        sgbm_candidates=(SGBMCandidate(3, 5, 50, 1, 1),),
        fast_max_disp_multiple=32,
        fast_candidates=(FastFoundationCandidate("23-36-37", 8),),
    )


def _rectification():
    P1 = np.array([[100.0, 0.0, 4.0, 0.0],
                   [0.0, 100.0, 3.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0]])
    P2 = P1.copy()
    P2[0, 3] = -10.0
    return StereoRectification(
        image_size=(8, 6),
        T_aux_main=np.eye(4),
        rotations={"main": np.eye(3), "aux": np.eye(3)},
        projections={"main": P1, "aux": P2},
        Q=np.eye(4),
        valid_rois={"main": (0, 0, 8, 6), "aux": (0, 0, 8, 6)},
        anchor_reference_T_aux_main=None,
    )


def test_disparity_range_is_derived_and_rounded_for_sgbm():
    minimum, count = disparity_range_from_depths(
        np.array([0.25, 0.50]), _rectification(), _config())

    assert minimum == 6
    assert count == 32


def test_fast_max_disparity_covers_complete_derived_interval():
    assert max_disparity_for_range(84, 176, 32) == 288


def test_left_right_validity_uses_opposite_disparity_signs():
    left = np.full((2, 8), 2.0, dtype=np.float32)
    right = np.full((2, 8), -2.0, dtype=np.float32)
    right[:, 3] = -4.0

    valid, residual, right_x = left_right_validity(left, right, 0.5)

    assert not valid[:, :2].any()
    assert valid[:, 2:5].all()
    assert not valid[:, 5].any()
    assert right_x[0, 4] == 2
    assert residual[0, 4] == 0.0


def test_inpainting_removes_sponge_tag_but_not_table_tag():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[:, :15] = 80
    tags = {
        "1": {"corners": [[3, 3], [8, 3], [8, 8], [3, 8]]},
        "10": {"corners": [[20, 3], [25, 3], [25, 8], [20, 8]]},
    }

    _, mask = inpaint_sponge_tags(image, tags, dilate_px=0, radius_px=2)

    assert mask[5, 5]
    assert not mask[5, 22]


def test_disparity_cloud_applies_masks_consistency_and_base_transform():
    disparity = np.full((3, 6), 2.0, dtype=np.float32)
    reverse = np.full((3, 6), -2.0, dtype=np.float32)
    left_mask = np.zeros((3, 6), dtype=bool)
    left_mask[1, 2:5] = True
    right_mask = np.zeros((3, 6), dtype=bool)
    right_mask[1, :3] = True
    Q = np.array([[1.0, 0.0, 0.0, -2.0],
                  [0.0, 1.0, 0.0, -1.0],
                  [0.0, 0.0, 0.0, 10.0],
                  [0.0, 0.0, 1.0, 0.0]])
    T = np.eye(4)
    T[:3, 3] = (0.2, 0.0, 0.0)

    result = disparity_to_cloud(
        disparity, reverse, left_mask, right_mask, Q, T, _config(),
        (np.array([-2.0, -2.0]), np.array([2.0, 2.0])))

    assert result.left_mask_pixels == 3
    assert result.correspondence_rejected == 0
    assert result.points_base.shape == (3, 3)
    np.testing.assert_allclose(result.points_base[:, 2], 5.0)
    np.testing.assert_allclose(result.points_base[0], (0.2, 0.0, 5.0))


def test_voxel_and_fixed_cloud_contract_are_deterministic():
    points = np.array([[0.001, 0.0, 0.0], [0.002, 0.0, 0.0],
                       [0.020, 0.0, 0.0]])
    confidence = np.array([0.5, 0.6, 0.9])

    reduced, reduced_confidence = voxel_downsample(points, confidence, 0.01)
    tensor, center = sample_cloud(reduced, reduced_confidence, 4)

    assert reduced.shape == (2, 3)
    assert tensor.shape == (4, 5)
    np.testing.assert_array_equal(tensor[:, 4], (1.0, 1.0, 0.0, 0.0))
    np.testing.assert_allclose(center, (0.0105, 0.0, 0.0))
