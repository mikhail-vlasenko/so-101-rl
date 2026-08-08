"""Pure geometry contracts for the live tag-vs-shape viewer."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from real.calib.extrinsics import mat_to_rt
from real.tracking.view_object_estimate import (
    comparison_metrics,
    make_view_model,
    tag_body_gt,
)
from src.shape_obs import box_sqrtm, sqrtm_upper


HALF_EXTENTS = np.array([0.03, 0.02, 0.0125])


def test_view_model_uses_calibrated_sponge_size():
    model, _, _, geom = make_view_model(HALF_EXTENTS)
    assert model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX
    np.testing.assert_allclose(model.geom_size[geom], HALF_EXTENTS)


def test_tag_body_gt_accepts_front_facing_pose_and_rejects_back_facing():
    front = np.eye(4)
    front[:3, :3] = Rotation.from_euler("x", np.pi).as_matrix()
    front[:3, 3] = [0.0, 0.0, 0.4]
    tag_poses = {
        "main": {1: mat_to_rt(front)},
        "aux": {},
    }
    anchors = {"main": np.eye(4)}
    T_base_body, accepted = tag_body_gt(
        tag_poses, anchors, {1: np.eye(4)})
    np.testing.assert_allclose(T_base_body, front, atol=1e-12)
    assert accepted[("main", 1)] < 1e-6

    back = front.copy()
    back[:3, :3] = np.eye(3)
    tag_poses["main"][1] = mat_to_rt(back)
    T_base_body, accepted = tag_body_gt(
        tag_poses, anchors, {1: np.eye(4)})
    assert T_base_body is None
    assert accepted == {}


def test_comparison_metrics_are_zero_for_exact_observation():
    T_base_body = np.eye(4)
    T_base_body[:3, :3] = Rotation.from_euler("xyz", [0.2, -0.4, 0.7]).as_matrix()
    T_base_body[:3, 3] = [0.22, 0.03, 0.04]
    sqrtm = box_sqrtm(T_base_body[:3, :3], HALF_EXTENTS)

    metrics = comparison_metrics(
        T_base_body, T_base_body[:3, 3], sqrtm_upper(sqrtm), HALF_EXTENTS)

    np.testing.assert_allclose(metrics["delta_mm"], 0.0, atol=1e-12)
    assert metrics["center_mm"] < 1e-12
    assert metrics["sqrtm_mm"] < 1e-12
    assert metrics["axis_deg"] < 1e-6
    np.testing.assert_allclose(metrics["half_mm"],
                               np.sort(HALF_EXTENTS * 1000.0), atol=1e-12)
