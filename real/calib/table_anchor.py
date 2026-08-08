"""Two-tag table-board anchoring for every real-camera consumer.

AprilTags 10 and 11 are calibrated as coplanar squares in the arm base frame.
A frame updates the camera pose only when both complete tags are detected. Their
eight corners are solved together as one large planar target, then checked two
ways: joint reprojection error and disagreement between the two independent
single-square camera poses. Accepted solves feed a rigid-pose EMA. A frame with
one/no tag, or a failed consistency gate, leaves that EMA untouched; callers
therefore coast on the exact last accepted camera pose.

The board calibration fixes every tag centre at base z=0 and every normal at
base +z. This makes the physical table plane—not a noisy per-tag rvec—the source
of roll and pitch. Once shared stereo calibration exists, the same observation
solver can update one rigid-rig EMA and derive both camera poses from its fixed
main-to-aux transform; until then each camera owns an identical tracker.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from real.calib.extrinsics import (
    PoseEMA,
    base_cam_from_table,
    load_table_anchor_poses,
    mat_inv,
    rt_to_mat,
)
from real.marker_spec import TABLE_TAG_IDS
from real.vision.pose import PoseEstimator, tag_object_points


CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "conf" / "config.yaml"


@dataclass(frozen=True)
class TableAnchorLimits:
    ema_alpha: float
    max_reprojection_rmse_px: float
    max_camera_translation_disagreement_mm: float
    max_camera_rotation_disagreement_deg: float


def load_table_anchor_limits(path: Path = CONFIG_PATH) -> TableAnchorLimits:
    with path.open() as stream:
        cfg = yaml.safe_load(stream)["table_anchor"]
    return TableAnchorLimits(
        ema_alpha=float(cfg["ema_alpha"]),
        max_reprojection_rmse_px=float(cfg["max_reprojection_rmse_px"]),
        max_camera_translation_disagreement_mm=float(
            cfg["max_camera_translation_disagreement_mm"]),
        max_camera_rotation_disagreement_deg=float(
            cfg["max_camera_rotation_disagreement_deg"]),
    )


@dataclass(frozen=True)
class TableAnchorQuality:
    visible_ids: tuple[int, ...]
    updated: bool
    reprojection_rmse_px: float | None
    camera_translation_disagreement_mm: float | None
    camera_rotation_disagreement_deg: float | None
    rejection: str | None


def _board_correspondences(detections, anchor_poses):
    object_points = []
    image_points = []
    for tag in TABLE_TAG_IDS:
        T_base_tag = anchor_poses[tag]
        corners_tag = tag_object_points(tag)
        corners_base = (corners_tag @ T_base_tag[:3, :3].T
                        + T_base_tag[:3, 3])
        object_points.append(corners_base)
        image_points.append(detections[tag].corners.astype(np.float64))
    return np.vstack(object_points), np.vstack(image_points)


def _camera_disagreement(candidates):
    first, second = (candidates[tag] for tag in TABLE_TAG_IDS)
    translation_mm = float(np.linalg.norm(
        first[:3, 3] - second[:3, 3]) * 1000.0)
    relative = first[:3, :3].T @ second[:3, :3]
    rotation_deg = float(np.degrees(
        np.linalg.norm(Rotation.from_matrix(relative).as_rotvec())))
    return translation_mm, rotation_deg


class TableAnchorTracker:
    """Both-tags-only board solve followed by an episode-local pose EMA."""

    def __init__(self, camera_matrix, dist_coeffs, *, anchor_poses=None,
                 limits=None):
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        self.dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)
        self.anchor_poses = (load_table_anchor_poses()
                             if anchor_poses is None else anchor_poses)
        assert set(self.anchor_poses) == set(TABLE_TAG_IDS)
        self.limits = (load_table_anchor_limits() if limits is None else limits)
        self.estimator = PoseEstimator(self.camera_matrix, self.dist_coeffs)
        self.ema = PoseEMA(self.limits.ema_alpha)
        self.quality = TableAnchorQuality((), False, None, None, None,
                                          "both table tags not visible")

    @property
    def seeded(self):
        return self.ema.seeded

    def value(self):
        return self.ema.value() if self.seeded else None

    def observe(self, detections) -> bool:
        """Possibly update the EMA; return true only for an accepted pair."""
        visible = tuple(tag for tag in TABLE_TAG_IDS if tag in detections)
        if len(visible) != len(TABLE_TAG_IDS):
            self.quality = TableAnchorQuality(
                visible, False, None, None, None,
                "both table tags not visible")
            return False

        candidates = {}
        for tag in TABLE_TAG_IDS:
            rvec, tvec = self.estimator.estimate(detections[tag])
            candidates[tag] = base_cam_from_table(
                self.anchor_poses[tag], rvec, tvec)
        translation_mm, rotation_deg = _camera_disagreement(candidates)
        if translation_mm > self.limits.max_camera_translation_disagreement_mm:
            self.quality = TableAnchorQuality(
                visible, False, None, translation_mm, rotation_deg,
                "tag camera positions disagree")
            return False
        if rotation_deg > self.limits.max_camera_rotation_disagreement_deg:
            self.quality = TableAnchorQuality(
                visible, False, None, translation_mm, rotation_deg,
                "tag camera rotations disagree")
            return False

        object_points, image_points = _board_correspondences(
            detections, self.anchor_poses)
        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_points, self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            raise RuntimeError("two-tag table-board solvePnP failed")
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        residual = projected.reshape(-1, 2) - image_points
        reprojection_rmse_px = float(np.sqrt(np.mean(np.sum(
            residual * residual, axis=1))))
        if reprojection_rmse_px > self.limits.max_reprojection_rmse_px:
            self.quality = TableAnchorQuality(
                visible, False, reprojection_rmse_px, translation_mm, rotation_deg,
                "board reprojection error too high")
            return False

        T_base_cam = mat_inv(rt_to_mat(rvec.reshape(3), tvec.reshape(3)))
        self.ema.update(T_base_cam)
        self.quality = TableAnchorQuality(
            visible, True, reprojection_rmse_px, translation_mm, rotation_deg, None)
        return True
