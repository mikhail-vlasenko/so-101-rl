"""Measure a camera's sim mount pose from the live table-tag anchoring.

The sim scene (so101.xml `tag_cam_mount` / `tag_cam_aux_mount`) needs each
camera's base-frame pose as a literal pos/quat. The real pipeline never stores
one — every camera re-anchors itself per frame from the table tag
(`base_cam_from_table`) — so this script simply takes that same measurement
once, robust-averages it over N frames, converts the OpenCV camera frame
(+x right, +y down, +z into the scene) to MuJoCo's camera convention (looks
down -z, +y up; a 180-degree flip about the camera x-axis, pinned by
tests/real/test_cam_mount_convention.py against the main camera's existing
extrinsics/XML pair), and prints the XML-ready line. Run it whenever a camera
is remounted; paste the output into so101.xml.

Usage:
    conda run -n mujoco_env python -m real.diagnostics.snapshot_cam_mount --camera aux
    conda run -n mujoco_env python -m real.diagnostics.snapshot_cam_mount --camera main
"""
import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np

from real.calib.extrinsics import (
    average_transforms,
    base_cam_from_table,
    load_extrinsics,
    mat_to_pos_quat,
    transform_spread,
)
from real.marker_spec import TABLE_TAG_ID
from real.vision.detect import make_detector
from real.vision.pose import PoseEstimator
from real.vision.stereo_rig import open_rig_camera

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCENE_XML = REPO_ROOT / "so101" / "scene.xml"

MOUNT_BODY = {"main": "tag_cam_mount", "aux": "tag_cam_aux_mount"}

# OpenCV camera frame -> MuJoCo camera frame: 180 degrees about the camera's
# x-axis (+y down/+z forward becomes +y up/-z forward).
_CV_TO_MJ = np.diag([1.0, -1.0, -1.0])


def mount_pose_from_T_base_cam(T_base_cam):
    """(pos (3,), quat wxyz (4,)) of the MuJoCo mount body for an OpenCV-frame
    `T_base_cam`. The mount body carries the camera at identity, so the body
    frame IS the MuJoCo camera frame."""
    T = T_base_cam.copy()
    T[:3, :3] = T[:3, :3] @ _CV_TO_MJ
    return mat_to_pos_quat(T)


def current_xml_mount(camera):
    """(pos, quat wxyz) of the camera's mount body as the scene XML has it now."""
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, MOUNT_BODY[camera])
    assert body_id >= 0, f"body '{MOUNT_BODY[camera]}' not found in {SCENE_XML}"
    return model.body_pos[body_id].copy(), model.body_quat[body_id].copy()


def main():
    parser = argparse.ArgumentParser(
        description="Snapshot a camera's sim mount pose from table-tag anchoring")
    parser.add_argument("--camera", choices=("main", "aux"), default="aux")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--family", choices=("apriltag", "aruco"), default="apriltag")
    args = parser.parse_args()

    T_base_table, _, _, _ = load_extrinsics()
    detector = make_detector(args.family)
    cap, camera_matrix, dist_coeffs = open_rig_camera(args.camera)
    estimator = PoseEstimator(camera_matrix, dist_coeffs)

    solves = []
    try:
        for _ in range(args.frames):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"camera read failed on '{args.camera}'")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dets = {d.id: d for d in detector.detect(gray)}
            if TABLE_TAG_ID in dets:
                solves.append(base_cam_from_table(
                    T_base_table, *estimator.estimate(dets[TABLE_TAG_ID])))
    finally:
        cap.release()

    if len(solves) < args.frames // 2:
        raise RuntimeError(
            f"table tag (id {TABLE_TAG_ID}) seen in only {len(solves)}/"
            f"{args.frames} frames; fix the '{args.camera}' framing first")

    T_base_cam = average_transforms(solves)
    trans_mm, rot_deg = transform_spread(solves, T_base_cam)
    pos, quat = mount_pose_from_T_base_cam(T_base_cam)

    print(f"{args.camera}: {len(solves)}/{args.frames} anchored frames, "
          f"per-frame spread {trans_mm.mean():.2f} mm / {rot_deg.mean():.3f} deg")
    xml_pos, xml_quat = current_xml_mount(args.camera)
    if xml_quat @ quat < 0.0:
        xml_quat = -xml_quat   # quaternion double cover: compare same-sign
    print(f"current XML: pos {np.array2string(xml_pos, precision=6)}  "
          f"quat {np.array2string(xml_quat, precision=6)}")
    print(f"  delta: {np.linalg.norm(pos - xml_pos) * 1e3:.1f} mm, "
          f"{np.linalg.norm(quat - xml_quat):.4f} quat")
    print("\npaste into so101.xml:")
    print(f'    <body name="{MOUNT_BODY[args.camera]}" '
          f'pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}"\n'
          f'          quat="{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}">')


if __name__ == "__main__":
    main()
