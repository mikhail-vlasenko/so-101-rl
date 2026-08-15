"""Measure a camera's sim mount pose from the live two-tag board anchoring.

The sim scene (so101.xml `tag_cam_mount` / `tag_cam_aux_mount`) needs each
camera's base-frame pose as a literal pos/quat. The real pipeline never stores
one — every camera re-anchors itself from accepted board observations — so this
script simply takes that same measurement
once, robust-averages it over N frames, converts the OpenCV camera frame
(+x right, +y down, +z into the scene) to MuJoCo's camera convention (looks
down -z, +y up; a 180-degree flip about the camera x-axis, pinned by
exact frame-direction and round-trip tests in
tests/real/test_cam_mount_convention.py), and prints the XML-ready line. Run it
whenever a camera is remounted. The complete stereo workflow calls the reusable
measurement and XML-update functions below so both mounts are replaced only
after both measurements succeed.

Usage:
    conda run -n mujoco_env python -m real.diagnostics.snapshot_cam_mount --camera aux
    conda run -n mujoco_env python -m real.diagnostics.snapshot_cam_mount --camera main
"""
import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import re

import cv2
import mujoco
import numpy as np

from real.calib.extrinsics import (
    average_transforms,
    mat_to_pos_quat,
    transform_spread,
)
from real.calib.table_anchor import TableAnchorTracker, load_table_anchor_limits
from real.marker_spec import TABLE_TAG_IDS
from real.vision.detect import make_detector
from real.vision.stereo_rig import open_rig_camera

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCENE_XML = REPO_ROOT / "so101" / "scene.xml"
SO101_XML = REPO_ROOT / "so101" / "so101.xml"

MOUNT_BODY = {"main": "tag_cam_mount", "aux": "tag_cam_aux_mount"}

# OpenCV camera frame -> MuJoCo camera frame: 180 degrees about the camera's
# x-axis (+y down/+z forward becomes +y up/-z forward).
_CV_TO_MJ = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class MountSnapshot:
    camera: str
    accepted_frames: int
    requested_frames: int
    translation_spread_mean_mm: float
    rotation_spread_mean_deg: float
    pos: np.ndarray
    quat: np.ndarray


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


def snapshot_mount(camera, frames=100, family="apriltag"):
    """Robustly measure one XML-ready MuJoCo camera mount pose."""
    if camera not in MOUNT_BODY:
        raise ValueError(f"unknown camera {camera!r}; expected one of {tuple(MOUNT_BODY)}")
    if frames <= 0:
        raise ValueError(f"frames must be positive, got {frames}")

    detector = make_detector(family)
    cap, camera_matrix, dist_coeffs = open_rig_camera(camera)
    tracker = TableAnchorTracker(
        camera_matrix, dist_coeffs,
        limits=replace(load_table_anchor_limits(), ema_alpha=1.0))

    solves = []
    try:
        for _ in range(frames):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"camera read failed on '{camera}'")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dets = {d.id: d for d in detector.detect(gray)}
            if tracker.observe(dets):
                solves.append(tracker.value())
    finally:
        cap.release()
        detector.close()

    minimum_accepted = (frames + 1) // 2
    if len(solves) < minimum_accepted:
        raise RuntimeError(
            f"table anchor pair {TABLE_TAG_IDS} accepted in only {len(solves)}/"
            f"{frames} frames; fix the '{camera}' framing first")

    T_base_cam = average_transforms(solves)
    trans_mm, rot_deg = transform_spread(solves, T_base_cam)
    pos, quat = mount_pose_from_T_base_cam(T_base_cam)
    return MountSnapshot(
        camera=camera,
        accepted_frames=len(solves),
        requested_frames=frames,
        translation_spread_mean_mm=float(trans_mm.mean()),
        rotation_spread_mean_deg=float(rot_deg.mean()),
        pos=pos,
        quat=quat,
    )


def _formatted_vector(values):
    return " ".join(f"{float(value):.6f}" for value in values)


def update_scene_mounts(snapshots, path=SO101_XML):
    """Atomically replace exactly the requested mount poses in the source XML."""
    unknown = set(snapshots) - set(MOUNT_BODY)
    if unknown:
        raise ValueError(f"unknown cameras in mount snapshots: {sorted(unknown)}")
    if not snapshots:
        raise ValueError("at least one mount snapshot is required")

    updated = path.read_text()
    for camera, snapshot in snapshots.items():
        if snapshot.camera != camera:
            raise ValueError(
                f"snapshot key {camera!r} disagrees with camera {snapshot.camera!r}")
        if snapshot.pos.shape != (3,) or snapshot.quat.shape != (4,):
            raise ValueError(
                f"{camera} mount pose must be pos (3,) and quat (4,), got "
                f"{snapshot.pos.shape} and {snapshot.quat.shape}")
        if not np.all(np.isfinite(snapshot.pos)) or not np.all(np.isfinite(snapshot.quat)):
            raise ValueError(f"{camera} mount pose contains non-finite values")
        body_name = MOUNT_BODY[camera]
        pattern = re.compile(
            rf'(<body name="{re.escape(body_name)}" pos=")[^"]+'
            rf'("\s+quat=")[^"]+(">)')
        replacement = (
            rf'\g<1>{_formatted_vector(snapshot.pos)}'
            rf'\g<2>{_formatted_vector(snapshot.quat)}\g<3>')
        updated, count = pattern.subn(replacement, updated)
        if count != 1:
            raise RuntimeError(
                f"expected exactly one editable body {body_name!r} in {path}, found {count}")

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(updated)
    temporary.replace(path)


def print_snapshot(snapshot, show_xml=True):
    print(
        f"{snapshot.camera}: {snapshot.accepted_frames}/{snapshot.requested_frames} "
        f"anchored frames, per-frame spread "
        f"{snapshot.translation_spread_mean_mm:.2f} mm / "
        f"{snapshot.rotation_spread_mean_deg:.3f} deg")
    xml_pos, xml_quat = current_xml_mount(snapshot.camera)
    if xml_quat @ snapshot.quat < 0.0:
        xml_quat = -xml_quat
    print(f"current XML: pos {np.array2string(xml_pos, precision=6)}  "
          f"quat {np.array2string(xml_quat, precision=6)}")
    print(f"  delta: {np.linalg.norm(snapshot.pos - xml_pos) * 1e3:.1f} mm, "
          f"{np.linalg.norm(snapshot.quat - xml_quat):.4f} quat")
    if show_xml:
        print("\npaste into so101.xml:")
        print(f'    <body name="{MOUNT_BODY[snapshot.camera]}" '
              f'pos="{_formatted_vector(snapshot.pos)}"\n'
              f'          quat="{_formatted_vector(snapshot.quat)}">')


def main():
    parser = argparse.ArgumentParser(
        description="Snapshot a camera's sim mount pose from two-tag anchoring")
    parser.add_argument("--camera", choices=("main", "aux"), default="aux")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--family", choices=("apriltag", "aruco"), default="apriltag")
    args = parser.parse_args()

    snapshot = snapshot_mount(args.camera, args.frames, args.family)
    print_snapshot(snapshot)


if __name__ == "__main__":
    main()
