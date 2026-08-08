"""Position-based calibration solve from arm-tag samples (no camera or servos).

Shared by the encoder-bias calibrator (`real/calib/calibrate_qpos.py`) and the residual
plot (`real/calib/plot_calib_residuals.py`). A "sample" is `(qpos[6], {tag_id: (rvec,
tvec)})`: the joint angles at a captured pose and each tag's pose in the camera
frame. From a spread of samples these helpers register the camera in the arm base
frame, anchor the fixed two-tag table board, and recover each arm tag's glue offset.

The camera measures every tag in its own frame; the only thing it can't know is how
that frame relates to the arm base. The *arm* tags hand it over for free: MuJoCo FK
already knows where `marker_finger` / `marker_wrist` sit in the base frame at any
pose, so each frame yields `T_base_cam = T_base_armtag(FK) ∘ T_cam_armtag⁻¹`. Anchor
that into both fixed table tags, level their shared plane to base z=0, and average
over poses. The registration is position-only (tag *centres*), so it is immune to the
glue rotation and to solvePnP's rvec flips on the small arm tags — the failure modes
that wreck an orientation bridge.

A tag glued with its printed "up" pointing down/sideways sits at a constant k·90°
offset from the sim marker-site frame; because each pose re-expresses that fixed
link-frame rotation differently in the base frame, it shows up as a large *spread*,
not a constant bias. `determine_quarter_turns` resolves it per tag using the
*approximate* sim camera orientation as a prior (only good enough to tell candidates
90° apart), so the rollout can apply the same correction to its marker_rot channel.
"""
import json

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from real.calib.extrinsics import (
    average_transforms,
    rigid_register,
    rt_to_mat,
    snap_inplane_offset,
    transform_spread,
)
from real.marker_spec import TABLE_TAG_IDS
from src.base_env import TAG_CAM_NAME


def site_mat(data, site_id):
    """4×4 world(base)-frame transform of a MuJoCo site from its xpos/xmat."""
    T = np.eye(4)
    T[:3, :3] = data.site_xmat[site_id].reshape(3, 3)
    T[:3, 3] = data.site_xpos[site_id]
    return T


def sim_cam_R_opencv(model, data):
    """Approximate camera orientation (world->cam, OpenCV convention) from the sim.

    MuJoCo's camera frame is OpenGL (looks down -z, +y up); solvePnP is OpenCV
    (+z forward, +y down), so flip y and z. Only used as a coarse prior to pick
    the right quarter turn, where being tens of degrees off is harmless."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, TAG_CAM_NAME)
    assert cam_id >= 0, f"camera {TAG_CAM_NAME!r} not found in model"
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)
    R_gl = data.cam_xmat[cam_id].reshape(3, 3).copy()
    return R_gl @ np.diag([1.0, -1.0, -1.0])


def save_samples(path, samples):
    out = {"samples": [
        {"qpos": qpos.tolist(),
         "tags": {str(tag): {"rvec": rvec.tolist(), "tvec": tvec.tolist()}
                  for tag, (rvec, tvec) in poses.items()}}
        for qpos, poses in samples]}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def load_samples(path):
    with open(path) as f:
        data = json.load(f)
    samples = []
    for s in data["samples"]:
        poses = {int(tag): (np.array(t["rvec"]), np.array(t["tvec"]))
                 for tag, t in s["tags"].items()}
        samples.append((np.array(s["qpos"]), poses))
    return samples


def determine_quarter_turns(samples, model, data, qposadr, site_ids, R_cam_approx):
    """Vote each arm tag's in-plane glue offset k (90° multiples) via the sim prior.

    Returns (quarter_turns {tag: k}, report {tag: (k, residuals_deg, votes)})."""
    votes = {tag: [] for tag in site_ids}
    resid = {tag: [] for tag in site_ids}
    for qpos, poses in samples:
        data.qpos[qposadr] = qpos
        mujoco.mj_kinematics(model, data)
        for tag, sid in site_ids.items():
            if tag not in poses:
                continue
            R_meas = R_cam_approx @ rt_to_mat(*poses[tag])[:3, :3]
            k, res = snap_inplane_offset(data.site_xmat[sid].reshape(3, 3), R_meas)
            votes[tag].append(k)
            resid[tag].append(res)
    quarter_turns, report = {}, {}
    for tag in site_ids:
        if not votes[tag]:
            raise RuntimeError(f"tag {tag} never captured; cannot determine its offset")
        k = int(np.bincount(votes[tag]).argmax())
        quarter_turns[tag] = k
        report[tag] = (k, np.array(resid[tag]), np.array(votes[tag]))
    return quarter_turns, report


def paired_points(samples, model, data, qposadr, site_ids):
    """Pair every detected arm tag with its FK position across all poses.

    Returns (src, dst, tags): `src[i]` is a tag centre in the camera frame (tvec),
    `dst[i]` the same tag's base-frame position from FK on the sample's qpos, and
    `tags[i]` its id. The qpos fed here is whatever the caller stored — pass
    encoder-bias-corrected qpos to make `dst` the true base-frame placement."""
    src, dst, tags = [], [], []
    for qpos, poses in samples:
        data.qpos[qposadr] = qpos
        mujoco.mj_kinematics(model, data)
        for tag, sid in site_ids.items():
            if tag not in poses:
                continue
            src.append(poses[tag][1])               # tvec: tag centre in camera frame
            dst.append(data.site_xpos[sid].copy())  # tag centre in base frame (FK)
            tags.append(tag)
    return np.array(src), np.array(dst), np.array(tags)


def solve_camera(samples, model, data, qposadr, site_ids):
    """Register T_base_cam from tag *centres* (Umeyama): pair each tag's camera-frame
    position (tvec) with its base-frame FK position, across all poses/tags. Returns
    (T_base_cam, rms_mm, tags, err_mm) with per-point residuals."""
    src, dst, tags = paired_points(samples, model, data, qposadr, site_ids)
    T_base_cam, rms = rigid_register(src, dst)
    err_mm = np.linalg.norm(dst - (src @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]), axis=1) * 1000.0
    return T_base_cam, rms * 1000.0, tags, err_mm


def solve_table_anchors(samples, T_base_cam):
    """Calibrate both fixed table tags while enforcing the known table plane.

    The camera/arm registration supplies each tag's base-frame centre and yaw.
    Roll, pitch and centre z are physical table constraints instead: every tag
    lies at z=0 with normal +z. This prevents planar PnP orientation bias from
    tilting the base-frame ground plane. Returns ``(poses, translation_spread,
    rotation_spread)`` with the repeatability arrays pooled across both tags.
    """
    anchors = {}
    all_translation_mm = []
    all_rotation_deg = []
    for tag in TABLE_TAG_IDS:
        cands = np.array([
            T_base_cam @ rt_to_mat(*poses[tag]) for _, poses in samples])
        raw = average_transforms(cands)
        trans_mm, rot_deg = transform_spread(cands, raw)
        all_translation_mm.extend(trans_mm)
        all_rotation_deg.extend(rot_deg)

        x_axis = raw[:3, 0]
        yaw = np.arctan2(x_axis[1], x_axis[0])
        leveled = np.eye(4)
        leveled[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
        leveled[:2, 3] = raw[:2, 3]
        anchors[tag] = leveled
    return (anchors, np.asarray(all_translation_mm),
            np.asarray(all_rotation_deg))
