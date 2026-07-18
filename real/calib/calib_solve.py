"""Position-based calibration solve from arm-tag samples (no camera or servos).

Shared by the encoder-bias calibrator (`real/calib/calibrate_qpos.py`) and the residual
plot (`real/calib/plot_calib_residuals.py`). A "sample" is `(qpos[6], {tag_id: (rvec,
tvec)})`: the joint angles at a captured pose and each tag's pose in the camera
frame. From a spread of samples these helpers register the camera in the arm base
frame, anchor the fixed table tag, and recover each arm tag's glue offset.

The camera measures every tag in its own frame; the only thing it can't know is how
that frame relates to the arm base. The *arm* tags hand it over for free: MuJoCo FK
already knows where `marker_finger` / `marker_wrist` sit in the base frame at any
pose, so each frame yields `T_base_cam = T_base_armtag(FK) ∘ T_cam_armtag⁻¹`. Anchor
that into the fixed table tag (`T_base_table = T_base_cam ∘ T_cam_table`) and average
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

from real.calib.extrinsics import (
    average_transforms,
    rigid_register,
    rt_to_mat,
    snap_inplane_offset,
    transform_spread,
)
from real.marker_spec import TABLE_TAG_ID
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


def solve_table(samples, T_base_cam):
    """Anchor the fixed table tag: T_base_table = T_base_cam ∘ T_cam_table, averaged
    over poses. The candidates' spread is the table-detection repeatability (camera
    and table are both fixed, so it should be tiny)."""
    cands = np.array([T_base_cam @ rt_to_mat(*poses[TABLE_TAG_ID]) for _, poses in samples])
    T_base_table = average_transforms(cands)
    trans_mm, rot_deg = transform_spread(cands, T_base_table)
    return T_base_table, trans_mm, rot_deg
