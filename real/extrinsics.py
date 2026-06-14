"""Camera ↔ arm-base extrinsics: the rigid transforms that put measured marker
poses into the base frame the policy was trained in.

The camera (`real/pose.py`) reports every tag in its own frame as a solvePnP
``(rvec, tvec)``. The policy, however, consumes marker poses in the arm **base**
frame (MuJoCo world = the `base` body), as xyz + axis-angle rotation vector
(`src/base_env.py::marker_world_poses`). The single missing link is the camera's
pose in the base frame, `T_base_cam`.

We never touch MuJoCo's OpenGL camera matrix here, so the OpenCV camera
convention (+x right, +y down, +z into the scene) is the *only* convention in
this module — no axis flip ever enters. `T_base_cam` is recovered live from the
fixed table tag (`base_cam_from_table`), and any arm tag is then mapped into the
base frame as `mat_to_pos_rotvec(T_base_cam @ T_cam_tag)`, emitting the same
`(pos, rotvec)` that `marker_world_poses` produces in sim so real and sim
observations are byte-equal.

All transforms are 4×4 homogeneous matrices `T_a_b` mapping a point in frame *b*
to frame *a*: ``p_a = T_a_b @ [p_b; 1]``.
"""
import os

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

EXTRINSICS_PATH = os.path.join(os.path.dirname(__file__), "extrinsics.yaml")


def rt_to_mat(rvec, tvec):
    """solvePnP (rvec, tvec) -> 4×4 transform of the tag in the camera frame."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def mat_to_rt(T):
    """4×4 transform -> (rvec (3,), tvec (3,)) in the solvePnP convention."""
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    return rvec.reshape(3), T[:3, 3].copy()


def mat_inv(T):
    """Inverse of a rigid 4×4 transform (transpose rotation, rotate translation)."""
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def mat_to_pos_rotvec(T):
    """4×4 transform -> (pos (3,), rotvec (3,)).

    `rotvec` is an axis-angle vector, the exact output format of
    `src/base_env.py::marker_world_poses`, so a base-frame tag pose built here is
    interchangeable with the sim's marker observation.
    """
    pos = T[:3, 3].copy()
    rotvec = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    return pos, rotvec


def pos_quat_to_mat(pos, quat_wxyz):
    """(pos, quaternion wxyz) -> 4×4 transform. wxyz matches MuJoCo's convention."""
    w, x, y, z = quat_wxyz
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()  # scipy is xyzw
    T[:3, 3] = pos
    return T


def mat_to_pos_quat(T):
    """4×4 transform -> (pos (3,), quaternion wxyz)."""
    x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()    # scipy is xyzw
    return T[:3, 3].copy(), np.array([w, x, y, z])


def base_cam_from_table(T_base_table, rvec_table, tvec_table):
    """Recover the camera pose in the base frame from a live table-tag detection.

    `T_base_table` is the once-calibrated pose of the fixed table tag in the base
    frame; the camera measures `T_cam_table` this frame. Re-deriving `T_base_cam`
    every frame is what makes the pipeline robust to the camera being bumped.
    """
    T_cam_table = rt_to_mat(rvec_table, tvec_table)
    return T_base_table @ mat_inv(T_cam_table)


def quarter_turn_mat(k):
    """4×4 rotation by k·90° about z (the tag's face normal).

    A tag glued square but rotated (its printed "up" pointing down/sideways) sits
    at a fixed k·90° offset from the sim marker-site frame. Applying the inverse
    in the tag frame — `T_cam_tag @ quarter_turn_mat(-k)` — re-aligns the measured
    pose to the convention the policy trained on, fixing both the calibration
    spread and the rollout marker_rot channel.
    """
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", 90.0 * k, degrees=True).as_matrix()
    return T


def snap_inplane_offset(R_intended, R_meas):
    """Best in-plane glue offset as (k in {0,1,2,3}, residual_deg).

    `E = R_intended.T @ R_meas` is the tag-frame rotation from where the sim site
    expects the tag to point to where it actually points. If the only misglue is
    a quarter turn, E ≈ Rz(k·90°); we return the k minimising the geodesic
    distance from E to Rz(k·90°) plus that distance (deg). A small residual
    confirms the offset really is a clean multiple of 90°; a large one means the
    tag is glued askew (or something else — encoders — is wrong).
    """
    E = R_intended.T @ R_meas
    best_k, best_res = 0, np.inf
    for k in range(4):
        Rz = Rotation.from_euler("z", 90.0 * k, degrees=True).as_matrix()
        res = np.degrees(np.linalg.norm(Rotation.from_matrix(Rz.T @ E).as_rotvec()))
        if res < best_res:
            best_k, best_res = k, res
    return best_k, best_res


def rigid_register(src, dst):
    """Least-squares rigid transform T (4×4) mapping src points -> dst, plus RMS.

    Umeyama with no scale (Kabsch + centroid translation). Uses only positions, so
    it is immune to tag orientation error and solvePnP's flip ambiguity — the
    right tool when the tag centre (tvec) is trustworthy but its rvec is noisy,
    as it is for small fiducials. RMS is the residual point distance in the input
    units, the honest quality gate for the fit.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    cs, cd = src.mean(0), dst.mean(0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cd - R @ cs
    rms = float(np.sqrt((np.linalg.norm(dst - (src @ R.T + T[:3, 3]), axis=1) ** 2).mean()))
    return T, rms


def average_transforms(mats):
    """Robust mean of rigid transforms: median translation + chordal-mean rotation.

    Median translation shrugs off a single bad sample; `Rotation.mean` is the
    proper Fréchet (chordal L2) mean on SO(3) rather than naive matrix averaging.
    """
    mats = np.asarray(mats)
    trans = np.median(mats[:, :3, 3], axis=0)
    rot = Rotation.from_matrix(mats[:, :3, :3]).mean().as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = trans
    return T


def transform_spread(mats, T_ref):
    """Per-sample deviation of `mats` from `T_ref`: (trans_mm (N,), rot_deg (N,)).

    The spread of the per-sample `T_base_table` candidates is the calibration's
    quality gate — large spread means the encoder calibration or the XML marker
    placement is off (see calibration_plan.md Step 1), not sensor noise.
    """
    mats = np.asarray(mats)
    trans_mm = np.linalg.norm(mats[:, :3, 3] - T_ref[:3, 3], axis=1) * 1000.0
    R_ref_inv = T_ref[:3, :3].T
    rot_deg = np.array([
        np.degrees(np.linalg.norm(Rotation.from_matrix(R_ref_inv @ m[:3, :3]).as_rotvec()))
        for m in mats
    ])
    return trans_mm, rot_deg


def save_extrinsics(path, T_base_table, T_base_cam, focus_absolute,
                    n_samples, spread_mm, spread_deg, quarter_turns):
    table_pos, table_quat = mat_to_pos_quat(T_base_table)
    cam_pos, cam_quat = mat_to_pos_quat(T_base_cam)
    data = {
        "t_base_table": {"pos": table_pos.tolist(), "quat": table_quat.tolist()},
        "t_base_cam_fixed": {"pos": cam_pos.tolist(), "quat": cam_quat.tolist()},
        # Per-arm-tag in-plane glue offset (k·90°), keyed by tag id; applied as
        # quarter_turn_mat(-k) to that tag's measured pose at rollout.
        "quarter_turns": {int(tag): int(k) for tag, k in quarter_turns.items()},
        "focus_absolute": int(focus_absolute),
        "n_samples": int(n_samples),
        "spread_mm": float(spread_mm),
        "spread_deg": float(spread_deg),
    }
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=None, sort_keys=False)


def load_extrinsics(path=EXTRINSICS_PATH):
    """Load (T_base_table, T_base_cam_fixed, focus_absolute, quarter_turns).

    `focus_absolute` lets the consumer open the camera at the focus the intrinsics
    (and these extrinsics) were calibrated at. `quarter_turns` maps tag id -> k.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    T_base_table = pos_quat_to_mat(data["t_base_table"]["pos"], data["t_base_table"]["quat"])
    T_base_cam = pos_quat_to_mat(data["t_base_cam_fixed"]["pos"], data["t_base_cam_fixed"]["quat"])
    quarter_turns = {int(tag): int(k) for tag, k in data["quarter_turns"].items()}
    return T_base_table, T_base_cam, int(data["focus_absolute"]), quarter_turns
