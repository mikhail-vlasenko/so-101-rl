"""Solve the sponge tags' in-plane placement on the box -> sponge_tags.yaml.

The shape-tracking dataset (real/tracking/record_shapes.py) needs the sponge's
GT body pose from whatever tag happens to be visible, so every glued tag's
transform to the body frame must be known. Each tag sits on a manually
declared face (FACE_OF_TAG below — edit after gluing) with three unknowns:
in-plane offset (u, v) and yaw about the face normal; the face plane itself is
pinned by the box geometry (half extents read from the sim scene XML — the
single source of the sponge dimensions).

The solve consumes frames where >= 2 sponge tags are co-visible in one camera:
each such pair gives a measured tag_i -> tag_j relative transform that the
per-tag placements must reproduce through the box. A weak prior toward the
face centers (tags are glued roughly centered) pins any gauge freedom left
when the observed pairs span too few distinct faces. GT body pose then follows
from any single visible tag: T_cam_body = T_cam_tag @ inv(T_body_tag).

Run (tags glued, sponge held so several faces are visible; reposition between
frames for view diversity):
    conda run -n mujoco_env python -m real.tracking.tag_body_calib --frames 60
"""
import argparse
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from real.calib.extrinsics import mat_inv, mat_to_pos_quat, pos_quat_to_mat, rt_to_mat
from real.marker_spec import SPONGE_TAG_IDS
from real.vision.detect import make_detector
from real.vision.pose import PoseEstimator
from real.vision.stereo_rig import CAMERA_NAMES, open_rig_camera

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCENE_XML = REPO_ROOT / "so101" / "scene_lift.xml"
SPONGE_TAGS_PATH = Path(__file__).resolve().parent / "sponge_tags.yaml"

# Which box face each sponge tag is glued to, by outward normal in the body
# frame (the cube_geom frame: x the 6 cm axis, y the 4 cm axis, z the 2.5 cm
# axis). EDIT to match the physical gluing before running; ids not glued this
# session can simply be removed. Id 1 on a largest (+z) face matches the
# historic cube_tag site.
FACE_OF_TAG = {
    1: "+z", 3: "-z",
    4: "+y", 5: "-y",
    6: "+x", 7: "-x",
}

# Length scale (m) weighting rotational residuals against translational ones
# in the pair fit — errors of 1 rad and 1/ROT_WEIGHT_M meters count equally.
ROT_WEIGHT_M = 0.02
# Weak zero-prior on (u, v, yaw): pins gauge freedom without fighting the data
# (the pair residuals are ~1e-3 m scale; the prior contributes ~1e-3 * offset).
PRIOR_WEIGHT = 1e-3

_AXES = {"x": 0, "y": 1, "z": 2}


def load_box_half_extents():
    """Sponge half extents from the sim scene — the single source of the box
    dimensions (so101/scene_lift.xml cube_geom)."""
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    assert gid >= 0, f"cube_geom not found in {SCENE_XML}"
    return model.geom_size[gid].copy()


def face_frame(face, half_extents):
    """(origin (3,), R (3,3)) of a face's tag frame in the body frame: origin
    at the face center, columns (a, b, n) with n the outward normal and (a, b)
    the in-plane axes of the (u, v) offset — right-handed (a x b = n)."""
    sign = {"+": 1.0, "-": -1.0}[face[0]]
    k = _AXES[face[1]]
    n = np.zeros(3)
    n[k] = sign
    a = np.zeros(3)
    b = np.zeros(3)
    if sign > 0:
        a[(k + 1) % 3] = 1.0
        b[(k + 2) % 3] = 1.0
    else:
        a[(k + 2) % 3] = 1.0
        b[(k + 1) % 3] = 1.0
    origin = n * half_extents[k]
    return origin, np.column_stack([a, b, n])


def tag_body_transform(face, half_extents, u, v, yaw):
    """T_body_tag for a tag on `face` at in-plane offset (u, v) and yaw (rad)
    about the outward normal. Tag frame: +z out of the printed face."""
    origin, R_face = face_frame(face, half_extents)
    T = np.eye(4)
    T[:3, :3] = R_face @ Rotation.from_euler("z", yaw).as_matrix()
    T[:3, 3] = origin + R_face[:, 0] * u + R_face[:, 1] * v
    return T


def solve_tag_placements(pairs, faces, half_extents):
    """Least-squares (u, v, yaw) per tag from co-visibility pairs.

    `pairs` is a list of (id_i, id_j, T_ti_tj) measured relative transforms
    (tag j in tag i's frame), `faces` maps tag id -> face string. Returns
    ({id: (u, v, yaw)}, residuals (N, 6)) with residuals in meters (pos) and
    ROT_WEIGHT_M-scaled radians (rot) per pair.
    """
    ids = sorted(faces)
    idx = {tag: 3 * i for i, tag in enumerate(ids)}
    for i, j, _ in pairs:
        assert i in faces and j in faces, f"pair ({i},{j}) has an undeclared face"

    def transforms(x):
        return {tag: tag_body_transform(faces[tag], half_extents,
                                        x[idx[tag]], x[idx[tag] + 1], x[idx[tag] + 2])
                for tag in ids}

    def residuals(x):
        T = transforms(x)
        out = []
        for i, j, T_meas in pairs:
            T_model = mat_inv(T[i]) @ T[j]
            err = mat_inv(T_model) @ T_meas
            out.append(np.concatenate([
                err[:3, 3],
                ROT_WEIGHT_M * Rotation.from_matrix(err[:3, :3]).as_rotvec()]))
        out.append(PRIOR_WEIGHT * x)
        return np.concatenate(out)

    x0 = np.zeros(3 * len(ids))
    fit = least_squares(residuals, x0)
    assert fit.success, f"tag placement solve failed: {fit.message}"
    solved = {tag: tuple(fit.x[idx[tag]:idx[tag] + 3]) for tag in ids}
    res = fit.fun[:-len(x0)].reshape(-1, 6) if pairs else np.zeros((0, 6))
    return solved, res


def save_sponge_tags(path, half_extents, faces, solved, res):
    tags = {}
    for tag, (u, v, yaw) in solved.items():
        pos, quat = mat_to_pos_quat(
            tag_body_transform(faces[tag], half_extents, u, v, yaw))
        tags[int(tag)] = {
            "face": faces[tag],
            "u_mm": float(u * 1e3), "v_mm": float(v * 1e3),
            "yaw_deg": float(np.degrees(yaw)),
            "pos": pos.tolist(), "quat": quat.tolist(),
        }
    data = {
        "half_extents": [float(h) for h in half_extents],
        "tags": tags,
        "n_pairs": int(len(res)),
        "residual_pos_mm_rms": float(np.sqrt((res[:, :3] ** 2).sum(1).mean()) * 1e3),
        "residual_rot_deg_rms": float(np.degrees(
            np.sqrt((res[:, 3:] ** 2).sum(1).mean()) / ROT_WEIGHT_M)),
    }
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=None, sort_keys=False)


def load_sponge_tags(path=SPONGE_TAGS_PATH):
    """(half_extents (3,), {tag id: T_body_tag (4,4)}) from sponge_tags.yaml."""
    with open(path) as f:
        data = yaml.safe_load(f)
    tags = {int(tag): pos_quat_to_mat(entry["pos"], entry["quat"])
            for tag, entry in data["tags"].items()}
    return np.array(data["half_extents"], dtype=np.float64), tags


def body_pose_from_tag(rvec, tvec, T_body_tag):
    """T_cam_body from one tag's solvePnP pose and its solved body transform."""
    return rt_to_mat(rvec, tvec) @ mat_inv(T_body_tag)


def collect_pairs(frames, family):
    """Capture `frames` frame-pairs from both rig cameras and accumulate all
    same-frame sponge-tag pairs as (id_i, id_j, T_ti_tj)."""
    detector = make_detector(family)
    wanted = set(SPONGE_TAG_IDS) & set(FACE_OF_TAG)
    caps, estimators = {}, {}
    for name in CAMERA_NAMES:
        caps[name], mat, dist = open_rig_camera(name)
        estimators[name] = PoseEstimator(mat, dist)
    pairs = []
    per_tag_seen = {tag: 0 for tag in sorted(wanted)}
    try:
        for k in range(frames):
            for name in CAMERA_NAMES:
                ok, frame = caps[name].read()
                if not ok:
                    raise RuntimeError(f"camera read failed on '{name}'")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                dets = {d.id: d for d in detector.detect(gray) if d.id in wanted}
                for tag in dets:
                    per_tag_seen[tag] += 1
                T = {tag: rt_to_mat(*estimators[name].estimate(d))
                     for tag, d in dets.items()}
                ids = sorted(T)
                for a in range(len(ids)):
                    for b in range(a + 1, len(ids)):
                        pairs.append((ids[a], ids[b],
                                      mat_inv(T[ids[a]]) @ T[ids[b]]))
            if (k + 1) % 10 == 0:
                print(f"  frame {k + 1}/{frames}: {len(pairs)} pairs, "
                      "seen " + " ".join(f"{t}:{n}" for t, n in per_tag_seen.items()),
                      flush=True)
            time.sleep(0.1)  # let the sponge be repositioned between frames
    finally:
        for cap in caps.values():
            cap.release()
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Solve sponge tag placements -> sponge_tags.yaml")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--family", choices=("apriltag", "aruco"), default="apriltag")
    args = parser.parse_args()

    half_extents = load_box_half_extents()
    print(f"box half extents (from {SCENE_XML.name}): {half_extents}")
    pairs = collect_pairs(args.frames, args.family)
    observed = {i for p in pairs for i in p[:2]}
    if not pairs:
        raise RuntimeError("no frame had >= 2 co-visible sponge tags")
    faces = {tag: FACE_OF_TAG[tag] for tag in sorted(observed)}
    print(f"solving {len(faces)} tags from {len(pairs)} pairs: {faces}")
    solved, res = solve_tag_placements(pairs, faces, half_extents)
    for tag, (u, v, yaw) in sorted(solved.items()):
        print(f"  tag {tag} ({faces[tag]}): u={u * 1e3:+.1f} mm  v={v * 1e3:+.1f} mm  "
              f"yaw={np.degrees(yaw):+.1f} deg")
    pos_rms = np.sqrt((res[:, :3] ** 2).sum(1).mean()) * 1e3
    rot_rms = np.degrees(np.sqrt((res[:, 3:] ** 2).sum(1).mean()) / ROT_WEIGHT_M)
    print(f"pair residuals: {pos_rms:.2f} mm RMS, {rot_rms:.2f} deg RMS")
    save_sponge_tags(SPONGE_TAGS_PATH, half_extents, faces, solved, res)
    print(f"wrote {SPONGE_TAGS_PATH}")


if __name__ == "__main__":
    main()
