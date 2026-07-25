"""Solve the sponge tags' in-plane placement on the box -> sponge_tags.yaml.

The shape-tracking dataset (real/tracking/record_shapes.py) needs the sponge's
GT body pose from whatever tag happens to be visible, so every glued tag's
transform to the body frame must be known. Each tag sits on some box face with
three unknowns: in-plane offset (u, v) and yaw about the face normal; the face
plane itself is pinned by the box geometry (half extents read from the sim
scene XML — the single source of the sponge dimensions).

Each tag's *axis* (AXIS_OF_TAG below) must be declared, because it is not
identifiable from the measurements — but its *sign* is solved, because that is
the part a human cannot read off a sponge.

Why the split. Pair evidence fixes the tag configuration only up to a global
rigid transform, and for any three mutually perpendicular faces there is
always a translation solving `t . n_i = h_i - (R p_i) . n_i`. So a fit will
happily slide the whole configuration onto whatever face planes it is given
and drive the residual to zero: naming the wrong axes costs nothing in
residual. Only the physical bound — the printed square has to stay on its
face (`placement_bounds`) — rules any assignment out, and it does not rule out
the x/y swap when the tags sit near their face centers. Declaring the axis is
therefore load-bearing, and it is also the easy half: the three face classes
are 60x40, 60x25 and 40x25 mm, distinguishable with a ruler. The signs are the
hard half (the two 40x25 ends look identical) and are searched over.

Of the 8 sign patterns, the 4 differing from the truth by an even number of
flips fit identically — they are the box's 180-degree rotations, which relabel
axes without moving a face — and are harmless: both the GT center and
`box_sqrtm` are invariant under them. The odd 4 are reflections, not rigid
motions, and lose by orders of magnitude.

How a wrong axis declaration behaves, measured on synthetic data: naming the
wrong axis for the tag on a LARGE face is caught, because the bounds cannot
fit a 60x40-face tag onto a 40x25 one and the fit lands pinned at the edge
with millimetres of residual. Swapping the two NARROW axes (60x25 vs 40x25) is
caught by nothing — it fits to 0.000 mm with entirely plausible offsets — and
it is the error that matters, since it pairs each axis with the wrong half
extent and corrupts every GT shape tensor. Measure those two faces; do not
infer them from the fit.

The solve consumes frames where >= 2 sponge tags are co-visible in one camera:
each such pair gives a measured tag_i -> tag_j relative transform that the
per-tag placements must reproduce through the box. A weak prior toward the
face centers (tags are glued roughly centered) pins any gauge freedom left
when the observed pairs span too few distinct faces. GT body pose then follows
from any single visible tag: T_cam_body = T_cam_tag @ inv(T_body_tag).

Measured pairs are cached to `tag_pairs.npz` so the fit can be re-run offline
(`--from-cache`) without the rig or another capture session.

Run (tags glued, sponge held so several faces are visible; keep turning it
through the capture for view diversity):
    conda run -n mujoco_env python -m real.tracking.tag_body_calib --frames 300
"""
import argparse
import itertools
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from real.calib.extrinsics import mat_inv, mat_to_pos_quat, pos_quat_to_mat, rt_to_mat
from real.marker_spec import SPONGE_TAG_IDS, TAG_SIZE_MM
from real.vision.detect import make_detector
from real.vision.pose import PoseEstimator
from real.vision.stereo_rig import CAMERA_NAMES, open_rig_camera

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCENE_XML = REPO_ROOT / "so101" / "scene_lift.xml"
SPONGE_TAGS_PATH = Path(__file__).resolve().parent / "sponge_tags.yaml"
PAIRS_PATH = Path(__file__).resolve().parent / "tag_pairs.npz"

FACES = ("+x", "-x", "+y", "-y", "+z", "-z")

# Which body axis each glued tag's face is normal to — EDIT to match the
# physical gluing. The axis is the face's size class, so read it off the box:
# z = the two 60x40 mm faces, y = 60x25 mm, x = 40x25 mm (from the cube_geom
# half extents 30/20/12.5 mm). Only the axis is declared; the sign is solved.
AXIS_OF_TAG = {1: "z", 3: "y", 4: "x"}

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


def face_margins(tag, face, half_extents):
    """How far the tag's center may sit from the face center along (u, v)
    before its printed square overhangs an edge. Non-positive on either axis
    means the tag is too big for that face."""
    _, R_face = face_frame(face, half_extents)
    half_tag = TAG_SIZE_MM[tag] / 2e3
    return tuple(half_extents[int(np.argmax(np.abs(R_face[:, col])))] - half_tag
                 for col in (0, 1))


def placement_bounds(faces, half_extents):
    """Per-parameter (lo, hi) arrays keeping every tag on its own face.

    Without these the fit is free to slide a tag off the box to buy a low
    residual, which is exactly how a wrong face assignment disguises itself
    (a real solve put a 20 mm tag 56 mm from the center of a 60 mm face).
    """
    lo, hi = [], []
    for tag in sorted(faces):
        margins = face_margins(tag, faces[tag], half_extents)
        assert min(margins) > 0.0, (
            f"tag {tag} ({TAG_SIZE_MM[tag]} mm) cannot fit on face {faces[tag]}")
        lo.extend([-margins[0], -margins[1], -np.pi])
        hi.extend([margins[0], margins[1], np.pi])
    return np.array(lo), np.array(hi)


def solve_tag_placements(pairs, faces, half_extents):
    """Least-squares (u, v, yaw) per tag from co-visibility pairs.

    `pairs` is a list of (id_i, id_j, T_ti_tj) measured relative transforms
    (tag j in tag i's frame), `faces` maps tag id -> face string. Placements
    are bounded to their face (placement_bounds). Returns
    ({id: (u, v, yaw)}, residuals (N, 6)) with residuals in meters (pos) and
    ROT_WEIGHT_M-scaled radians (rot) per pair.
    """
    assert pairs, "no co-visibility pairs to solve from"
    ids = sorted(faces)
    idx = {tag: 3 * i for i, tag in enumerate(ids)}
    for i, j, _ in pairs:
        assert i in faces and j in faces, f"pair ({i},{j}) has an undeclared face"

    # A handful of tags means only a handful of distinct (i, j) combinations,
    # while pairs number in the hundreds: model each combination once per
    # evaluation and fan it out. Keeps the face search (dozens of fits) cheap.
    combos = sorted({(i, j) for i, j, _ in pairs})
    combo_of_pair = np.array([combos.index((i, j)) for i, j, _ in pairs])
    T_meas = np.stack([T for _, _, T in pairs])

    def transforms(x):
        return {tag: tag_body_transform(faces[tag], half_extents,
                                        x[idx[tag]], x[idx[tag] + 1], x[idx[tag] + 2])
                for tag in ids}

    def residuals(x):
        T = transforms(x)
        T_model = np.stack([mat_inv(T[i]) @ T[j] for i, j in combos])
        err = np.linalg.inv(T_model[combo_of_pair]) @ T_meas
        rot = Rotation.from_matrix(err[:, :3, :3]).as_rotvec()
        return np.concatenate([
            np.concatenate([err[:, :3, 3], ROT_WEIGHT_M * rot], axis=1).ravel(),
            PRIOR_WEIGHT * x])

    x0 = np.zeros(3 * len(ids))
    fit = least_squares(residuals, x0, bounds=placement_bounds(faces, half_extents))
    assert fit.success, f"tag placement solve failed: {fit.message}"
    solved = {tag: tuple(fit.x[idx[tag]:idx[tag] + 3]) for tag in ids}
    return solved, fit.fun[:-len(x0)].reshape(-1, 6)


def residual_rms(res):
    """(position RMS in mm, rotation RMS in degrees) of a pair residual block."""
    pos_mm = float(np.sqrt((res[:, :3] ** 2).sum(1).mean()) * 1e3)
    rot_deg = float(np.degrees(
        np.sqrt((res[:, 3:] ** 2).sum(1).mean()) / ROT_WEIGHT_M))
    return pos_mm, rot_deg


def face_assignments(axis_of_tag):
    """Every sign pattern over the declared per-tag axes.

    The axes come from AXIS_OF_TAG (not identifiable from the data — see the
    module docstring); only the 2^n outward directions are searched.
    """
    tags = sorted(axis_of_tag)
    assert len({axis_of_tag[tag] for tag in tags}) == len(tags), (
        f"two tags share an axis in {axis_of_tag}: they would be on the same "
        "or opposite faces, and opposite faces are never co-visible")
    for signs in itertools.product("+-", repeat=len(tags)):
        yield {tag: sign + axis_of_tag[tag] for tag, sign in zip(tags, signs)}


def solve_faces_and_placements(pairs, axis_of_tag, half_extents):
    """Fit every sign pattern over the declared axes; return them sorted
    best-first as (cost, faces, solved, res). The leading four tie — they are
    the box's rotations — and any of them is a correct body frame."""
    scored = []
    for faces in face_assignments(axis_of_tag):
        solved, res = solve_tag_placements(pairs, faces, half_extents)
        scored.append((float((res ** 2).sum()), faces, solved, res))
    scored.sort(key=lambda entry: entry[0])
    return scored


def save_pairs(path, pairs):
    np.savez(path,
             ids=np.array([[i, j] for i, j, _ in pairs], dtype=np.int64),
             transforms=np.array([T for _, _, T in pairs]))


def load_pairs(path):
    data = np.load(path)
    return [(int(i), int(j), T)
            for (i, j), T in zip(data["ids"], data["transforms"])]


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
    pos_mm, rot_deg = residual_rms(res)
    data = {
        "half_extents": [float(h) for h in half_extents],
        "tags": tags,
        "n_pairs": int(len(res)),
        "residual_pos_mm_rms": pos_mm,
        "residual_rot_deg_rms": rot_deg,
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


def collect_pairs(frames, family, tags):
    """Capture `frames` frame-pairs from both rig cameras and accumulate all
    same-frame sponge-tag pairs as (id_i, id_j, T_ti_tj)."""
    detector = make_detector(family)
    wanted = set(tags)
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
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--family", choices=("apriltag", "aruco"), default="apriltag")
    parser.add_argument("--tags", type=int, nargs="+", default=list(SPONGE_TAG_IDS),
                        help="tag ids glued to the sponge this session")
    parser.add_argument("--from-cache", action="store_true",
                        help=f"re-fit the pairs in {PAIRS_PATH.name} instead of capturing")
    args = parser.parse_args()

    half_extents = load_box_half_extents()
    print(f"box half extents (from {SCENE_XML.name}): {half_extents}")
    if args.from_cache:
        pairs = load_pairs(PAIRS_PATH)
        print(f"loaded {len(pairs)} cached pairs from {PAIRS_PATH.name}")
    else:
        pairs = collect_pairs(args.frames, args.family, args.tags)
        if not pairs:
            raise RuntimeError("no frame had >= 2 co-visible sponge tags")
        save_pairs(PAIRS_PATH, pairs)
    observed = sorted({i for p in pairs for i in p[:2]})
    missing = set(observed) - set(AXIS_OF_TAG)
    assert not missing, f"tags {sorted(missing)} seen but not in AXIS_OF_TAG"
    axes = {tag: AXIS_OF_TAG[tag] for tag in observed}

    print(f"\nsolving signs for declared axes {axes} over {len(pairs)} pairs ...")
    scored = solve_faces_and_placements(pairs, axes, half_extents)
    for cost, faces, solved, res in scored:
        pos_mm, rot_deg = residual_rms(res)
        layout = " ".join(f"{tag}:{faces[tag]}" for tag in observed)
        print(f"  {layout}   {pos_mm:6.2f} mm  {rot_deg:7.2f} deg")

    _, faces, solved, res = scored[0]
    print(f"\nbest assignment: {faces}")
    for tag, (u, v, yaw) in sorted(solved.items()):
        margins = face_margins(tag, faces[tag], half_extents)
        pinned = ("   <-- pinned at the face edge"
                  if min(m - abs(p) for m, p in zip(margins, (u, v))) < 1e-4
                  else "")
        print(f"  tag {tag} ({faces[tag]}): u={u * 1e3:+.1f} mm  v={v * 1e3:+.1f} mm  "
              f"yaw={np.degrees(yaw):+.1f} deg{pinned}")
    pos_rms, rot_rms = residual_rms(res)
    print(f"pair residuals: {pos_rms:.2f} mm RMS, {rot_rms:.2f} deg RMS")
    print("offsets pinned at a face edge mean AXIS_OF_TAG names a face too "
          "small for its tag. Swapping the two narrow axes shows up in neither "
          "the residual nor the offsets — measure those faces.")
    save_sponge_tags(SPONGE_TAGS_PATH, half_extents, faces, solved, res)
    print(f"wrote {SPONGE_TAGS_PATH}")


if __name__ == "__main__":
    main()
