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
axes without moving a face — and are harmless for box-surface GT. The odd 4 are
reflections, not rigid motions, and lose by orders of magnitude.

How a wrong axis declaration behaves, measured on synthetic data: naming the
wrong axis for the tag on a LARGE face is caught, because the bounds cannot
fit a 60x40-face tag onto a 40x25 one and the fit lands pinned at the edge
with millimetres of residual. Swapping the two NARROW axes (60x25 vs 40x25) is
caught by nothing — it fits to 0.000 mm with entirely plausible offsets — and
it is the error that matters, since it pairs each axis with the wrong half
extent and corrupts the GT box surface. Measure those two faces; do not
infer them from the fit.

Capture is placement-based. Move the sponge, release it, and hold still: the
tool waits for stable image corners, rejects tags seen past 60 degrees, then
aggregates a stationary burst from both cameras into one placement. Two
distinct tags must be visible somewhere across the stereo rig, not necessarily
in one camera. Move the sponge again to re-arm capture. The native preview shows
both cameras, accepted/rejected tags, settling progress, pair coverage and each
capture event.

The solver first uses rigid tag-pair transforms to choose the box-face signs,
then jointly minimizes raw corner reprojection error over all cameras,
placements and tags. Each placement gets one sponge pose; tag transforms are
shared globally. A robust first fit rejects corner-level outliers before the
final fit. Captures are cached in `tag_placements.npz` so `--from-cache` can
re-solve without the rig. Every accepted placement is saved atomically; stopping
early and running the normal command again resumes toward the requested total.
Use `--new-session` to archive that cache and start over.

`--validate` captures a separate held-out set in
`tag_validation_placements.npz` and reports how far independently inferred
sponge centers disagree. It never refits or overwrites `sponge_tags.yaml`.

Run:
    conda run -n mujoco_env python -m real.tracking.tag_body_calib --placements 30
    conda run -n mujoco_env python -m real.tracking.tag_body_calib --validate --placements 15
"""
import argparse
from collections import deque
from dataclasses import dataclass
import itertools
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from real.calib.extrinsics import (
    average_transforms,
    mat_inv,
    mat_to_pos_quat,
    pos_quat_to_mat,
    rt_to_mat,
)
from real.calib.table_anchor import TableAnchorTracker
from real.marker_spec import TABLE_TAG_IDS, TAG_SIZE_MM
from real.vision.detect import Detection, make_detector
from real.vision.overlay import (
    GREEN,
    RED,
    TABLE_BLUE,
    WHITE,
    YELLOW,
    OverlayLine,
    StereoViewer,
    TagStyle,
    annotate_tags,
)
from real.vision.pose import PoseEstimator, tag_object_points
from real.vision.stereo_rig import CAMERA_NAMES, open_rig_camera

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCENE_XML = REPO_ROOT / "so101" / "scene_lift.xml"
SPONGE_TAGS_PATH = Path(__file__).resolve().parent / "sponge_tags.yaml"
PLACEMENTS_PATH = Path(__file__).resolve().parent / "tag_placements.npz"
VALIDATION_PLACEMENTS_PATH = Path(__file__).resolve().parent / "tag_validation_placements.npz"

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
# Degrees from a pair's consensus rotation beyond which a measurement is not
# noise but solvePnP's planar-square pose ambiguity: for a near-fronto-parallel
# square the solver can return the pose mirrored about an in-plane axis, which
# lands ~140 deg out. Measured on the rig, good detections sit within 3.5 deg
# at p90 and the flips are ~2% of frames, so anything past this gap is a flip.
# They must be removed rather than absorbed: six 140 deg outliers outweigh
# three hundred 2 deg inliers in a squared-error fit.
PAIR_FLIP_REJECT_DEG = 20.0
# How far the printed square may hang past a face edge. A 20 mm tag on a 25 mm
# face has only +-2.5 mm of play, and the tags are glued to card on soft foam,
# so a slight overhang is ordinary — clamping it instead would bias the solved
# placement. Kept small: the bound is what rules out a face too small to hold
# its tag at all (see the module docstring).
TAG_OVERHANG_ALLOWANCE_M = 0.003

# Calibration capture quality. Unlike the deployment visibility gate, these
# thresholds deliberately favor a smaller, cleaner subset of measurements.
MAX_INCIDENCE_DEG = 60.0
STATIONARY_DWELL_S = 0.75
# The smoke dataset's genuinely-static frame-max is 0.22 px at p95 / 0.82 px
# worst, while moving frames reach 13 px at p90. 2.5 px leaves room for a hand-
# held sponge and detector flicker without admitting ordinary repositioning.
STATIONARY_CORNER_SHIFT_PX = 2.5
REARM_CORNER_SHIFT_PX = 25.0
CAPTURE_FRAMES = 12
MIN_PAIR_PLACEMENTS = 4

# Robust joint-reprojection rejection. Intrinsic calibration itself sits near
# 0.2-0.3 px RMS; a 1 px floor leaves ordinary detector jitter alone while
# still removing a blurred or mirrored observation.
REPROJECTION_OUTLIER_FLOOR_PX = 1.0
REPROJECTION_OUTLIER_MAD_K = 3.5
REPROJECTION_MAX_DROP_FRAC = 0.2

_AXES = {"x": 0, "y": 1, "z": 2}


@dataclass
class StereoPlacement:
    """One settled sponge pose: camera anchors plus median image corners."""

    anchors: dict[str, np.ndarray]
    corners: dict[tuple[str, int], np.ndarray]


@dataclass(frozen=True)
class GateResult:
    state: str
    dwell_s: float
    motion_px: float
    reset_window: bool
    capture: bool


class StationaryPlacementGate:
    """Image-space settle/re-arm state machine for automatic placement capture.

    Input keys are ``(camera_name, tag_id)`` and values are canonical four-corner
    arrays. Exact key continuity is intentional: a detection appearing/disappearing
    restarts the dwell instead of mixing different evidence in one placement.
    """

    def __init__(self, dwell_s=STATIONARY_DWELL_S,
                 still_px=STATIONARY_CORNER_SHIFT_PX,
                 rearm_px=REARM_CORNER_SHIFT_PX,
                 capture_frames=CAPTURE_FRAMES):
        self.dwell_s = float(dwell_s)
        self.still_px = float(still_px)
        self.rearm_px = float(rearm_px)
        self.capture_frames = int(capture_frames)
        self._previous = None
        self._stable_since = None
        self._stable_frames = 0
        self._captured = None
        self._armed = True

    @staticmethod
    def _distinct_tags(corners):
        return {tag for _, tag in corners}

    @staticmethod
    def _motion_px(a, b):
        common = set(a) & set(b)
        if len({tag for _, tag in common}) < 2:
            return np.inf
        # A third view/tag may flicker at the incidence boundary. The median
        # preserves the two-tag rigid-motion signal without letting that one
        # detector toggle reset an otherwise-settled placement.
        return float(np.median([
            np.linalg.norm(a[key] - b[key], axis=1).mean() for key in common
        ]))

    def update(self, now, corners):
        corners = {key: np.asarray(value, dtype=np.float64).copy()
                   for key, value in corners.items()}
        valid = len(self._distinct_tags(corners)) >= 2

        if not self._armed:
            common = set(corners) & set(self._captured)
            moved = any(float(np.linalg.norm(
                corners[key] - self._captured[key], axis=1).mean()) >= self.rearm_px
                        for key in common)
            changed_view = valid and not common
            if moved or changed_view:
                self._armed = True
                self._previous = None
                self._stable_since = None
                self._stable_frames = 0
                return GateResult("settling", 0.0, np.inf, True, False)
            return GateResult("move sponge", 0.0, 0.0, False, False)

        if not valid:
            self._previous = corners
            self._stable_since = None
            self._stable_frames = 0
            return GateResult("need 2 tags", 0.0, np.inf, True, False)

        motion = (np.inf if self._previous is None
                  else self._motion_px(self._previous, corners))
        self._previous = corners
        if motion > self.still_px:
            self._stable_since = float(now)
            self._stable_frames = 1
            return GateResult("hold still", 0.0, motion, True, False)

        if self._stable_since is None:
            self._stable_since = float(now)
            self._stable_frames = 1
        else:
            self._stable_frames += 1
        dwell = float(now) - self._stable_since
        capture = dwell >= self.dwell_s and self._stable_frames >= self.capture_frames
        if capture:
            self._armed = False
            self._captured = corners
            return GateResult("captured", dwell, motion, False, True)
        return GateResult("settling", dwell, motion, False, False)


def incidence_angle_deg(rvec, tvec):
    """Angle between a tag's outward normal and its ray toward the camera."""
    R = Rotation.from_rotvec(np.asarray(rvec, dtype=np.float64)).as_matrix()
    tvec = np.asarray(tvec, dtype=np.float64)
    toward_camera = -tvec / np.linalg.norm(tvec)
    return float(np.degrees(np.arccos(np.clip(R[:, 2] @ toward_camera, -1.0, 1.0))))


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
    """How far the tag's center may sit from the face center along (u, v),
    allowing TAG_OVERHANG_ALLOWANCE_M of the printed square past the edge.
    Non-positive on either axis means the tag is too big for that face."""
    _, R_face = face_frame(face, half_extents)
    half_tag = TAG_SIZE_MM[tag] / 2e3 - TAG_OVERHANG_ALLOWANCE_M
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


def reject_flipped_pairs(pairs, max_deviation_deg=PAIR_FLIP_REJECT_DEG):
    """Drop pair measurements whose relative rotation disagrees with the rest
    of their pair's — see PAIR_FLIP_REJECT_DEG. The tags are rigid on one box,
    so every measurement of a given pair should report the same transform.
    Returns (kept pairs, dropped count)."""
    kept, dropped = [], 0
    for key in sorted({(i, j) for i, j, _ in pairs}):
        group = [p for p in pairs if p[:2] == key]
        R = Rotation.from_matrix(np.stack([T[:3, :3] for _, _, T in group]))
        keep = np.ones(len(group), dtype=bool)
        # Two passes: the first consensus is itself pulled off by the flips.
        for _ in range(2):
            angles = np.degrees(np.linalg.norm(
                (R * R[keep].mean().inv()).as_rotvec(), axis=1))
            keep = angles <= max_deviation_deg
        assert keep.sum() >= 2, (
            f"pair {key}: only {keep.sum()} mutually consistent measurements")
        kept.extend(p for p, k in zip(group, keep) if k)
        dropped += int((~keep).sum())
    return kept, dropped


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


def save_placements(path, placements, camera_matrices, dist_coeffs):
    """Atomically persist settled placements without pickle/object arrays."""
    path = Path(path)
    cameras = tuple(CAMERA_NAMES)
    placement_idx, camera_idx, tag_ids, corners = [], [], [], []
    for p_idx, placement in enumerate(placements):
        for (camera, tag), value in sorted(placement.corners.items()):
            placement_idx.append(p_idx)
            camera_idx.append(cameras.index(camera))
            tag_ids.append(tag)
            corners.append(value)
    anchors = np.stack([
        np.stack([placement.anchors[camera] for camera in cameras])
        for placement in placements
    ])
    temp = path.with_name(f".{path.name}.tmp")
    with open(temp, "wb") as f:
        np.savez(
            f,
            version=np.array(1, dtype=np.int64),
            cameras=np.array(cameras),
            anchors=anchors,
            placement_idx=np.asarray(placement_idx, dtype=np.int64),
            camera_idx=np.asarray(camera_idx, dtype=np.int64),
            tag_ids=np.asarray(tag_ids, dtype=np.int64),
            corners=np.asarray(corners, dtype=np.float64),
            camera_matrices=np.stack([camera_matrices[camera] for camera in cameras]),
            dist_coeffs=np.stack([dist_coeffs[camera] for camera in cameras]),
        )
    temp.replace(path)


def load_placements(path):
    """Return ``(placements, camera_matrices, dist_coeffs)`` from the cache."""
    data = np.load(path, allow_pickle=False)
    assert int(data["version"]) == 1, f"unsupported placement cache version {data['version']}"
    cameras = tuple(str(name) for name in data["cameras"])
    assert cameras == tuple(CAMERA_NAMES), (cameras, CAMERA_NAMES)
    placements = [StereoPlacement(
        anchors={camera: data["anchors"][p_idx, c_idx].copy()
                 for c_idx, camera in enumerate(cameras)},
        corners={})
        for p_idx in range(len(data["anchors"]))]
    for p_idx, c_idx, tag, value in zip(
            data["placement_idx"], data["camera_idx"], data["tag_ids"], data["corners"]):
        placements[int(p_idx)].corners[(cameras[int(c_idx)], int(tag))] = value.copy()
    mats = {camera: data["camera_matrices"][c_idx].copy()
            for c_idx, camera in enumerate(cameras)}
    dists = {camera: data["dist_coeffs"][c_idx].copy()
             for c_idx, camera in enumerate(cameras)}
    return placements, mats, dists


def aggregate_placement(samples, estimators):
    """Median a stationary frame burst into one stereo placement."""
    assert len(samples) >= CAPTURE_FRAMES, len(samples)
    anchors = {
        camera: average_transforms([sample["anchors"][camera] for sample in samples])
        for camera in CAMERA_NAMES
    }
    all_keys = sorted({key for sample in samples for key in sample["corners"]})
    corners = {}
    for key in all_keys:
        values = [sample["corners"][key] for sample in samples if key in sample["corners"]]
        if len(values) < CAPTURE_FRAMES // 2:
            continue
        median = np.median(np.stack(values), axis=0)
        camera, tag = key
        rvec, tvec = estimators[camera].estimate(Detection(tag, median.astype(np.float32)))
        if incidence_angle_deg(rvec, tvec) <= MAX_INCIDENCE_DEG:
            corners[key] = median
    distinct = {tag for _, tag in corners}
    assert len(distinct) >= 2, f"settled burst retained only tags {sorted(distinct)}"
    return StereoPlacement(anchors, corners)


def placement_tag_poses(placement, estimators):
    """Per ``(camera, tag)`` tag transforms in camera and base frames."""
    cam, base = {}, {}
    for (camera, tag), corners in placement.corners.items():
        detection = Detection(tag, corners.astype(np.float32))
        T_cam_tag = rt_to_mat(*estimators[camera].estimate(detection))
        cam[(camera, tag)] = T_cam_tag
        base[(camera, tag)] = placement.anchors[camera] @ T_cam_tag
    return cam, base


def pairs_from_placements(placements, estimators):
    """One rigid pair measurement per tag pair and settled placement.

    Same-camera evidence is preferred because the camera anchor cancels. When
    no camera sees both tags, their base-frame poses connect the two views.
    """
    pairs = []
    for placement in placements:
        cam_poses, base_poses = placement_tag_poses(placement, estimators)
        tags = sorted({tag for _, tag in placement.corners})
        for i, j in itertools.combinations(tags, 2):
            same_camera = [mat_inv(cam_poses[(camera, i)]) @ cam_poses[(camera, j)]
                           for camera in CAMERA_NAMES
                           if (camera, i) in cam_poses and (camera, j) in cam_poses]
            if same_camera:
                measured = average_transforms(same_camera)
            else:
                world_i = average_transforms([
                    pose for (camera, tag), pose in base_poses.items() if tag == i])
                world_j = average_transforms([
                    pose for (camera, tag), pose in base_poses.items() if tag == j])
                measured = mat_inv(world_i) @ world_j
            pairs.append((i, j, measured))
    return pairs


def _pose_disagreement(a, b):
    center_mm = np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1e3
    angle_deg = np.degrees(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())
    return np.array([center_mm, angle_deg])


def validation_errors(placements, camera_matrices, dist_coeffs, tag_transforms):
    """Held-out body-pose disagreement by tag pair and by camera.

    Each observation independently infers ``T_base_body`` through the saved
    ``T_body_tag``. Camera views of one tag are averaged before cross-tag
    comparison, so tag-pair residuals do not overweight a placement merely
    because both cameras saw the same face. Values are ``(center_mm, angle_deg)``.
    """
    estimators = {camera: PoseEstimator(camera_matrices[camera], dist_coeffs[camera])
                  for camera in CAMERA_NAMES}
    cross_tag = {}
    cross_camera = {}
    for placement in placements:
        _, base_poses = placement_tag_poses(placement, estimators)
        observed = {tag for _, tag in base_poses}
        missing = observed - set(tag_transforms)
        assert not missing, f"validation saw unsolved sponge tags {sorted(missing)}"
        body_poses = {
            key: T_base_tag @ mat_inv(tag_transforms[key[1]])
            for key, T_base_tag in base_poses.items()
        }
        per_tag = {
            tag: average_transforms([
                pose for (_, observed_tag), pose in body_poses.items()
                if observed_tag == tag
            ])
            for tag in sorted(observed)
        }
        for i, j in itertools.combinations(sorted(per_tag), 2):
            cross_tag.setdefault((i, j), []).append(
                _pose_disagreement(per_tag[i], per_tag[j]))
        for tag in sorted(observed):
            camera_poses = [body_poses[(camera, tag)] for camera in CAMERA_NAMES
                            if (camera, tag) in body_poses]
            if len(camera_poses) == 2:
                cross_camera.setdefault(tag, []).append(
                    _pose_disagreement(camera_poses[0], camera_poses[1]))
    return ({key: np.asarray(values) for key, values in cross_tag.items()},
            {key: np.asarray(values) for key, values in cross_camera.items()})


def print_validation_report(cross_tag, cross_camera, tags):
    """Print held-out RMS/p95 consistency and require every tag pair."""
    required = set(itertools.combinations(sorted(tags), 2))
    missing = {pair: len(cross_tag.get(pair, ())) for pair in required
               if len(cross_tag.get(pair, ())) < MIN_PAIR_PLACEMENTS}
    if missing:
        raise RuntimeError(
            f"need at least {MIN_PAIR_PLACEMENTS} held-out placements per tag pair; "
            f"insufficient {missing}")

    def line(label, values):
        center = values[:, 0]
        angle = values[:, 1]
        return (f"  {label}: n={len(values):2d}  center "
                f"{np.sqrt(np.mean(center ** 2)):.2f} mm RMS / "
                f"{np.percentile(center, 95):.2f} mm p95; rotation "
                f"{np.sqrt(np.mean(angle ** 2)):.2f} deg RMS")

    print("\nheld-out cross-tag body-center disagreement:")
    for pair in sorted(cross_tag):
        print(line(f"tags {pair[0]}-{pair[1]}", cross_tag[pair]))
    combined = np.concatenate([cross_tag[pair] for pair in sorted(required)])
    print(line("all tag pairs", combined))
    print("cross-camera disagreement for the same tag:")
    if cross_camera:
        for tag in sorted(cross_camera):
            print(line(f"tag {tag}", cross_camera[tag]))
    else:
        print("  no tag was accepted by both cameras in the same placement")
    print("These are internal-consistency errors, not absolute error against an "
          "external metrology reference.")


def pair_counts(pairs):
    return {pair: sum((i, j) == pair for i, j, _ in pairs)
            for pair in sorted({(i, j) for i, j, _ in pairs})}


def require_pair_coverage(pairs, tags):
    counts = pair_counts(pairs)
    required = set(itertools.combinations(sorted(tags), 2))
    missing = {pair: counts.get(pair, 0) for pair in required
               if counts.get(pair, 0) < MIN_PAIR_PLACEMENTS}
    assert not missing, (
        f"need at least {MIN_PAIR_PLACEMENTS} stationary placements per tag pair; "
        f"insufficient {missing}, all counts {counts}")


def _transforms_from_solution(faces, half_extents, solved):
    return {tag: tag_body_transform(faces[tag], half_extents, *solved[tag])
            for tag in sorted(faces)}


def pair_residuals(pairs, faces, half_extents, solved):
    transforms = _transforms_from_solution(faces, half_extents, solved)
    residuals = []
    for i, j, measured in pairs:
        model = mat_inv(transforms[i]) @ transforms[j]
        err = mat_inv(model) @ measured
        residuals.append(np.concatenate([
            err[:3, 3],
            ROT_WEIGHT_M * Rotation.from_matrix(err[:3, :3]).as_rotvec(),
        ]))
    return np.asarray(residuals)


def _initial_body_poses(placements, estimators, tag_transforms):
    poses = []
    for placement in placements:
        _, base_poses = placement_tag_poses(placement, estimators)
        candidates = [T_base_tag @ mat_inv(tag_transforms[tag])
                      for (_, tag), T_base_tag in base_poses.items()]
        poses.append(average_transforms(candidates))
    return poses


def fit_joint_reprojection(placements, faces, half_extents, initial_solved,
                           camera_matrices, dist_coeffs):
    """Fit shared tag placements and one body pose per settled placement."""
    estimators = {camera: PoseEstimator(camera_matrices[camera], dist_coeffs[camera])
                  for camera in CAMERA_NAMES}
    tags = sorted(faces)
    tag_idx = {tag: 3 * index for index, tag in enumerate(tags)}
    initial_tags = _transforms_from_solution(faces, half_extents, initial_solved)
    initial_body = _initial_body_poses(placements, estimators, initial_tags)
    tag_x = np.concatenate([np.asarray(initial_solved[tag]) for tag in tags])
    body_x = np.concatenate([
        np.concatenate([Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]])
        for T in initial_body
    ])
    x0 = np.concatenate([tag_x, body_x])
    tag_lo, tag_hi = placement_bounds(faces, half_extents)
    bounds = (np.concatenate([tag_lo, np.full(len(body_x), -np.inf)]),
              np.concatenate([tag_hi, np.full(len(body_x), np.inf)]))
    observations = [(p_idx, camera, tag, corners)
                    for p_idx, placement in enumerate(placements)
                    for (camera, tag), corners in sorted(placement.corners.items())]

    def unpack(x):
        tag_transforms = {
            tag: tag_body_transform(faces[tag], half_extents,
                                    *x[tag_idx[tag]:tag_idx[tag] + 3])
            for tag in tags
        }
        offset = 3 * len(tags)
        bodies = []
        for p_idx in range(len(placements)):
            values = x[offset + 6 * p_idx:offset + 6 * (p_idx + 1)]
            T = np.eye(4)
            T[:3, :3] = Rotation.from_rotvec(values[:3]).as_matrix()
            T[:3, 3] = values[3:]
            bodies.append(T)
        return tag_transforms, bodies

    def observation_residuals(x):
        tag_transforms, bodies = unpack(x)
        blocks = []
        for p_idx, camera, tag, observed in observations:
            points_tag = tag_object_points(tag)
            points_body = (tag_transforms[tag][:3, :3] @ points_tag.T).T \
                + tag_transforms[tag][:3, 3]
            points_base = (bodies[p_idx][:3, :3] @ points_body.T).T + bodies[p_idx][:3, 3]
            T_cam_base = mat_inv(placements[p_idx].anchors[camera])
            points_cam = (T_cam_base[:3, :3] @ points_base.T).T + T_cam_base[:3, 3]
            projected, _ = cv2.projectPoints(
                points_cam, np.zeros(3), np.zeros(3),
                camera_matrices[camera], dist_coeffs[camera])
            blocks.append((projected.reshape(4, 2) - observed).ravel())
        return np.concatenate(blocks)

    fit = least_squares(observation_residuals, x0, bounds=bounds,
                        loss="soft_l1", f_scale=0.75, x_scale="jac")
    assert fit.success, f"joint stereo reprojection fit failed: {fit.message}"
    tag_transforms, body_poses = unpack(fit.x)
    solved = {}
    for tag in tags:
        start = tag_idx[tag]
        solved[tag] = tuple(fit.x[start:start + 3])
    flat = observation_residuals(fit.x).reshape(len(observations), 4, 2)
    errors = np.sqrt(np.mean(np.sum(np.square(flat), axis=2), axis=1))
    return solved, body_poses, observations, errors


def reject_reprojection_outliers(placements, observations, errors):
    """Drop bad camera/tag observations; discard placements left underconstrained."""
    median = float(np.median(errors))
    mad = float(np.median(np.abs(errors - median)))
    cutoff = max(REPROJECTION_OUTLIER_FLOOR_PX,
                 median + REPROJECTION_OUTLIER_MAD_K * 1.4826 * mad)
    bad = {(p_idx, camera, tag)
           for (p_idx, camera, tag, _), error in zip(observations, errors)
           if error > cutoff}
    if len(bad) > REPROJECTION_MAX_DROP_FRAC * len(observations):
        raise RuntimeError(
            f"reprojection rejection wants to drop {len(bad)}/{len(observations)} "
            f"observations (>{REPROJECTION_MAX_DROP_FRAC:.0%}); capture is suspect")
    filtered = []
    dropped_placements = 0
    for p_idx, placement in enumerate(placements):
        corners = {key: value for key, value in placement.corners.items()
                   if (p_idx, key[0], key[1]) not in bad}
        if len({tag for _, tag in corners}) < 2:
            dropped_placements += 1
            continue
        filtered.append(StereoPlacement(placement.anchors, corners))
    return filtered, len(bad), dropped_placements, cutoff


def save_sponge_tags(path, half_extents, faces, solved, res, calibration_meta=None):
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
    if calibration_meta is not None:
        data.update(calibration_meta)
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


def _show_capture_view(viewer, frames, detections, accepted, angles, anchors, result,
                       n_placements, target_placements, counts, delay_ms):
    annotated = {}
    camera_lines = {}
    for camera in CAMERA_NAMES:
        styles = {}
        for tag in detections[camera]:
            if tag in TABLE_TAG_IDS:
                styles[tag] = TagStyle(f"table {tag}", TABLE_BLUE, 3)
            elif (camera, tag) in accepted:
                styles[tag] = TagStyle(f"tag {tag} {angles[camera][tag]:.0f}deg OK",
                                       GREEN, 3)
            else:
                styles[tag] = TagStyle(
                    f"tag {tag} {angles[camera][tag]:.0f}deg > {MAX_INCIDENCE_DEG:.0f}",
                    RED, 3)
        annotated[camera] = annotate_tags(
            frames[camera].copy(), detections[camera].values(), styles)
        anchor_ok = camera in anchors
        camera_lines[camera] = [OverlayLine(
            f"{camera}  table={'OK' if anchor_ok else 'MISSING'}",
            GREEN if anchor_ok else RED)]
    state_color = (GREEN if result.state.startswith("captured")
                   else YELLOW if result.state == "settling" else RED)
    coverage = "  ".join(f"{i}-{j}:{n}" for (i, j), n in sorted(counts.items())) or "none"
    header_lines = [
        OverlayLine(
            f"{result.state.upper()}  motion {result.motion_px:.2f}/"
            f"{STATIONARY_CORNER_SHIFT_PX:.2f}px  "
            f"dwell {result.dwell_s:.2f}/{STATIONARY_DWELL_S:.2f}s  "
            f"placements {n_placements}/{target_placements}", state_color),
        OverlayLine(f"pair coverage: {coverage}", WHITE),
        OverlayLine("Move after capture  |  ENTER/q/ESC save and stop", WHITE),
    ]
    return viewer.show(annotated, camera_lines, header_lines, delay_ms)


def collect_placements(target_placements, family, tags, gui=True,
                       initial_placements=(), cached_mats=None, cached_dists=None,
                       cache_path=PLACEMENTS_PATH):
    """Auto-capture distinct stationary placements from the complete stereo rig."""
    detector = make_detector(family)
    wanted = set(tags) | set(TABLE_TAG_IDS)
    caps, mats, dists, estimators, anchor_trackers = {}, {}, {}, {}, {}
    for camera in CAMERA_NAMES:
        caps[camera], mats[camera], dists[camera] = open_rig_camera(camera)
        estimators[camera] = PoseEstimator(mats[camera], dists[camera])
        anchor_trackers[camera] = TableAnchorTracker(mats[camera], dists[camera])
    if cached_mats is not None:
        for camera in CAMERA_NAMES:
            if not np.array_equal(mats[camera], cached_mats[camera]):
                raise RuntimeError(f"{camera} intrinsics changed since placement cache")
            if not np.array_equal(dists[camera], cached_dists[camera]):
                raise RuntimeError(f"{camera} distortion changed since placement cache")

    gate = StationaryPlacementGate()
    recent = deque(maxlen=CAPTURE_FRAMES)
    placements = list(initial_placements)
    counts = pair_counts(pairs_from_placements(placements, estimators)) if placements else {}
    last_print = 0.0
    capture_flash_until = 0.0
    viewer = StereoViewer("sponge stereo tag calibration") if gui else None
    print("Move the sponge to a new pose and release it. Capture is automatic after "
          f"{STATIONARY_DWELL_S:.2f}s still with >=2 tags across the rig at "
          f"<= {MAX_INCIDENCE_DEG:.0f} degrees. Move it clearly after each capture.")
    try:
        while len(placements) < target_placements:
            frames, detections, angles, anchors, accepted = {}, {}, {}, {}, {}
            for camera in CAMERA_NAMES:
                ok, frame = caps[camera].read()
                if not ok:
                    raise RuntimeError(f"camera read failed on '{camera}'")
                frames[camera] = frame
                found = {d.id: d for d in detector.detect(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) if d.id in wanted}
                detections[camera] = found
                angles[camera] = {}
                anchor_trackers[camera].observe(found)
                anchor = anchor_trackers[camera].value()
                if anchor is not None:
                    anchors[camera] = anchor
                for tag, detection in found.items():
                    if tag in TABLE_TAG_IDS:
                        continue
                    rvec, tvec = estimators[camera].estimate(detection)
                    angle = incidence_angle_deg(rvec, tvec)
                    angles[camera][tag] = angle
                    if angle <= MAX_INCIDENCE_DEG:
                        accepted[(camera, tag)] = detection.corners.copy()

            gate_input = accepted if set(anchors) == set(CAMERA_NAMES) else {}
            now = time.monotonic()
            result = gate.update(now, gate_input)
            if result.reset_window:
                recent.clear()
            if len({tag for _, tag in gate_input}) >= 2:
                recent.append({
                    "anchors": {camera: anchors[camera].copy() for camera in CAMERA_NAMES},
                    "corners": {key: value.copy() for key, value in accepted.items()},
                })
            if result.capture:
                assert len(recent) == CAPTURE_FRAMES, len(recent)
                placement = aggregate_placement(list(recent), estimators)
                placements.append(placement)
                counts = pair_counts(pairs_from_placements(placements, estimators))
                save_placements(cache_path, placements, mats, dists)
                print(f"CAPTURED placement {len(placements)}/{target_placements}: "
                      f"tags {sorted({tag for _, tag in placement.corners})}, "
                      f"pair coverage {counts}; saved {cache_path.name}", flush=True)
                capture_flash_until = now + 1.0

            if gui:
                display_result = result
                if now < capture_flash_until and not result.capture:
                    display_result = GateResult(
                        f"captured #{len(placements)}", result.dwell_s,
                        result.motion_px, False, False)
                final_capture = result.capture and len(placements) == target_placements
                delay_ms = 1000 if final_capture else 1
                key = _show_capture_view(
                    viewer, frames, detections, accepted, angles, anchors,
                    display_result, len(placements), target_placements, counts, delay_ms)
                if key in (13, 27, ord("q")):
                    break
            elif now - last_print >= 1.0:
                print(f"{result.state}: dwell {result.dwell_s:.2f}s, "
                      f"placements {len(placements)}/{target_placements}", flush=True)
                last_print = now
    finally:
        for cap in caps.values():
            cap.release()
        if viewer is not None:
            viewer.close()
        detector.close()
    complete = len(placements) >= target_placements
    return placements, mats, dists, complete


def main():
    parser = argparse.ArgumentParser(
        description="Solve sponge tag placements -> sponge_tags.yaml")
    parser.add_argument("--placements", type=int, default=30,
                        help="distinct settled sponge poses to auto-capture")
    parser.add_argument("--family", choices=("apriltag", "aruco"), default="apriltag")
    parser.add_argument("--tags", type=int, nargs="+", default=sorted(AXIS_OF_TAG),
                        help="tag ids glued to the sponge this session")
    parser.add_argument("--from-cache", action="store_true",
                        help=f"re-fit {PLACEMENTS_PATH.name} instead of capturing")
    parser.add_argument("--validate", action="store_true",
                        help="capture held-out placements and report center consistency")
    parser.add_argument("--new-session", action="store_true",
                        help="archive any placement cache and start from zero")
    parser.add_argument("--no-gui", action="store_true",
                        help="disable the annotated stereo preview")
    args = parser.parse_args()
    assert not (args.from_cache and (args.new_session or args.validate)), \
        "--from-cache is mutually exclusive with --new-session and --validate"

    half_extents = load_box_half_extents()
    print(f"box half extents (from {SCENE_XML.name}): {half_extents}")
    if args.from_cache:
        placements, mats, dists = load_placements(PLACEMENTS_PATH)
        print(f"loaded {len(placements)} settled placements from {PLACEMENTS_PATH.name}")
    else:
        cache_path = VALIDATION_PLACEMENTS_PATH if args.validate else PLACEMENTS_PATH
        session_kind = "validation" if args.validate else "calibration"
        if args.new_session and cache_path.exists():
            stamp = time.strftime("%Y%m%d_%H%M%S")
            backup = cache_path.with_name(f"{cache_path.stem}_{stamp}.npz")
            cache_path.replace(backup)
            print(f"archived previous placement cache -> {backup}")
        if cache_path.exists():
            placements, cached_mats, cached_dists = load_placements(cache_path)
            print(f"resuming {len(placements)}/{args.placements} placements from "
                  f"{cache_path.name}")
        else:
            placements, cached_mats, cached_dists = [], None, None
        if len(placements) < args.placements:
            placements, mats, dists, complete = collect_placements(
                args.placements, args.family, args.tags, gui=not args.no_gui,
                initial_placements=placements, cached_mats=cached_mats,
                cached_dists=cached_dists, cache_path=cache_path)
        else:
            mats, dists, complete = cached_mats, cached_dists, True
        if not complete:
            if placements:
                print(f"saved partial {session_kind}: {len(placements)}/{args.placements} "
                      f"placements in {cache_path}; run the same command to continue")
            else:
                print("stopped before the first placement was captured; nothing saved")
            return

    if args.validate:
        _, tag_transforms = load_sponge_tags()
        cross_tag, cross_camera = validation_errors(
            placements, mats, dists, tag_transforms)
        print_validation_report(cross_tag, cross_camera, args.tags)
        return

    estimators = {camera: PoseEstimator(mats[camera], dists[camera])
                  for camera in CAMERA_NAMES}
    pairs = pairs_from_placements(placements, estimators)
    require_pair_coverage(pairs, args.tags)
    pairs, dropped = reject_flipped_pairs(pairs)
    print(f"dropped {dropped} flipped measurements, {len(pairs)} pairs remain")
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

    _, faces, initial_solved, _ = scored[0]
    print("\nrefining all tag placements + per-placement body poses from raw "
          "stereo corner reprojection ...")
    solved, _, observations, errors = fit_joint_reprojection(
        placements, faces, half_extents, initial_solved, mats, dists)
    filtered, dropped_obs, dropped_placements, cutoff = reject_reprojection_outliers(
        placements, observations, errors)
    if dropped_obs or dropped_placements:
        print(f"reprojection gate: dropped {dropped_obs}/{len(observations)} tag views "
              f"and {dropped_placements} underconstrained placements at {cutoff:.2f} px")
        filtered_pairs = pairs_from_placements(filtered, estimators)
        require_pair_coverage(filtered_pairs, args.tags)
        solved, _, observations, errors = fit_joint_reprojection(
            filtered, faces, half_extents, solved, mats, dists)
        placements = filtered
        pairs = filtered_pairs
    res = pair_residuals(pairs, faces, half_extents, solved)
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
    reprojection_rms = float(np.sqrt(np.mean(np.square(errors))))
    print(f"joint corner reprojection: {reprojection_rms:.3f} px RMS, "
          f"{np.percentile(errors, 95):.3f} px observation p95")
    print("a placement pinned at its bound means AXIS_OF_TAG names a face too "
          "small for its tag. Swapping the two narrow axes shows up in neither "
          "the residual nor the offsets — measure those faces.")
    save_sponge_tags(SPONGE_TAGS_PATH, half_extents, faces, solved, res, {
        "n_placements": int(len(placements)),
        "n_camera_tag_observations": int(len(observations)),
        "reprojection_px_rms": reprojection_rms,
        "reprojection_px_p95": float(np.percentile(errors, 95)),
        "max_incidence_deg": MAX_INCIDENCE_DEG,
    })
    print(f"wrote {SPONGE_TAGS_PATH}")


if __name__ == "__main__":
    main()
