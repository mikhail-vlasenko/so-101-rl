"""Synthetic-data tests for the sponge tag placement solver
(real/tracking/tag_body_calib.py): known (u, v, yaw) placements generate
pairwise relative-transform measurements; the solver must recover them."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from real.calib.extrinsics import mat_inv
from real.marker_spec import TAG_SIZE_MM
from real.tracking.tag_body_calib import (
    GateResult,
    StationaryPlacementGate,
    StereoPlacement,
    TAG_OVERHANG_ALLOWANCE_M,
    face_assignments,
    face_frame,
    face_margins,
    fit_joint_reprojection,
    incidence_angle_deg,
    load_placements,
    pairs_from_placements,
    reject_flipped_pairs,
    reject_reprojection_outliers,
    residual_rms,
    save_placements,
    solve_faces_and_placements,
    solve_tag_placements,
    tag_body_transform,
    validation_errors,
)
from real.vision.pose import PoseEstimator, tag_object_points

HALF = np.array([0.03, 0.02, 0.0125])

FACES = {1: "+z", 3: "-z", 4: "+y", 5: "-y", 6: "+x", 7: "-x"}

# The box's rotation group: identity plus a half turn about each axis. These
# relabel the face names without moving any physical face, so the face search
# cannot (and need not) distinguish between them.
BOX_SYMMETRIES = (np.diag([1, 1, 1]), np.diag([1, -1, -1]),
                  np.diag([-1, 1, -1]), np.diag([-1, -1, 1]))


def _relabel(faces, S):
    """`faces` as named after applying box symmetry `S`."""
    out = {}
    for tag, face in faces.items():
        k = "xyz".index(face[1])
        sign = (1 if face[0] == "+" else -1) * int(S[k, k])
        out[tag] = ("+" if sign > 0 else "-") + face[1]
    return out


def _symmetry_orbit(faces):
    return {frozenset(_relabel(faces, S).items()) for S in BOX_SYMMETRIES}


def test_face_frames_are_right_handed_outward():
    for face in ("+x", "-x", "+y", "-y", "+z", "-z"):
        origin, R = face_frame(face, HALF)
        a, b, n = R[:, 0], R[:, 1], R[:, 2]
        np.testing.assert_allclose(np.cross(a, b), n, atol=1e-12)
        # Origin sits at the face center: along the normal at the half extent.
        np.testing.assert_allclose(origin @ n, np.abs(origin).max(), atol=1e-12)
        assert origin @ n > 0.0  # outward


def test_tag_body_transform_geometry():
    """A tag's +z is the outward face normal; (u, v) moves it in-plane."""
    T = tag_body_transform("+z", HALF, u=0.005, v=-0.003, yaw=0.4)
    np.testing.assert_allclose(T[:3, 2], [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], [0.005, -0.003, HALF[2]], atol=1e-12)
    T = tag_body_transform("-x", HALF, u=0.0, v=0.0, yaw=0.0)
    np.testing.assert_allclose(T[:3, 2], [-1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], [-HALF[0], 0.0, 0.0], atol=1e-12)


def _synthetic_pairs(rng, truth, n_pairs, noise_pos=0.0, noise_rot=0.0,
                     faces=FACES):
    """Random co-visibility pairs with optional measurement noise."""
    T = {tag: tag_body_transform(faces[tag], HALF, *params)
         for tag, params in truth.items()}
    ids = sorted(truth)
    pairs = []
    for _ in range(n_pairs):
        i, j = rng.choice(ids, size=2, replace=False)
        T_meas = mat_inv(T[i]) @ T[j]
        if noise_pos > 0.0:
            T_noise = np.eye(4)
            T_noise[:3, :3] = Rotation.from_rotvec(
                rng.normal(0, noise_rot, 3)).as_matrix()
            T_noise[:3, 3] = rng.normal(0, noise_pos, 3)
            T_meas = T_meas @ T_noise
        pairs.append((int(i), int(j), T_meas))
    return pairs


def _random_truth(rng, faces=FACES):
    """Physically realizable placements: a glued tag stays within its face's
    margins (which allow a small overhang), as the solver's bounds require."""
    truth = {}
    for tag, face in faces.items():
        margin_u, margin_v = face_margins(tag, face, HALF)
        truth[tag] = (rng.uniform(-0.8 * margin_u, 0.8 * margin_u),
                      rng.uniform(-0.8 * margin_v, 0.8 * margin_v),
                      rng.uniform(-0.3, 0.3))
    return truth


def test_solver_recovers_exact_placements():
    rng = np.random.default_rng(0)
    truth = _random_truth(rng)
    pairs = _synthetic_pairs(rng, truth, n_pairs=60)
    solved, res = solve_tag_placements(pairs, FACES, HALF)
    for tag, (u, v, yaw) in truth.items():
        su, sv, syaw = solved[tag]
        assert abs(su - u) < 1e-4, tag
        assert abs(sv - v) < 1e-4, tag
        assert abs(syaw - yaw) < 1e-3, tag
    assert np.sqrt((res[:, :3] ** 2).sum(1).mean()) < 1e-4


def test_solver_tolerates_measurement_noise():
    """With ~1 mm / ~0.5 deg pair noise the recovered placements stay within
    a millimeter and a degree — the accuracy the GT pipeline needs."""
    rng = np.random.default_rng(1)
    truth = _random_truth(rng)
    pairs = _synthetic_pairs(rng, truth, n_pairs=300,
                             noise_pos=0.001, noise_rot=0.01)
    solved, _ = solve_tag_placements(pairs, FACES, HALF)
    for tag, (u, v, yaw) in truth.items():
        su, sv, syaw = solved[tag]
        assert abs(su - u) < 1e-3, tag
        assert abs(sv - v) < 1e-3, tag
        assert abs(syaw - yaw) < np.radians(1.0), tag


def test_solver_transform_roundtrip():
    """The full T_body_tag rebuilt from solved params maps tag-frame points to
    the same body-frame points as the ground truth."""
    rng = np.random.default_rng(2)
    truth = _random_truth(rng)
    pairs = _synthetic_pairs(rng, truth, n_pairs=80)
    solved, _ = solve_tag_placements(pairs, FACES, HALF)
    probe = np.array([0.01, -0.01, 0.0, 1.0])
    for tag in truth:
        T_true = tag_body_transform(FACES[tag], HALF, *truth[tag])
        T_est = tag_body_transform(FACES[tag], HALF, *solved[tag])
        np.testing.assert_allclose(T_est @ probe, T_true @ probe, atol=2e-4)


def test_face_assignments_search_signs_only():
    """Axes are declared (unidentifiable from pair data); only the outward
    directions are searched."""
    axes = {1: "z", 3: "y", 4: "x"}
    combos = list(face_assignments(axes))
    assert len(combos) == 8
    assert all({f[1] for f in faces.values()} == {"x", "y", "z"} for faces in combos)
    assert len({frozenset(faces.items()) for faces in combos}) == 8


def test_face_assignments_reject_two_tags_on_one_axis():
    with pytest.raises(AssertionError):
        list(face_assignments({1: "z", 3: "z"}))


def test_swapping_the_narrow_axes_is_invisible_but_the_large_face_is_not():
    """Why AXIS_OF_TAG is declared rather than fitted. Pair evidence fixes the
    configuration only up to a global rigid transform, and three mutually
    perpendicular faces always admit one — so a wrong axis costs no residual.
    Only the on-face bound can object, and it objects only when the named face
    is too small for its tag. Deterministic: tags glued a few mm off center.
    """
    truth_faces = {1: "+z", 3: "+y", 4: "+x"}
    truth = {1: (0.004, -0.002, 0.2), 3: (0.001, 0.006, -0.1),
             4: (-0.003, 0.001, 0.3)}
    T = {tag: tag_body_transform(truth_faces[tag], HALF, *truth[tag])
         for tag in truth}
    ids = sorted(T)
    pairs = [(i, j, mat_inv(T[i]) @ T[j]) for i in ids for j in ids if i < j] * 40

    # The two narrow faces exchanged: fits exactly, and every tag sits well
    # inside its face, so nothing at all flags the error.
    solved, res = solve_tag_placements(pairs, {1: "+z", 3: "+x", 4: "-y"}, HALF)
    assert residual_rms(res)[0] < 0.01
    for tag, (u, v, _) in solved.items():
        margins = face_margins(tag, {1: "+z", 3: "+x", 4: "-y"}[tag], HALF)
        assert abs(u) < margins[0] and abs(v) < margins[1]

    # The large-face tag moved onto a 40x25 face: the bound cannot absorb it.
    _, res = solve_tag_placements(pairs, {1: "+x", 3: "+z", 4: "+y"}, HALF)
    assert residual_rms(res)[0] > 1.0


def test_face_margins_shrink_by_the_printed_tag():
    """A 20 mm tag, less the 3 mm overhang allowance, leaves 13 mm of play on
    the 60 mm axis but only 5.5 mm on the 25 mm one."""
    half_tag = TAG_SIZE_MM[1] / 2e3 - TAG_OVERHANG_ALLOWANCE_M
    np.testing.assert_allclose(face_margins(1, "+z", HALF),
                               (HALF[0] - half_tag, HALF[1] - half_tag))
    # +y's in-plane axes are (z, x): the tight bound lands on u, not v.
    np.testing.assert_allclose(face_margins(1, "+y", HALF),
                               (HALF[2] - half_tag, HALF[0] - half_tag))
    np.testing.assert_allclose(face_margins(1, "+x", HALF),
                               (HALF[1] - half_tag, HALF[2] - half_tag))


def test_sign_search_recovers_the_gluing_up_to_box_symmetry():
    """Given the axes, the sign search must land on the true gluing or one of
    its three rotational images, and the reflected patterns must lose big."""
    rng = np.random.default_rng(4)
    truth_faces = {1: "+z", 3: "+y", 4: "-x"}
    truth = _random_truth(rng, truth_faces)
    pairs = _synthetic_pairs(rng, truth, n_pairs=200, noise_pos=0.0005,
                             noise_rot=0.005, faces=truth_faces)

    axes = {tag: face[1] for tag, face in truth_faces.items()}
    scored = solve_faces_and_placements(pairs, axes, HALF)
    orbit = _symmetry_orbit(truth_faces)
    assert frozenset(scored[0][1].items()) in orbit
    # The four rotational images fit equally well and sort to the front; the
    # four reflections are not rigid motions and cannot fit at all.
    assert {frozenset(faces.items()) for _, faces, _, _ in scored[:4]} == orbit
    assert scored[4][0] > 100.0 * scored[3][0]


def test_symmetry_images_agree_on_the_gt_body_pose_up_to_a_half_turn():
    """Why the tie is harmless: the images differ by a 180-degree body rotation,
    which leaves the sponge's center and its box second moment unchanged."""
    rng = np.random.default_rng(5)
    truth_faces = {1: "+z", 3: "+y", 4: "+x"}
    truth = _random_truth(rng, truth_faces)
    pairs = _synthetic_pairs(rng, truth, n_pairs=120, faces=truth_faces)
    axes = {tag: face[1] for tag, face in truth_faces.items()}
    scored = solve_faces_and_placements(pairs, axes, HALF)

    # Each image implies a body frame; a tag's pose in it differs only by the
    # symmetry rotation, so the box's second moment matrix is identical.
    moments = []
    for _, faces, solved, _ in scored[:4]:
        T = tag_body_transform(faces[1], HALF, *solved[1])
        R = mat_inv(T)[:3, :3]          # body axes as seen from tag 1
        moments.append((R * HALF ** 2) @ R.T)
    for M in moments[1:]:
        # atol is the fit's convergence floor, four orders below the moments.
        np.testing.assert_allclose(M, moments[0], atol=1e-7)


def test_reject_flipped_pairs_drops_mirrored_solvepnp_solutions():
    """On the rig 2% of detections came back mirrored ~140 deg out, and in a
    squared-error fit those few outweighed the hundreds of good ones."""
    rng = np.random.default_rng(9)
    truth_faces = {1: "+z", 3: "+y", 4: "+x"}
    truth = _random_truth(rng, truth_faces)
    pairs = _synthetic_pairs(rng, truth, n_pairs=90, noise_pos=0.0005,
                             noise_rot=0.01, faces=truth_faces)
    flip = Rotation.from_rotvec(np.radians(140.0) * np.array([1.0, 0.0, 0.0]))
    corrupted = {5, 23, 61}
    for k in corrupted:
        i, j, T = pairs[k]
        T = T.copy()
        T[:3, :3] = T[:3, :3] @ flip.as_matrix()
        pairs[k] = (i, j, T)

    kept, dropped = reject_flipped_pairs(pairs)
    assert dropped == len(corrupted)
    assert len(kept) == len(pairs) - len(corrupted)
    for k in corrupted:
        assert not any(np.array_equal(T, pairs[k][2]) for _, _, T in kept)


def test_solver_rejects_undeclared_tag():
    rng = np.random.default_rng(3)
    truth = _random_truth(rng)
    pairs = _synthetic_pairs(rng, truth, n_pairs=10)
    with pytest.raises(AssertionError):
        solve_tag_placements(pairs, {1: "+z"}, HALF)


def _corner_map(shift=0.0):
    square = np.array([[0.0, 0.0], [10.0, 0.0],
                       [10.0, 10.0], [0.0, 10.0]])
    return {("main", 1): square + shift,
            ("aux", 3): square + np.array([30.0 + shift, 0.0])}


def test_stationary_gate_captures_once_then_requires_real_movement():
    gate = StationaryPlacementGate(dwell_s=0.2, still_px=0.5,
                                   rearm_px=10.0, capture_frames=3)
    assert gate.update(0.0, _corner_map()).state == "hold still"
    assert gate.update(0.1, _corner_map(0.1)).state == "settling"
    captured = gate.update(0.21, _corner_map(0.2))
    assert captured == GateResult("captured", pytest.approx(0.21),
                                  pytest.approx(0.12071, abs=1e-5), False, True)
    assert gate.update(0.3, _corner_map(0.2)).state == "move sponge"
    rearmed = gate.update(0.4, _corner_map(20.0))
    assert rearmed.reset_window
    assert rearmed.state == "settling"


def test_stationary_gate_needs_two_distinct_tags_not_two_camera_views():
    square = _corner_map()[("main", 1)]
    gate = StationaryPlacementGate(dwell_s=0.0, capture_frames=1)
    result = gate.update(0.0, {("main", 1): square, ("aux", 1): square})
    assert result.state == "need 2 tags"


def test_stationary_gate_ignores_a_flickering_third_tag():
    first = _corner_map()
    first[("main", 4)] = first[("main", 1)] + 60.0
    gate = StationaryPlacementGate(dwell_s=1.0, still_px=0.5, capture_frames=2)
    assert gate.update(0.0, first).state == "hold still"
    result = gate.update(0.1, _corner_map(0.1))
    assert result.state == "settling"
    assert result.motion_px == pytest.approx(0.12071, abs=1e-5)


def test_incidence_angle_uses_outward_normal_toward_camera():
    front = Rotation.from_euler("x", 180.0, degrees=True)
    assert incidence_angle_deg(front.as_rotvec(), [0.0, 0.0, 1.0]) == pytest.approx(0.0)
    tilted = Rotation.from_euler("y", 60.0, degrees=True) * front
    assert incidence_angle_deg(tilted.as_rotvec(), [0.0, 0.0, 1.0]) == pytest.approx(60.0)


def _synthetic_stereo_placements():
    K = np.array([[800.0, 0.0, 320.0],
                  [0.0, 800.0, 240.0],
                  [0.0, 0.0, 1.0]])
    mats = {camera: K.copy() for camera in ("main", "aux")}
    dists = {camera: np.zeros(5) for camera in ("main", "aux")}
    anchors = {"main": np.eye(4), "aux": np.eye(4)}
    anchors["aux"][:3, 3] = [0.12, 0.0, 0.0]
    faces = {1: "-z", 3: "+y", 4: "+x"}
    truth = {1: (0.003, -0.002, 0.08),
             3: (0.001, 0.004, -0.12),
             4: (-0.002, 0.001, 0.16)}
    tag_transforms = {tag: tag_body_transform(faces[tag], HALF, *truth[tag])
                      for tag in truth}
    placements = []
    for index in range(8):
        T_base_body = np.eye(4)
        T_base_body[:3, :3] = Rotation.from_euler(
            "xyz", [2.0 * index, -1.5 * index, 3.0 * index], degrees=True).as_matrix()
        T_base_body[:3, 3] = [-0.025 + 0.008 * index,
                              -0.015 + 0.004 * index, 0.48 + 0.005 * index]
        corners = {}
        for camera, tag in (("main", 1), ("aux", 3),
                            (("main" if index % 2 == 0 else "aux"), 4)):
            points_body = (tag_transforms[tag][:3, :3] @ tag_object_points(tag).T).T \
                + tag_transforms[tag][:3, 3]
            points_base = (T_base_body[:3, :3] @ points_body.T).T + T_base_body[:3, 3]
            T_cam_base = mat_inv(anchors[camera])
            points_cam = (T_cam_base[:3, :3] @ points_base.T).T + T_cam_base[:3, 3]
            pixels = (points_cam[:, :2] / points_cam[:, 2, None]) * 800.0
            pixels += np.array([320.0, 240.0])
            corners[(camera, tag)] = pixels
        placements.append(StereoPlacement(
            {camera: value.copy() for camera, value in anchors.items()}, corners))
    return placements, mats, dists, faces, truth


def test_cross_camera_tags_form_pair_without_same_camera_covisibility():
    _, _, _, faces, truth = _synthetic_stereo_placements()
    transforms = {tag: tag_body_transform(faces[tag], HALF, *truth[tag])
                  for tag in truth}
    T_base_body = np.eye(4)
    T_base_body[:3, 3] = [0.02, -0.01, 0.4]
    anchors = {"main": np.eye(4), "aux": np.eye(4)}
    anchors["aux"][:3, 3] = [0.12, 0.0, 0.0]
    camera_tag = {
        ("main", 1): mat_inv(anchors["main"]) @ T_base_body @ transforms[1],
        ("aux", 3): mat_inv(anchors["aux"]) @ T_base_body @ transforms[3],
    }

    class FakeEstimator:
        def __init__(self, camera):
            self.camera = camera

        def estimate(self, detection):
            T = camera_tag[(self.camera, detection.id)]
            return Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]

    square = np.zeros((4, 2))
    placement = StereoPlacement(anchors, {("main", 1): square, ("aux", 3): square})
    estimators = {camera: FakeEstimator(camera) for camera in anchors}
    pairs = pairs_from_placements([placement], estimators)
    assert len(pairs) == 1
    i, j, measured = pairs[0]
    assert (i, j) == (1, 3)
    np.testing.assert_allclose(measured, mat_inv(transforms[i]) @ transforms[j],
                               atol=1e-12)


def test_joint_stereo_reprojection_recovers_tag_placements():
    placements, mats, dists, faces, truth = _synthetic_stereo_placements()
    initial = {tag: (u + 0.0005, v - 0.0004, yaw + 0.015)
               for tag, (u, v, yaw) in truth.items()}
    solved, _, _, errors = fit_joint_reprojection(
        placements, faces, HALF, initial, mats, dists)
    for tag in truth:
        np.testing.assert_allclose(solved[tag], truth[tag], atol=2e-5)
    assert errors.max() < 1e-4


def test_held_out_validation_recovers_cross_tag_consistency():
    placements, mats, dists, faces, truth = _synthetic_stereo_placements()
    transforms = {tag: tag_body_transform(faces[tag], HALF, *truth[tag])
                  for tag in truth}
    cross_tag, cross_camera = validation_errors(placements, mats, dists, transforms)
    assert set(cross_tag) == {(1, 3), (1, 4), (3, 4)}
    # Independent planar solvePnP has depth ambiguity even for exact projected
    # corners; the validator intentionally includes that deployment-path error.
    assert max(values[:, 0].max() for values in cross_tag.values()) < 2.0
    assert cross_camera == {}

    baseline = {pair: values.copy() for pair, values in cross_tag.items()}
    biased = {tag: value.copy() for tag, value in transforms.items()}
    biased[3][:3, 3] += [0.002, 0.0, 0.0]
    cross_tag, _ = validation_errors(placements, mats, dists, biased)
    np.testing.assert_allclose(cross_tag[(1, 4)], baseline[(1, 4)])
    assert not np.allclose(cross_tag[(1, 3)], baseline[(1, 3)])
    assert not np.allclose(cross_tag[(3, 4)], baseline[(3, 4)])


def test_stereo_placement_cache_roundtrip(tmp_path):
    placements, mats, dists, _, _ = _synthetic_stereo_placements()
    path = tmp_path / "placements.npz"
    save_placements(path, placements, mats, dists)
    assert not (tmp_path / ".placements.npz.tmp").exists()
    loaded, loaded_mats, loaded_dists = load_placements(path)
    assert len(loaded) == len(placements)
    for before, after in zip(placements, loaded):
        assert set(after.corners) == set(before.corners)
        for key in before.corners:
            np.testing.assert_allclose(after.corners[key], before.corners[key])
        for camera in before.anchors:
            np.testing.assert_allclose(after.anchors[camera], before.anchors[camera])
    for camera in mats:
        np.testing.assert_allclose(loaded_mats[camera], mats[camera])
        np.testing.assert_allclose(loaded_dists[camera], dists[camera])


def test_reprojection_gate_discards_bad_view_without_losing_placement():
    placements, _, _, _, _ = _synthetic_stereo_placements()
    observations = [(p_idx, camera, tag, corners)
                    for p_idx, placement in enumerate(placements)
                    for (camera, tag), corners in sorted(placement.corners.items())]
    errors = np.full(len(observations), 0.2)
    errors[0] = 8.0
    filtered, dropped, dropped_placements, cutoff = reject_reprojection_outliers(
        placements, observations, errors)
    assert dropped == 1
    assert dropped_placements == 0
    assert len(filtered) == len(placements)
    assert len(filtered[0].corners) == len(placements[0].corners) - 1
    assert cutoff == pytest.approx(1.0)
