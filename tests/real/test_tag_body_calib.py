"""Synthetic-data tests for the sponge tag placement solver
(real/tracking/tag_body_calib.py): known (u, v, yaw) placements generate
pairwise relative-transform measurements; the solver must recover them."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from real.calib.extrinsics import mat_inv
from real.tracking.tag_body_calib import (
    face_assignments,
    face_frame,
    face_margins,
    residual_rms,
    solve_faces_and_placements,
    solve_tag_placements,
    tag_body_transform,
)

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
    """Physically realizable placements: a glued tag cannot overhang its face,
    and the solver bounds placements accordingly."""
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
    """A 20 mm tag has 10 mm of slack on the 60 mm axis but only 2.5 mm on the
    25 mm one — the narrow faces are why a wrong assignment is detectable."""
    np.testing.assert_allclose(face_margins(1, "+z", HALF), (0.02, 0.01))
    # +y's in-plane axes are (z, x): the tight bound lands on u, not v.
    np.testing.assert_allclose(face_margins(1, "+y", HALF), (0.0025, 0.02))
    np.testing.assert_allclose(face_margins(1, "+x", HALF), (0.01, 0.0025))


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


def test_solver_rejects_undeclared_tag():
    rng = np.random.default_rng(3)
    truth = _random_truth(rng)
    pairs = _synthetic_pairs(rng, truth, n_pairs=10)
    with pytest.raises(AssertionError):
        solve_tag_placements(pairs, {1: "+z"}, HALF)
