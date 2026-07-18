"""Contract tests for the detected-marker sim overlay (panel.sim_stream.
draw_detected_markers). Camera-measured marker poses are drawn on top of the
arm's own marker sites so calibration drift is visible: a box per marker when
orientation is known, a sphere when only position is, and zeroed (undetected)
markers skipped — matching real/rollout/marker_obs.py's zeroing convention.
"""

import mujoco
import numpy as np
import pytest

from panel.sim_stream import _MARKER_BOX_HALF, _rotvec_to_mat, draw_detected_markers

# Minimal model so MjvScene allocates its geoms buffer (a bare MjvScene() has
# none); the geometry is irrelevant — we only draw decorative overlay geoms.
_MODEL = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")


def fresh_scene(maxgeom: int = 100) -> mujoco.MjvScene:
    """An empty scene with an allocated geoms buffer, like a passive viewer's
    user_scn after the caller resets ngeom."""
    scn = mujoco.MjvScene(_MODEL, maxgeom=maxgeom)
    scn.ngeom = 0
    return scn


def test_box_when_rotation_known():
    pos = np.array([[0.1, 0.2, 0.3], [0.0, -0.1, 0.25]])
    rot = np.array([[0.0, 0.0, 0.5], [0.1, 0.0, 0.0]])
    scn = fresh_scene()
    draw_detected_markers(scn, pos, rot, include_rot=True)
    assert scn.ngeom == 2
    assert scn.geoms[0].type == mujoco.mjtGeom.mjGEOM_BOX
    np.testing.assert_allclose(scn.geoms[0].pos, pos[0])
    np.testing.assert_allclose(scn.geoms[0].size, _MARKER_BOX_HALF)
    # Orientation matches the rotvec -> matrix conversion.
    np.testing.assert_allclose(scn.geoms[0].mat.flatten(), _rotvec_to_mat(rot[0]),
                               atol=1e-9)


def test_sphere_when_rotation_unknown():
    pos = np.array([[0.1, 0.2, 0.3]])
    rot = np.array([[0.0, 0.0, 0.0]])
    scn = fresh_scene()
    draw_detected_markers(scn, pos, rot, include_rot=False)
    assert scn.ngeom == 1
    assert scn.geoms[0].type == mujoco.mjtGeom.mjGEOM_SPHERE


def test_zeroed_markers_skipped():
    # Second marker is the all-zero "undetected" sentinel; only the first draws.
    pos = np.array([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]])
    rot = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.0]])
    scn = fresh_scene()
    draw_detected_markers(scn, pos, rot, include_rot=True)
    assert scn.ngeom == 1
    np.testing.assert_allclose(scn.geoms[0].pos, pos[0])


def test_appends_without_clobbering_existing_geoms():
    # The offscreen Renderer hands us a scene already holding the model geoms;
    # the overlay must append, never overwrite.
    scn = fresh_scene()
    scn.ngeom = 5
    draw_detected_markers(scn, np.array([[0.1, 0.2, 0.3]]),
                          np.array([[0.0, 0.0, 0.0]]), include_rot=False)
    assert scn.ngeom == 6


def test_respects_maxgeom():
    scn = fresh_scene(maxgeom=1)
    pos = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    rot = np.zeros((2, 3))
    draw_detected_markers(scn, pos, rot, include_rot=False)
    assert scn.ngeom == 1  # second marker dropped rather than overflowing


def test_rotvec_to_mat_is_orthonormal():
    mat = _rotvec_to_mat(np.array([0.3, -0.4, 0.5])).reshape(3, 3)
    np.testing.assert_allclose(mat @ mat.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(mat), 1.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
