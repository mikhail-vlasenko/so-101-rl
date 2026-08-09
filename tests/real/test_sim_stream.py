"""Point-cloud overlay contracts shared by native and streamed rollout views."""

import mujoco
import numpy as np
import pytest

from panel.sim_stream import (
    POINT_CLOUD_MAX_POINTS,
    draw_point_cloud,
    sample_point_cloud,
)


_MODEL = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")


def fresh_scene(maxgeom: int = 256) -> mujoco.MjvScene:
    scene = mujoco.MjvScene(_MODEL, maxgeom=maxgeom)
    scene.ngeom = 0
    return scene


def test_cloud_sample_is_stable_and_bounded():
    points = np.arange(300 * 3, dtype=np.float64).reshape(300, 3) / 1000.0
    first = sample_point_cloud(points)
    second = sample_point_cloud(points)

    assert first.shape == (POINT_CLOUD_MAX_POINTS, 3)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[0], points[0])
    np.testing.assert_array_equal(first[-1], points[-1])


def test_draw_cloud_adds_sampled_spheres_at_cloud_points():
    points = np.array([[0.1, 0.2, 0.3], [0.2, -0.1, 0.4]])
    scene = fresh_scene()
    draw_point_cloud(scene, points)

    assert scene.ngeom == 2
    assert all(scene.geoms[i].type == mujoco.mjtGeom.mjGEOM_SPHERE
               for i in range(scene.ngeom))
    np.testing.assert_allclose(scene.geoms[0].pos, points[0])
    np.testing.assert_allclose(scene.geoms[1].pos, points[1])


def test_draw_cloud_respects_remaining_scene_capacity():
    scene = fresh_scene(maxgeom=3)
    scene.ngeom = 2
    draw_point_cloud(scene, np.ones((5, 3)))
    assert scene.ngeom == 3


@pytest.mark.parametrize("points", [np.ones(3), np.ones((3, 2)),
                                     np.array([[np.nan, 0.0, 0.0]])])
def test_invalid_cloud_fails_loud(points):
    with pytest.raises(ValueError, match="point cloud"):
        sample_point_cloud(points)


def test_invalid_sample_limit_fails_loud():
    with pytest.raises(ValueError, match="max_points"):
        sample_point_cloud(np.empty((0, 3)), max_points=0)
