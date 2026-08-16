"""Synthetic dense-cloud visibility and sim/real BPS twin contract."""

import numpy as np
import pytest
import mujoco

from real.tracking.dense_stereo import cloud_to_bps
from src.base_env import (
    RuntimeEnvConfig,
    cube_surface_points_world,
    cube_visible_surface,
)
from src.bps import BPSObsState, load_bps_config
from src.lift_env import SO101LiftEnv
from src.sim_bps import SyntheticBPSGenerator, SyntheticCloudConfig


def _env_config():
    return {
        "action_scale": 0.07,
        "use_servo_profile": True,
        "max_steps": 20,
        "n_substeps": 10,
        "cube_low": [0.15, -0.15],
        "cube_high": [0.30, 0.15],
        "cube_smallest_face_only": False,
        "cube_no_flat_spawns": False,
        "floor_contact_penalty": 0.0,
        "floor_proximity_thresh": 0.003,
        "floor_proximity_penalty": 0.0,
        "floor_force_coeff": 0.0,
        "poke_force_coeff": 0.0,
        "cube_tip_coeff": 0.0,
        "target_height": 0.10,
    }


def _generator(**overrides):
    values = {
        "point_noise_sigma_m": 0.0,
        "point_dropout_probability": 0.0,
        "whole_view_loss_probability": 0.0,
        "voxel_size_m": 0.002,
    }
    values.update(overrides)
    return SyntheticBPSGenerator(load_bps_config(), SyntheticCloudConfig(**values))


def test_synthetic_cloud_uses_randomized_physical_faces_and_real_transform():
    env = SO101LiftEnv(env_cfg=_env_config(), cfg=RuntimeEnvConfig())
    env.reset(seed=4)
    capture = _generator().capture(
        env.model, env.data, env.cube_cams, env.cube_geom_id,
        env.cube_body_id, env.cube_half_extents, np.random.default_rng(4))
    assert capture is not None
    assert capture.points_base.shape[0] >= 64
    assert capture.left_visible_count >= capture.correspondence_count
    assert 0.0 < capture.measurement.valid_fraction <= 1.0
    assert capture.measurement.distances.shape == (64,)

    # Every generated sample lies on one of the current randomized box faces.
    rotation = env.data.geom_xmat[env.cube_geom_id].reshape(3, 3)
    center = env.data.geom_xpos[env.cube_geom_id]
    local = (capture.points_base - center) @ rotation
    normalized = np.abs(local / env.cube_half_extents)
    np.testing.assert_allclose(normalized.max(axis=1), 1.0, atol=1e-12)

    real_entry = cloud_to_bps(
        capture.points_base, capture.measurement.valid_fraction, load_bps_config())
    np.testing.assert_array_equal(real_entry.distances,
                                  capture.measurement.distances)
    np.testing.assert_array_equal(real_entry.center_base,
                                  capture.measurement.center_base)
    assert real_entry.valid_fraction == capture.measurement.valid_fraction
    env.close()


def test_dropout_noise_whole_loss_and_held_age_are_configurable():
    env = SO101LiftEnv(env_cfg=_env_config(), cfg=RuntimeEnvConfig())
    env.reset(seed=8)
    args = (env.model, env.data, env.cube_cams, env.cube_geom_id,
            env.cube_body_id, env.cube_half_extents)
    clean = _generator().capture(*args, np.random.default_rng(1))
    degraded = _generator(
        point_noise_sigma_m=0.0005,
        point_dropout_probability=0.5,
    ).capture(*args, np.random.default_rng(1))
    assert clean is not None and degraded is not None
    assert degraded.correspondence_count < clean.correspondence_count
    assert degraded.measurement.valid_fraction < clean.measurement.valid_fraction
    assert not np.array_equal(degraded.measurement.distances,
                              clean.measurement.distances)
    assert _generator(whole_view_loss_probability=1.0).capture(
        *args, np.random.default_rng(1)) is None

    state = BPSObsState()
    state.ingest(2.0, clean.measurement)
    state.ingest(2.1, None)
    held = state.serve(2.4)
    np.testing.assert_array_equal(held.distances, clean.measurement.distances)
    assert held.age_s == pytest.approx(0.4)
    env.close()


def test_env_batches_live_and_dense_visibility_per_camera(monkeypatch):
    env = SO101LiftEnv(env_cfg=_env_config(), cfg=RuntimeEnvConfig())
    env.reset(seed=12)
    point_counts = []
    from src import base_env
    original = base_env.visible_surface_mask

    def record_size(model, data, camera, excluded_body_id, points, normals):
        point_counts.append(len(points))
        return original(
            model, data, camera, excluded_body_id, points, normals)

    monkeypatch.setattr(base_env, "visible_surface_mask", record_size)
    state = env._capture_camera_state()

    live_points, live_normals = cube_surface_points_world(
        env.data, env.cube_geom_id, env.cube_half_extents)
    expected_points = len(live_points) + len(env._bps_generator.unit_points)
    assert point_counts == [expected_points, expected_points]
    for i, camera in enumerate(env.cube_cams):
        fraction, centroid = cube_visible_surface(
            env.model, env.data, camera, env.cube_body_id,
            live_points, live_normals)
        assert state.cube_vis_frac[i] == fraction
        np.testing.assert_array_equal(state.cube_vis_centroid[i], centroid)
    assert state.bps_measurement is not None
    env.close()


def test_capture_gate_skips_dense_rays_while_object_moves(monkeypatch):
    env = SO101LiftEnv(env_cfg=_env_config(), cfg=RuntimeEnvConfig())
    env.reset(seed=13)
    point_counts = []
    from src import base_env
    original = base_env.visible_surface_mask

    def record_size(model, data, camera, excluded_body_id, points, normals):
        point_counts.append(len(points))
        return original(
            model, data, camera, excluded_body_id, points, normals)

    monkeypatch.setattr(base_env, "visible_surface_mask", record_size)
    capture_dt = 0.0556
    capture_t = env.data.time
    live_count = len(cube_surface_points_world(
        env.data, env.cube_geom_id, env.cube_half_extents)[0])
    combined_count = live_count + len(env._bps_generator.unit_points)
    measurements = []
    for index in range(14):
        capture_t += capture_dt
        if index == 0:
            env.data.qpos[env.cube_qpos_idx] += 0.01
            mujoco.mj_forward(env.model, env.data)
        state = env._capture_camera_state(
            capture_object=True, gate_dense=True, capture_t=capture_t)
        measurements.append(state.bps_measurement)

    per_frame_counts = np.asarray(point_counts).reshape(-1, 2)[:, 0]
    assert per_frame_counts[0] == combined_count
    assert np.all(per_frame_counts[1:10] == live_count)
    assert all(measurement is None for measurement in measurements[:10])
    assert per_frame_counts[-1] == combined_count
    assert measurements[-1] is not None
    env.close()
