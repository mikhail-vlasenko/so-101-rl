"""Real lift observation construction must match the simulation layout."""

import numpy as np

from real.rollout.rollout_lift import build_state_frame
from src.base_env import EE_OBJECT_DELTA_DIM, state_dim_for


def test_build_state_frame_appends_ee_minus_held_live_centroid():
    qpos = np.arange(6, dtype=np.float32)
    qvel = -qpos
    marker_pos = np.arange(6, dtype=np.float32).reshape(2, 3)
    marker_rot = np.zeros((2, 3), dtype=np.float32)
    marker_age = np.array([0.1, 0.2], dtype=np.float32)
    live = np.array([0.2, -0.1, 0.04], dtype=np.float32)
    ee_pos = np.array([0.25, -0.03, 0.10], dtype=np.float32)
    prev_actions = np.zeros((1, 6), dtype=np.float32)

    frame = build_state_frame(
        qpos, qvel, marker_pos, marker_rot, marker_age, live, 0.05,
        prev_actions, ee_pos, False)

    assert frame.shape == (state_dim_for(1, False),)
    np.testing.assert_allclose(
        frame[-EE_OBJECT_DELTA_DIM:], ee_pos - live, atol=1e-7)
