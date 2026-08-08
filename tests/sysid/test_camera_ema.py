"""Unit test for the static-camera pose EMA (real.calib.extrinsics.PoseEMA).

The cameras are bolted down, so every re-anchoring consumer (the marker
pipeline, the dataset recorder, the object tracker) EMAs the per-frame
accepted two-tag board solve to denoise the jitter that would
otherwise move every derived pose in common. We drive PoseEMA directly with a
fixed pose plus white noise.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from real.calib.extrinsics import PoseEMA, mat_to_pos_quat, pos_quat_to_mat
from real.calib.table_anchor import load_table_anchor_limits


CAM_EMA_ALPHA = load_table_anchor_limits().ema_alpha


def test_ema_seeds_on_first_call():
    """The first solve is passed through unchanged (nothing to average yet)."""
    ema = PoseEMA(CAM_EMA_ALPHA)
    pos = np.array([0.1, -0.4, 0.2])
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    out_pos, _ = mat_to_pos_quat(ema.update(pos_quat_to_mat(pos, quat)))
    np.testing.assert_allclose(out_pos, pos, atol=1e-12)


def test_ema_denoises_static_camera():
    """A bolted-down camera + white per-frame noise: the EMA output has far less
    translation variance than the raw per-frame solves and settles near the mean."""
    rng = np.random.default_rng(0)
    true_pos = np.array([0.1, -0.4, 0.2])
    true_quat = np.roll(Rotation.from_euler("xyz", [10, -20, 5], degrees=True).as_quat(), 1)
    sigma = 0.003
    ema = PoseEMA(CAM_EMA_ALPHA)
    raw, smoothed = [], []
    for _ in range(2000):
        noisy_pos = true_pos + rng.normal(0.0, sigma, 3)
        out = ema.update(pos_quat_to_mat(noisy_pos, true_quat))
        raw.append(noisy_pos)
        smoothed.append(mat_to_pos_quat(out)[0])
    raw = np.array(raw)
    smoothed = np.array(smoothed[500:])   # drop the warm-up transient
    # Steady-state EMA std of white noise is sqrt(alpha/(2-alpha)) ~ 0.16 of the input.
    assert smoothed.std(axis=0).mean() < 0.3 * raw.std(axis=0).mean()
    np.testing.assert_allclose(smoothed.mean(axis=0), true_pos, atol=sigma)


def test_ema_tracks_a_move():
    """After the camera is bumped to a new fixed pose, the smoothed estimate
    converges there (the re-anchoring is smoothed, not frozen)."""
    ema = PoseEMA(CAM_EMA_ALPHA)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    start = np.array([0.1, -0.4, 0.2])
    ema.update(pos_quat_to_mat(start, quat))
    moved = np.array([0.15, -0.35, 0.25])
    for _ in range(300):
        out_pos, _ = mat_to_pos_quat(ema.update(pos_quat_to_mat(moved, quat)))
    np.testing.assert_allclose(out_pos, moved, atol=1e-3)
    assert CAM_EMA_ALPHA < 0.2   # heavy smoothing, not a passthrough
