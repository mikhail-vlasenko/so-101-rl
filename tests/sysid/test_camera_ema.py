"""Unit test for the static-camera EMA in real.rollout.marker_obs.CameraMarkerSource.

The camera is bolted down, so the pipeline EMAs the per-frame table-tag re-anchor
(base_cam_from_table) to denoise the jitter that would otherwise move every arm/
cube tag in common. We drive _ema_camera directly (skipping the hardware/detector
setup in __init__) with a fixed pose plus white noise.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from real.calib.extrinsics import mat_to_pos_quat, pos_quat_to_mat
from real.rollout.marker_obs import CAM_EMA_ALPHA, CameraMarkerSource


def _fresh_source():
    src = object.__new__(CameraMarkerSource)   # no camera/detector needed for the EMA
    src._cam_ema_pos = None
    src._cam_ema_quat = None
    return src


def test_ema_seeds_on_first_call():
    """The first solve is passed through unchanged (nothing to average yet)."""
    src = _fresh_source()
    pos = np.array([0.1, -0.4, 0.2])
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    out_pos, _ = mat_to_pos_quat(src._ema_camera(pos_quat_to_mat(pos, quat)))
    np.testing.assert_allclose(out_pos, pos, atol=1e-12)


def test_ema_denoises_static_camera():
    """A bolted-down camera + white per-frame noise: the EMA output has far less
    translation variance than the raw per-frame solves and settles near the mean."""
    rng = np.random.default_rng(0)
    true_pos = np.array([0.1, -0.4, 0.2])
    true_quat = np.roll(Rotation.from_euler("xyz", [10, -20, 5], degrees=True).as_quat(), 1)
    sigma = 0.003
    src = _fresh_source()
    raw, smoothed = [], []
    for _ in range(2000):
        noisy_pos = true_pos + rng.normal(0.0, sigma, 3)
        out = src._ema_camera(pos_quat_to_mat(noisy_pos, true_quat))
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
    src = _fresh_source()
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    start = np.array([0.1, -0.4, 0.2])
    src._ema_camera(pos_quat_to_mat(start, quat))
    moved = np.array([0.15, -0.35, 0.25])
    for _ in range(300):
        out_pos, _ = mat_to_pos_quat(src._ema_camera(pos_quat_to_mat(moved, quat)))
    np.testing.assert_allclose(out_pos, moved, atol=1e-3)
    assert CAM_EMA_ALPHA < 0.2   # heavy smoothing, not a passthrough
