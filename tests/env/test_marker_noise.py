"""Unit tests for src.marker_noise: anisotropic camera-frame AprilTag position
noise. The env-level behaviour (per-tag diffs in a real obs) is covered in
test_obs_noise.py; here we pin the pure geometry."""
import numpy as np

from src.marker_noise import anisotropic_pos_noise, load_focal_px, pos_noise_sigmas


def test_load_focal_px_matches_intrinsics():
    """Focal is the mean of fx/fy from real/vision/camera_intrinsics.yaml (~968 px)."""
    assert 960.0 < load_focal_px() < 975.0


def test_depth_dominates_and_scales_with_distance():
    cam = np.zeros(3)
    f, W = 968.0, 0.02
    lat_a, dep_a, d_a = pos_noise_sigmas(np.array([0.0, 0.0, 0.3]), cam, W, f, 0.4, 2.0)
    # A 20 mm tag at 0.3 m: depth error is tens of times the image-plane error.
    assert dep_a > 20.0 * lat_a
    np.testing.assert_allclose(d_a, [0.0, 0.0, 1.0], atol=1e-9)
    # lateral ~ Z, depth ~ Z**2: doubling the distance doubles lateral, quadruples depth.
    lat_b, dep_b, _ = pos_noise_sigmas(np.array([0.0, 0.0, 0.6]), cam, W, f, 0.4, 2.0)
    np.testing.assert_allclose(lat_b / lat_a, 2.0, rtol=1e-6)
    np.testing.assert_allclose(dep_b / dep_a, 4.0, rtol=1e-6)


def test_depth_factor_only_scales_depth():
    cam = np.zeros(3)
    tag = np.array([0.1, 0.0, 0.3])
    lat1, dep1, _ = pos_noise_sigmas(tag, cam, 0.02, 968.0, 0.4, 1.0)
    lat2, dep2, _ = pos_noise_sigmas(tag, cam, 0.02, 968.0, 0.4, 3.0)
    np.testing.assert_allclose(lat1, lat2, rtol=1e-9)   # lateral independent of depth_factor
    np.testing.assert_allclose(dep2 / dep1, 3.0, rtol=1e-9)


def test_sample_covariance_matches_ellipsoid():
    """Many draws recover the intended covariance: variance along the ray is the
    depth sigma squared, orthogonal to it the lateral sigma squared, zero mean."""
    rng = np.random.default_rng(0)
    cam = np.array([0.1, -0.4, 0.2])
    tag = np.array([0.0, 0.05, 0.15])
    f, W, px, k = 968.0, 0.02, 0.5, 2.0
    lat_s, dep_s, d = pos_noise_sigmas(tag, cam, W, f, px, k)
    samples = np.array([anisotropic_pos_noise(rng, tag, cam, W, f, px, k)
                        for _ in range(20000)])
    cov = np.cov(samples.T)
    np.testing.assert_allclose(d @ cov @ d, dep_s ** 2, rtol=0.1)
    e = np.array([1.0, 0.0, 0.0])
    e = e - (e @ d) * d
    e /= np.linalg.norm(e)
    np.testing.assert_allclose(e @ cov @ e, lat_s ** 2, rtol=0.1)
    np.testing.assert_allclose(samples.mean(axis=0), 0.0, atol=5e-4)
