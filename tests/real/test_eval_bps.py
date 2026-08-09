"""Offline cached-cloud BPS acceptance report."""

import numpy as np

from real.tracking.eval_bps import validate_cached_clouds


def test_cached_cloud_validator_checks_every_cloud(tmp_path):
    rng = np.random.default_rng(2)
    paths = []
    for index in range(3):
        points = rng.uniform(
            (-0.03, -0.02, -0.0125), (0.03, 0.02, 0.0125), size=(200, 3))
        points += (0.2 + 0.01 * index, 0.0, 0.03)
        path = tmp_path / f"{index:06d}.npz"
        np.savez(path, points=points.astype(np.float32), left_mask_pixels=1000)
        paths.append(path)
    report = validate_cached_clouds(paths)
    assert report["clouds"] == 3
    assert report["points"]["min"] == 200
    assert report["largest_raw_basis_distance_mm"] < 80.0
    assert report["passes_all_gates"]
