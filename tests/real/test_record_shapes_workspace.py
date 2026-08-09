"""Dataset coverage bins must follow the configured physical workspace."""
import numpy as np

from real.tracking.record_shapes import Coverage, load_workspace_bounds


def test_coverage_workspace_reads_lift_config():
    low, high = load_workspace_bounds()
    np.testing.assert_allclose(low, (0.10, -0.10))
    np.testing.assert_allclose(high, (0.30, 0.10))


def test_coverage_does_not_clip_out_of_workspace_placements_into_edge_bins():
    coverage = Coverage(np.array((0.10, -0.10)), np.array((0.30, 0.10)))
    pose = np.eye(4)
    pose[:2, 3] = (0.16, -0.126)

    coverage.add(pose, static=True)

    assert coverage.outside_workspace_frames == 1
    assert coverage.counts == {}
