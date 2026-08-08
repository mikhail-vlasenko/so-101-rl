"""Dataset coverage bins must follow the configured physical workspace."""
import numpy as np

from real.tracking.record_shapes import load_workspace_bounds


def test_coverage_workspace_reads_lift_config():
    low, high = load_workspace_bounds()
    np.testing.assert_allclose(low, (0.10, -0.10))
    np.testing.assert_allclose(high, (0.30, 0.10))
