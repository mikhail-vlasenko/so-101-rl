"""Supervised IK normalization and architecture-grid contracts."""

import numpy as np
import torch

from src.ik_supervised import IKMLP, architecture_specs, normalized_supervised_arrays


def test_normalized_arrays_encode_current_pose_target_and_joint_delta():
    arrays = {
        "current_q": np.array([
            [0.0, -1.0, 0.5, 0.2, -0.4, 0.6],
            [1.0, 0.0, -0.5, -0.2, 0.4, 0.3],
        ], dtype=np.float32),
        "target_q": np.array([
            [0.2, -0.8, 0.4, 0.3, -0.1, 0.6],
            [0.5, 0.4, -0.1, -0.4, 0.2, 0.3],
        ], dtype=np.float32),
        "sponge_xyz": np.array([
            [0.2, 0.0, 0.1],
            [0.1, -0.1, 0.2],
        ], dtype=np.float32),
        "joint_limits": np.array([
            [-2.0, 2.0], [-2.0, 2.0], [-1.0, 1.0],
            [-1.0, 1.0], [-3.0, 3.0], [0.0, 2.0],
        ], dtype=np.float32),
        "target_low_m": np.array([0.1, -0.1, 0.0], dtype=np.float32),
        "target_high_m": np.array([0.3, 0.1, 0.2], dtype=np.float32),
    }
    features, labels = normalized_supervised_arrays(arrays)

    assert features.shape == (2, 9)
    assert labels.shape == (2, 5)
    np.testing.assert_allclose(features[0, 6:], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(features[1, 6:], [-1.0, -1.0, 1.0], atol=1e-6)
    expected_delta = ((arrays["target_q"][0, :5] - arrays["current_q"][0, :5])
                      / (arrays["joint_limits"][:5, 1]
                         - arrays["joint_limits"][:5, 0]))
    np.testing.assert_allclose(labels[0], expected_delta)


def test_mlp_grid_is_100_seeded_runs_plus_linear_baseline():
    config = {
        "depths": [1, 2, 4, 6],
        "widths": [16, 32, 64, 128, 256],
        "seeds": [0, 1, 2, 3, 4],
    }
    specs = architecture_specs(config)
    assert len(specs) == 101
    assert sum(spec.baseline for spec in specs) == 1
    assert len({spec.name for spec in specs}) == len(specs)


def test_mlp_depth_is_number_of_hidden_layers():
    model = IKMLP(input_dim=9, output_dim=5, depth=3, width=16, activation="silu")
    linear_layers = [layer for layer in model.modules() if isinstance(layer, torch.nn.Linear)]
    assert len(linear_layers) == 4
    assert model(torch.zeros(7, 9)).shape == (7, 5)
