"""Offline IK teacher and matched structured dataset-split contracts."""

from pathlib import Path

import numpy as np

from src.ik_dataset import IKTeacher, assign_structured_splits, spatial_cell_ids
from src.ik_common import (
    TEST_CELL,
    TEST_SAMPLE,
    TRAIN,
    VAL_CELL,
    VAL_SAMPLE,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_closest_ik_reaches_known_pose_without_moving_gripper():
    teacher = IKTeacher(
        str(REPO_ROOT / "so101/scene.xml"),
        position_tolerance_m=0.0005,
        joint_limit_margin_fraction=0.01,
        ik_max_iterations=80,
    )
    current = np.array([0.0, -1.4, 1.2, 0.0, 0.0, 0.5])
    known_solution = np.array([0.7, -1.4, 1.2, 0.0, 0.0, 0.5])
    target_xyz = teacher.forward(known_solution)
    solved = teacher.solve(current, target_xyz, np.random.default_rng(3), restarts=3)

    assert solved is not None
    assert np.linalg.norm(teacher.forward(solved) - target_xyz) <= 0.0005
    assert solved[-1] == current[-1]
    solved_distance = np.sum(((solved[:-1] - current[:-1])
                              / teacher.joint_range[:-1]) ** 2)
    known_distance = np.sum(((known_solution[:-1] - current[:-1])
                             / teacher.joint_range[:-1]) ** 2)
    assert solved_distance <= known_distance + 1e-6


def test_validation_and_test_have_matching_sample_and_cell_structure():
    low = np.array([0.10, -0.10, 0.03])
    high = np.array([0.30, 0.10, 0.20])
    cells = np.array([5, 5, 4])
    rng = np.random.default_rng(7)
    points = []
    for x_index in range(cells[0]):
        for y_index in range(cells[1]):
            for z_index in range(cells[2]):
                cell_low = low + (high - low) * np.array(
                    [x_index, y_index, z_index]) / cells
                cell_high = low + (high - low) * (
                    np.array([x_index, y_index, z_index]) + 1) / cells
                points.append(rng.uniform(cell_low, cell_high, size=(20, 3)))
    points = np.concatenate(points)

    split, cell_ids = assign_structured_splits(
        points, low, high, cells,
        validation_cell_fraction=0.10,
        test_cell_fraction=0.10,
        validation_sample_fraction=0.10,
        test_sample_fraction=0.10,
        rng=np.random.default_rng(11),
    )
    np.testing.assert_array_equal(
        cell_ids, spatial_cell_ids(points, low, high, cells))

    val_cells = set(cell_ids[split == VAL_CELL])
    test_cells = set(cell_ids[split == TEST_CELL])
    shared_cells = set(cell_ids[split == TRAIN])
    assert len(val_cells) == len(test_cells) == 10
    assert val_cells.isdisjoint(test_cells)
    assert val_cells.isdisjoint(shared_cells)
    assert test_cells.isdisjoint(shared_cells)
    assert set(cell_ids[split == VAL_SAMPLE]) == shared_cells
    assert set(cell_ids[split == TEST_SAMPLE]) == shared_cells
    assert np.count_nonzero(split == VAL_SAMPLE) == np.count_nonzero(split == TEST_SAMPLE)
    assert np.count_nonzero(split == VAL_CELL) == np.count_nonzero(split == TEST_CELL)


def test_heldout_cell_assignment_balances_uneven_cell_occupancy():
    low = np.zeros(3)
    high = np.ones(3)
    cells = np.array([5, 2, 1])
    rng = np.random.default_rng(4)
    points = []
    counts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for flat_cell, count in enumerate(counts):
        x_index, y_index, z_index = np.unravel_index(flat_cell, tuple(cells))
        cell_low = np.array([x_index, y_index, z_index]) / cells
        cell_high = (np.array([x_index, y_index, z_index]) + 1) / cells
        points.append(rng.uniform(cell_low, cell_high, size=(count, 3)))
    points = np.concatenate(points)

    split, _ = assign_structured_splits(
        points, low, high, cells,
        validation_cell_fraction=0.20,
        test_cell_fraction=0.20,
        validation_sample_fraction=0.10,
        test_sample_fraction=0.10,
        rng=np.random.default_rng(8),
    )
    val_count = np.count_nonzero(split == VAL_CELL)
    test_count = np.count_nonzero(split == TEST_CELL)
    assert abs(val_count - test_count) <= max(counts)
