"""Dependency-free schema shared by IK dataset, training, and reporting."""

from src.robot_spec import JOINT_NAMES


TRAIN = 0
VAL_SAMPLE = 1
VAL_CELL = 2
TEST_SAMPLE = 3
TEST_CELL = 4
SPLIT_NAMES = (
    "train", "val_sample", "val_cell", "test_sample", "test_cell",
)
ARM_JOINT_NAMES = tuple(JOINT_NAMES[:-1])
