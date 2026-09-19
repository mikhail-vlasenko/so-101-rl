"""Completed-episode table travel and actual motion reward, not height proxies."""

from types import SimpleNamespace

import pytest
from stable_baselines3.common.logger import Logger

from src.callbacks import CubeDragCallback


def test_table_slide_callback_averages_only_completed_lift_episodes():
    logger = Logger(folder=None, output_formats=[])
    callback = CubeDragCallback()
    callback.model = SimpleNamespace(logger=logger)
    callback.locals = {
        "dones": [False, True, True, True],
        "infos": [
            {"episode_table_slide_distance_m": 100.0, "episode_motion_penalty": -100.0},
            {"episode_table_slide_distance_m": 0.02, "episode_motion_penalty": -1.0},
            {"episode_table_slide_distance_m": 0.04, "episode_motion_penalty": -2.0},
            {"task_name": "pickplace", "cube_drag_ratio": 0.2},
        ],
    }
    assert callback._on_step()
    assert logger.name_to_value["rollout/mean_table_slide_distance_m"] == pytest.approx(0.03)
    assert logger.name_to_value["rollout/mean_motion_penalty"] == pytest.approx(-1.5)
    assert logger.name_to_value["rollout/cube_drag_ratio"] == pytest.approx(0.2)
