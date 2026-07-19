"""Training callbacks and metrics logging for SB3."""

import time

import gymnasium
from stable_baselines3.common.callbacks import BaseCallback


class TimeLimitCallback(BaseCallback):
    """Stop training after a given number of minutes."""

    def __init__(self, time_limit_minutes: float):
        super().__init__()
        self.time_limit_seconds = time_limit_minutes * 60
        self._start_time: float = 0

    def _on_training_start(self) -> None:
        self._start_time = time.monotonic()

    def _on_step(self) -> bool:
        elapsed = time.monotonic() - self._start_time
        if elapsed >= self.time_limit_seconds:
            print(f"Time limit reached ({self.time_limit_seconds / 60:.0f}m). Stopping training.")
            return False
        return True


class MaxCubeHeightCallback(BaseCallback):
    """Log mean max cube height across episodes in each log window."""

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and "max_cube_height" in info:
                val = info["max_cube_height"]
                self.logger.record_mean("rollout/mean_max_cube_height", val)
                if "task_name" in info:
                    self.logger.record_mean(f"rollout/{info['task_name']}/mean_max_cube_height", val)
        return True


class FloorContactCallback(BaseCallback):
    """Log mean per-episode floor contact ratio across episodes in each log window."""

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and "floor_contact_ratio" in info:
                val = info["floor_contact_ratio"]
                self.logger.record_mean("rollout/floor_contact_ratio", val)
                if "task_name" in info:
                    self.logger.record_mean(f"rollout/{info['task_name']}/floor_contact_ratio", val)
        return True


class MarkerHiddenCallback(BaseCallback):
    """Log mean per-episode obs-availability metrics: fraction of marker-steps
    (arm tags) hidden from the tag camera, fraction of steps the cube's live
    channel had no both-view measurement, and the mean served precise-channel
    age."""

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if not done:
                continue
            for key in ("marker_hidden_ratio", "live_hidden_ratio",
                        "precise_age_mean"):
                if key in info:
                    val = info[key]
                    self.logger.record_mean(f"rollout/{key}", val)
                    if "task_name" in info:
                        self.logger.record_mean(f"rollout/{info['task_name']}/{key}", val)
        return True


class CubeDragCallback(BaseCallback):
    """Log mean per-episode cube-drag ratio (cube near floor + lateral motion)."""

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and "cube_drag_ratio" in info:
                val = info["cube_drag_ratio"]
                self.logger.record_mean("rollout/cube_drag_ratio", val)
                if "task_name" in info:
                    self.logger.record_mean(f"rollout/{info['task_name']}/cube_drag_ratio", val)
        return True


class RingContactCallback(BaseCallback):
    """Log mean per-episode ring (wall) contact ratio across episodes in each log window."""

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and "ring_contact_ratio" in info:
                val = info["ring_contact_ratio"]
                self.logger.record_mean("rollout/ring_contact_ratio", val)
                if "task_name" in info:
                    self.logger.record_mean(f"rollout/{info['task_name']}/ring_contact_ratio", val)
        return True


class XYProgressCallback(BaseCallback):
    """Log mean episode XY progress and regress rewards."""

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and "xy_progress" in info:
                self.logger.record_mean("rollout/mean_xy_progress", info["xy_progress"])
                self.logger.record_mean("rollout/mean_xy_regress", info["xy_regress"])
                if "task_name" in info:
                    task = info["task_name"]
                    self.logger.record_mean(f"rollout/{task}/mean_xy_progress", info["xy_progress"])
                    self.logger.record_mean(f"rollout/{task}/mean_xy_regress", info["xy_regress"])
        return True


class MeanReturnCallback(BaseCallback):
    """Log mean episode return using record_mean, with per-task breakdown."""

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and "episode" in info:
                ep_return = info["episode"]["r"]
                self.logger.record_mean("rollout/mean_return", ep_return)
                if "task_name" in info:
                    self.logger.record_mean(f"rollout/{info['task_name']}/mean_return", ep_return)
        return True


class EpisodeLengthCallback(BaseCallback):
    """Log mean episode length (steps) across episodes in each log window.

    Lower = the arm reaches a grasped lift sooner (the "lift on the first try"
    objective). Batch-averaged via record_mean over the rollout window; SB3's
    built-in rollout/ep_len_mean uses stats_window_size (1 here), so it only ever
    reflects the most recent episode and is not comparable across runs."""

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and "episode" in info:
                ep_len = info["episode"]["l"]
                self.logger.record_mean("rollout/mean_ep_length", ep_len)
                if "task_name" in info:
                    self.logger.record_mean(f"rollout/{info['task_name']}/mean_ep_length", ep_len)
        return True


class LiftSuccessCallback(BaseCallback):
    """Log the fraction of lift episodes that ended in a grasped lift to target height.

    Pairs with mean_ep_length: episode length only means "fast" if success is high,
    so read the two together (a policy that never succeeds also runs the full 300)."""

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and "lift_success" in info:
                self.logger.record_mean(
                    f"rollout/{info['task_name']}/success_rate", float(info["lift_success"]),
                )
        return True


class CompletionRateCallback(BaseCallback):
    """Log the fraction of episodes where the cube was successfully placed."""

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and "placed" in info:
                self.logger.record_mean("rollout/completion_rate", float(info["placed"]))
                if "task_name" in info:
                    self.logger.record_mean(
                        f"rollout/{info['task_name']}/completion_rate", float(info["placed"]),
                    )
        return True


class ReachStatsCallback(BaseCallback):
    """Log reach-task metrics: success rate and EE-below-min ratio."""

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and "reached" in info:
                task = info["task_name"]
                self.logger.record_mean(f"rollout/{task}/reached_rate", float(info["reached"]))
                self.logger.record_mean(f"rollout/{task}/ee_below_ratio", info["ee_below_ratio"])
                self.logger.record_mean(
                    f"rollout/{task}/mean_limit_excess_sq", info["mean_limit_excess_sq"],
                )
        return True


class EpisodeCountCallback(BaseCallback):
    """Log the number of completed episodes per rollout."""

    def __init__(self):
        super().__init__()
        self._episode_count = 0

    def _on_step(self) -> bool:
        for done in self.locals["dones"]:
            if done:
                self._episode_count += 1
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record("rollout/episodes", self._episode_count)
        self._episode_count = 0


class EvalStatsTracker(gymnasium.Wrapper):
    """Track max_cube_height at episode end for eval logging."""

    def __init__(self, env):
        super().__init__(env)
        self.episode_max_cube_heights: list[float] = []

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        if term or trunc:
            if "max_cube_height" in info:
                self.episode_max_cube_heights.append(info["max_cube_height"])
        return obs, rew, term, trunc, info

    def pop_stats(self) -> list[float]:
        heights = self.episode_max_cube_heights.copy()
        self.episode_max_cube_heights.clear()
        return heights


class EvalStatsCallback(BaseCallback):
    """Log mean max_cube_height after each evaluation round."""

    def __init__(self, stats_tracker: EvalStatsTracker):
        super().__init__()
        self.stats_tracker = stats_tracker

    def _on_step(self) -> bool:
        heights = self.stats_tracker.pop_stats()
        if heights:
            self.logger.record("eval/mean_max_cube_height", sum(heights) / len(heights))
        return True
