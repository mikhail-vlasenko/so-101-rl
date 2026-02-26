"""Training callbacks and metrics logging for SB3."""

import gymnasium
from stable_baselines3.common.callbacks import BaseCallback


class MeanMaxPhaseCallback(BaseCallback):
    """Log mean max_phase across episodes in each log window.

    Uses logger.record_mean() which accumulates values and automatically
    averages + resets on each dump_logs() call.
    """

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and "max_phase" in info:
                self.logger.record_mean("rollout/mean_max_phase", float(info["max_phase"]))
        return True


class MaxCubeHeightCallback(BaseCallback):
    """Log mean max cube height across episodes in each log window."""

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and "max_cube_height" in info:
                self.logger.record_mean("rollout/mean_max_cube_height", info["max_cube_height"])
        return True


class FloorContactCallback(BaseCallback):
    """Log fraction of steps where the arm touches the floor."""

    def __init__(self):
        super().__init__()
        self._contact_steps = 0
        self._total_steps = 0

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "floor_contact" in info:
                self._total_steps += 1
                if info["floor_contact"]:
                    self._contact_steps += 1
        if self._total_steps > 0:
            self.logger.record_mean(
                "rollout/floor_contact_ratio",
                self._contact_steps / self._total_steps,
            )
        return True


class XYProgressCallback(BaseCallback):
    """Log mean episode XY progress and regress rewards."""

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and "xy_progress" in info:
                self.logger.record_mean("rollout/mean_xy_progress", info["xy_progress"])
                self.logger.record_mean("rollout/mean_xy_regress", info["xy_regress"])
        return True


class MaxPhaseTracker(gymnasium.Wrapper):
    """Track max_phase and max_cube_height at episode end for eval logging."""

    def __init__(self, env):
        super().__init__(env)
        self.episode_phases: list[float] = []
        self.episode_max_cube_heights: list[float] = []

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        if term or trunc:
            if "max_phase" in info:
                self.episode_phases.append(float(info["max_phase"]))
            if "max_cube_height" in info:
                self.episode_max_cube_heights.append(info["max_cube_height"])
        return obs, rew, term, trunc, info

    def pop_stats(self) -> tuple[list[float], list[float]]:
        phases = self.episode_phases.copy()
        heights = self.episode_max_cube_heights.copy()
        self.episode_phases.clear()
        self.episode_max_cube_heights.clear()
        return phases, heights


class EvalPhaseCallback(BaseCallback):
    """Log mean max_phase and max_cube_height after each evaluation round."""

    def __init__(self, phase_tracker: MaxPhaseTracker):
        super().__init__()
        self.phase_tracker = phase_tracker

    def _on_step(self) -> bool:
        phases, heights = self.phase_tracker.pop_stats()
        if phases:
            self.logger.record("eval/mean_max_phase", sum(phases) / len(phases))
        if heights:
            self.logger.record("eval/mean_max_cube_height", sum(heights) / len(heights))
        return True
