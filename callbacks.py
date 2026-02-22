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


class MaxPhaseTracker(gymnasium.Wrapper):
    """Track max_phase at episode end for eval logging."""

    def __init__(self, env):
        super().__init__(env)
        self.episode_phases: list[float] = []

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        if (term or trunc) and "max_phase" in info:
            self.episode_phases.append(float(info["max_phase"]))
        return obs, rew, term, trunc, info

    def pop_phases(self) -> list[float]:
        phases = self.episode_phases.copy()
        self.episode_phases.clear()
        return phases


class EvalPhaseCallback(BaseCallback):
    """Log mean max_phase after each evaluation round."""

    def __init__(self, phase_tracker: MaxPhaseTracker):
        super().__init__()
        self.phase_tracker = phase_tracker

    def _on_step(self) -> bool:
        phases = self.phase_tracker.pop_phases()
        if phases:
            self.logger.record("eval/mean_max_phase", sum(phases) / len(phases))
        return True
