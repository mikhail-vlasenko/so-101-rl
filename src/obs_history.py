"""Lag-tap observation history (.claude/plans/obs_history_features.md).

Fixed history features instead of a recurrent policy: the policy input carries
the actor block at a handful of past control ticks (geometrically spaced lags,
conf/config.yaml:history_taps), so it can identify per-episode DR latents and
remember events (a grasp that closed on nothing) without BPTT.

One shared class serves both sides of the sim-to-real contract: SO101BaseEnv
feeds it in reset/step and the real rollout scripts feed it from the same
per-tick frames — a silent mismatch in tap indexing between the two would be
undetectable at deploy, so both must go through this exact code
(tests/env/test_obs_history.py pins the convention).
"""

import numpy as np


class ObsHistory:
    """Ring buffer serving lag taps over one per-tick observation frame.

    taps are control-tick lags — strictly ascending, starting at 0 (the
    current frame). tapped() concatenates the frames [t - tap for tap in
    taps], newest first. Reset convention: the boot frame fills the whole
    buffer, so before any history exists every tap reads the current obs —
    matching the real boot condition, where no history exists either.
    """

    def __init__(self, taps, dim: int):
        taps = [int(t) for t in taps]
        assert taps and taps[0] == 0, f"taps must start at 0, got {taps}"
        assert taps == sorted(set(taps)), f"taps must be strictly ascending, got {taps}"
        self.taps = tuple(taps)
        self.dim = int(dim)
        self._buf = np.zeros((self.taps[-1] + 1, self.dim), dtype=np.float32)
        self._pos = 0  # ring row holding the newest frame

    @property
    def n_taps(self) -> int:
        return len(self.taps)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        """Seed the whole buffer with the boot frame; returns tapped()."""
        frame = np.asarray(frame, dtype=np.float32)
        assert frame.shape == (self.dim,), (frame.shape, self.dim)
        self._buf[:] = frame
        self._pos = 0
        return self.tapped()

    def push(self, frame: np.ndarray) -> np.ndarray:
        """Advance one control tick with this tick's frame; returns tapped()."""
        frame = np.asarray(frame, dtype=np.float32)
        assert frame.shape == (self.dim,), (frame.shape, self.dim)
        self._pos = (self._pos + 1) % len(self._buf)
        self._buf[self._pos] = frame
        return self.tapped()

    def tapped(self) -> np.ndarray:
        """The concatenated tap frames, newest first: (n_taps * dim,)."""
        rows = [(self._pos - t) % len(self._buf) for t in self.taps]
        return self._buf[rows].reshape(-1)
