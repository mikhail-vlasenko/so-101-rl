"""Temporary training wrapper for the final pre-BPS tag policy family.

The current environment always advances and serves the BPS observation path.
This wrapper changes only the observation returned to SB3, replacing it with
the actor-only prefix of :meth:`SO101BaseEnv.legacy_tag_obs`.  It exists so the
last useful tag checkpoint can be adapted to current physics before one final
distillation; deployed policies must use the normal BPS environment.
"""

import gymnasium
import numpy as np
from gymnasium import spaces

from src.base_env import legacy_tag_actor_dim_for


class LegacyTagActorObs(gymnasium.ObservationWrapper):
    """Serve the symmetric actor-only observation expected by old teachers."""

    def __init__(self, env):
        super().__init__(env)
        base = env.unwrapped
        self.actor_dim = legacy_tag_actor_dim_for(
            base.prev_actions_n, base.marker_include_rot)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.actor_dim,),
            dtype=np.float32)

    def observation(self, observation):
        del observation
        return self.env.unwrapped.legacy_tag_obs()[:self.actor_dim]
