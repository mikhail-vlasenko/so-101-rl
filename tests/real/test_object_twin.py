"""Sim/real twin contract for the dual-channel object obs.

The sim env (src/base_env._ingest_frame) and the real rollout
(real/rollout/object_obs.ObjectSource) must drive the SAME state machine
(src/shape_obs.ObjectChannelDriver) with the same call pattern: seed the
pre-window static evidence, ingest each detected live measurement with its
gate point, refresh precise only through the gate. This test records one sim
episode's raw measurement sequence and pushes it through a standalone driver
exactly the way ObjectSource does — the served channel values must be
identical to what the env's observation carried. The shared module makes this
near-tautological by design; the test pins that nobody forks the logic later.
"""

import numpy as np
import pytest
from hydra import compose, initialize

from src.base_env import RuntimeEnvConfig
from src.lift_env import SO101LiftEnv
from src.shape_obs import STATIC_DWELL_S, ObjectChannelDriver

# Default layout (marker_include_rot=False): see tests/env/test_cube_channels.py.
LIVE = slice(20, 23)
LIVE_AGE = 23
CENTER = slice(24, 27)
SQRTM = slice(27, 33)
PRECISE_AGE = 33


class RecordingLiftEnv(SO101LiftEnv):
    """Logs every ingested camera frame (the raw measurement sequence)."""

    def _ingest_frame(self, capture_t, frame):
        self.ingest_log.append((capture_t, frame))
        super()._ingest_frame(capture_t, frame)


@pytest.fixture(scope="module")
def lift_env_cfg():
    with initialize(config_path="../../conf", version_base=None):
        return compose(config_name="config", overrides=["env=lift"]).lift_env


def test_env_channels_match_standalone_driver_replay(lift_env_cfg):
    """Replay the env's measurement sequence through a fresh driver with the
    ObjectSource call pattern; served channels must match the env obs exactly
    (dropout on, so the replay covers misses and stale-gate stretches too)."""
    env = RecordingLiftEnv(
        env_cfg=lift_env_cfg, xml_path="so101/scene_lift.xml",
        cfg=RuntimeEnvConfig(marker_dropout={"near": 0.3, "far": 0.2}))
    env.ingest_log = []
    obs, _ = env.reset(seed=0)

    driver = ObjectChannelDriver()
    replayed = 0

    def replay_new_frames():
        nonlocal replayed
        for capture_t, frame in env.ingest_log[replayed:]:
            if frame.live_detected:
                # The very first detection carries the pre-episode static
                # evidence, exactly like env.reset / the real warmup.
                if replayed == 0 and not driver._hist_t:
                    driver.seed_static(capture_t - STATIC_DWELL_S, frame.live_gate)
                driver.ingest_live(capture_t, frame.live,
                                   gate_point=frame.live_gate)
                if driver.gate_open(frame.vis_frac):
                    driver.ingest_precise(capture_t, frame.precise_center,
                                          frame.precise_sqrtm)
            replayed += 1

    def assert_matches(obs):
        live, live_age, center, sqrtm6, precise_age = driver.serve(env.data.time)
        np.testing.assert_array_equal(obs[LIVE], live.astype(np.float32))
        np.testing.assert_array_equal(obs[LIVE_AGE], np.float32(live_age))
        np.testing.assert_array_equal(obs[CENTER], center.astype(np.float32))
        np.testing.assert_array_equal(obs[SQRTM], sqrtm6.astype(np.float32))
        np.testing.assert_array_equal(obs[PRECISE_AGE], np.float32(precise_age))

    replay_new_frames()
    assert_matches(obs)

    rng = np.random.default_rng(1)
    for _ in range(120):
        action = rng.uniform(-1.0, 1.0, size=6).astype(np.float32)
        obs, _, term, trunc, _ = env.step(action)
        replay_new_frames()
        assert_matches(obs)
        if term or trunc:
            break
    # The episode must actually exercise the machinery.
    assert replayed > 50
