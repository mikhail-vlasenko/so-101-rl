"""Sim/real twin contract for live gating and held BPS publication.

The sim env (src/base_env._ingest_frame) and the real rollout
(the Stage 5 real worker) must drive the same live gate and BPS hold state.
This test records one sim
episode's raw measurement sequence and pushes it through a standalone driver
exactly the way the worker contract requires — the served channel values must be
identical to what the env's observation carried. The shared module makes this
near-tautological by design; the test pins that nobody forks the logic later.
"""

import numpy as np
import pytest
from hydra import compose, initialize

from src.base_env import RuntimeEnvConfig, state_dim_for
from src.bps import BPS_DISTANCE_DIM, BPSObsState
from src.lift_env import SO101LiftEnv
from src.shape_obs import STATIC_DWELL_S, ObjectChannelDriver

# Default layout (marker_include_rot=False): see tests/env/test_cube_channels.py.
LIVE = slice(20, 23)
LIVE_AGE = 23
_STATE_DIM = state_dim_for(2, False)
BPS = slice(_STATE_DIM, _STATE_DIM + BPS_DISTANCE_DIM)
CENTER = slice(BPS.stop, BPS.stop + 3)
PRECISE_AGE = CENTER.stop
VALID_FRACTION = PRECISE_AGE + 1


class RecordingLiftEnv(SO101LiftEnv):
    """Logs every live/BPS delivery from the shared capture sequence."""

    def _ingest_live_frame(self, capture_t, frame):
        self.ingest_log.append(("live", capture_t, frame))
        super()._ingest_live_frame(capture_t, frame)

    def _ingest_bps_frame(self, capture_t, frame):
        self.ingest_log.append(("bps", capture_t, frame))
        super()._ingest_bps_frame(capture_t, frame)


@pytest.fixture(scope="module")
def lift_env_cfg():
    with initialize(config_path="../../conf", version_base=None):
        return compose(config_name="config", overrides=["env=lift"]).lift_env


def test_env_channels_match_standalone_driver_replay(lift_env_cfg):
    """Replay the env's measurement sequence through a fresh driver with the
    shared worker call pattern; served channels must match the env obs exactly
    (dropout on, so the replay covers misses and stale-gate stretches too)."""
    env = RecordingLiftEnv(
        env_cfg=lift_env_cfg, xml_path="so101/scene_lift.xml",
        cfg=RuntimeEnvConfig(marker_dropout={"near": 0.3, "far": 0.2}))
    env.ingest_log = []
    obs, _ = env.reset(seed=0)

    driver = ObjectChannelDriver()
    bps_state = BPSObsState()
    bps_eligible = set()
    replayed = 0

    def replay_new_frames():
        nonlocal replayed
        for modality, capture_t, frame in env.ingest_log[replayed:]:
            if modality == "live" and frame.live_detected:
                # The very first detection carries the pre-episode static
                # evidence, exactly like env.reset / the real warmup.
                if not driver._hist_t:
                    driver.seed_static(capture_t - STATIC_DWELL_S, frame.live_gate)
                driver.ingest_live(capture_t, frame.live,
                                   gate_point=frame.live_gate)
                if driver.gate_open(frame.vis_frac):
                    bps_eligible.add(capture_t)
            elif modality == "bps":
                eligible = capture_t in bps_eligible
                bps_eligible.discard(capture_t)
                if capture_t <= env._episode_start_t and frame.live_detected:
                    eligible = driver.gate_open(frame.vis_frac)
                if eligible:
                    bps_state.ingest(capture_t, frame.bps_measurement)
            replayed += 1

    def assert_matches(obs):
        live, live_age = driver.serve(env.data.time)
        bps = bps_state.serve(env.data.time)
        np.testing.assert_array_equal(obs[LIVE], live.astype(np.float32))
        np.testing.assert_array_equal(obs[LIVE_AGE], np.float32(live_age))
        np.testing.assert_array_equal(obs[BPS], bps.distances)
        np.testing.assert_array_equal(obs[CENTER], bps.center_base)
        np.testing.assert_array_equal(obs[PRECISE_AGE], np.float32(bps.age_s))
        np.testing.assert_array_equal(obs[VALID_FRACTION],
                                      np.float32(bps.valid_fraction))

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
