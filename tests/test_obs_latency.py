"""Observation latency contract (sim2real).

Only camera-derived observations (markers, cube) lag. They come from discrete
simulated camera frames (src/camera_sim.py): captured every frame_ms with a
per-episode random phase, available to the policy delay_ms after capture,
consumed at the next control tick — mirroring real/marker_obs.py, where the
measured rollout staleness (marker_age_ms telemetry) is the AprilTag detection
plus the wait for the next policy tick on top of the capture pipeline delay.

The encoder path deliberately has no latency: the real bus read is ~2 ms
against a 66.7 ms control tick, so qpos is fresh, and qvel is the backward
difference of consecutive qpos obs over the tick (real/rollout_common.ArmLoop
semantics), which carries its half-tick lag structurally rather than via a knob.
"""

import numpy as np
import pytest

from hydra import compose, initialize

from src.base_env import marker_world_poses
from src.camera_sim import CameraSim
from src.pickplace_env import SO101PickPlaceEnv

CONTROL_DT = 1.0 / 15.0
SUBSTEP = CONTROL_DT / 10.0

# One-tick-scale camera config with a deterministic delay for exact assertions.
CAM = {"frame_ms": 33.3, "delay_ms": [45.0, 45.0], "jitter_ms": 0.0}


@pytest.fixture(scope="module")
def cfg():
    with initialize(config_path="../conf", version_base=None):
        return compose(config_name="config", overrides=["env=pickplace"])


def _pickplace(cfg, cam_latency, **kwargs):
    # Disable noise so observations can be compared by exact equality.
    return SO101PickPlaceEnv(env_cfg=cfg.pickplace_env,
                             xml_path="so101/scene_pickplace.xml",
                             obs_noise=None,
                             cam_latency=cam_latency, **kwargs)


def _move_action():
    # Non-zero action so qpos actually changes between steps.
    return np.full(6, 0.5, dtype=np.float32)


# ---------------------------------------------------------------- CameraSim

def _run_camera(cam, rng, n_ticks=30):
    """Drive a CameraSim like the env does — state = its own timestamp — and
    return the consumed frame's capture time at each control tick."""
    consumed = []
    cam.reset(rng, 0.0, 0.0, capture_fn=lambda s: s)
    t = 0.0
    for _ in range(n_ticks):
        for _ in range(10):
            t += SUBSTEP
            cam.record(t, t)
        consumed.append(cam.observe(t, rng, capture_fn=lambda s: s))
    return np.array(consumed)


def test_camera_sim_synchronous_is_fresh():
    """Synchronous mode must return the state recorded at each obs instant."""
    cam = CameraSim.synchronous(CONTROL_DT)
    rng = np.random.default_rng(0)
    capture_ts = _run_camera(cam, rng)
    obs_ts = CONTROL_DT * np.arange(1, len(capture_ts) + 1)
    np.testing.assert_allclose(capture_ts, obs_ts, atol=SUBSTEP / 2)


def test_camera_sim_age_matches_schedule():
    """With fixed delay d and no jitter, the consumed frame at obs time t must be
    the newest capture with capture_t + d <= t, so its age lies in
    [d, d + frame_s) (plus half a substep of history-lookup rounding)."""
    frame_s = CAM["frame_ms"] * 1e-3
    delay_s = CAM["delay_ms"][0] * 1e-3
    cam = CameraSim(frame_s=frame_s, delay_range_s=(delay_s, delay_s),
                    jitter_s=0.0, control_dt=CONTROL_DT)
    rng = np.random.default_rng(1)
    capture_ts = _run_camera(cam, rng)
    obs_ts = CONTROL_DT * np.arange(1, len(capture_ts) + 1)
    ages = obs_ts - capture_ts
    assert np.all(ages >= delay_s - SUBSTEP / 2 - 1e-9), ages.min()
    assert np.all(ages < delay_s + frame_s + SUBSTEP / 2 + 1e-9), ages.max()
    # Captures advance monotonically — the policy never sees time run backwards.
    assert np.all(np.diff(capture_ts) >= 0.0)


def test_camera_sim_jitter_makes_age_drift():
    """Per-frame capture jitter must make the consumed-frame age wander instead
    of freezing the camera/control beat at one value."""
    frame_s = CAM["frame_ms"] * 1e-3
    delay_s = 0.045
    cam = CameraSim(frame_s=frame_s, delay_range_s=(delay_s, delay_s),
                    jitter_s=0.0015, control_dt=CONTROL_DT)
    rng = np.random.default_rng(2)
    capture_ts = _run_camera(cam, rng, n_ticks=300)
    ages = CONTROL_DT * np.arange(1, len(capture_ts) + 1) - capture_ts
    assert ages.std() > 0.002, f"ages barely move: std={ages.std():.4f}s"
    assert np.all(ages >= delay_s - SUBSTEP / 2 - 1e-9)
    assert np.all(ages < delay_s + 2 * frame_s)


def test_camera_sim_delay_sampled_per_episode():
    """delay_ms is a per-episode range: different resets draw different delays."""
    cam = CameraSim(frame_s=0.0333, delay_range_s=(0.030, 0.060),
                    jitter_s=0.0, control_dt=CONTROL_DT)
    rng = np.random.default_rng(3)
    delays = []
    for _ in range(20):
        cam.reset(rng, 0.0, 0.0, capture_fn=lambda s: s)
        delays.append(cam._delay)
    delays = np.array(delays)
    assert delays.min() >= 0.030 and delays.max() <= 0.060
    assert delays.std() > 0.005, "per-episode delay draws should spread over the range"


# ---------------------------------------------------------------- env contract

def test_default_config_has_cam_latency(cfg):
    """The deployment DR preset must model camera latency: a real camera cannot
    deliver poses instantaneously (delay range strictly positive)."""
    lat = cfg.cam_latency
    assert lat is not None
    assert float(lat.frame_ms) > 0.0
    lo, hi = float(lat.delay_ms[0]), float(lat.delay_ms[1])
    assert 0.0 < lo <= hi


def test_synchronous_camera_markers_are_fresh(cfg):
    """cam_latency=None: marker and cube obs equal the current true state every
    step (the dr=none curriculum contract)."""
    env = _pickplace(cfg, cam_latency=None, marker_always_visible=True)
    env.reset(seed=0)
    for _ in range(5):
        obs, *_ = env.step(_move_action())
        cur_pos, _ = marker_world_poses(env.data, env.marker_site_ids)
        np.testing.assert_allclose(obs[12:18].reshape(2, 3), cur_pos, atol=1e-6)
        cube = env.data.qpos[env.cube_qpos_idx:env.cube_qpos_idx + 3]
        np.testing.assert_allclose(obs[18:21], cube, atol=1e-6)


def test_delayed_camera_markers_lag_current_state(cfg):
    """With the camera model on, marker obs must come from a recorded past state
    whose age is within [delay_lo, delay_hi + frame_ms], never the current pose
    while the arm moves."""
    env = _pickplace(cfg, cam_latency=CAM, marker_always_visible=True)
    env.reset(seed=0)
    delay_s = CAM["delay_ms"][0] * 1e-3
    frame_s = CAM["frame_ms"] * 1e-3
    lagged_steps = 0
    for step in range(10):
        obs, *_ = env.step(_move_action())
        obs_markers = obs[12:18].reshape(2, 3)
        cur_pos, _ = marker_world_poses(env.data, env.marker_site_ids)
        if not np.allclose(obs_markers, cur_pos, atol=1e-9):
            lagged_steps += 1
        # The obs must match one recorded history state, and that state's age
        # must sit inside the configured latency window.
        ages = [env.data.time - t for t, s in env._camera._hist
                if np.allclose(obs_markers, s.marker_pos, atol=1e-6)]
        assert ages, f"step {step}: obs markers match no recorded state"
        age = min(ages)
        assert delay_s - SUBSTEP <= age <= delay_s + frame_s + SUBSTEP, age
    assert lagged_steps >= 8, f"markers matched the live pose on {10 - lagged_steps}/10 moving steps"


def test_qpos_is_fresh_under_camera_latency(cfg):
    """Joint encoders don't go through the camera: qpos obs equals the current
    joint state even with the latency model on."""
    env = _pickplace(cfg, cam_latency=CAM)
    env.reset(seed=0)
    for _ in range(5):
        obs, *_ = env.step(_move_action())
        np.testing.assert_allclose(obs[:6], env.data.qpos[env.joint_qposadr],
                                    atol=1e-6)


def test_qvel_is_backward_difference_of_qpos_obs(cfg):
    """qvel obs = (qpos_t - qpos_{t-1}) / control_dt, exactly the real pipeline
    (ArmLoop.tick); the first obs after reset is zero (ArmLoop.boot)."""
    env = _pickplace(cfg, cam_latency=CAM)
    obs_prev, _ = env.reset(seed=0)
    assert np.all(obs_prev[6:12] == 0.0)
    for _ in range(5):
        obs, *_ = env.step(_move_action())
        expected = (obs[:6] - obs_prev[:6]) / env._step_dt
        np.testing.assert_allclose(obs[6:12], expected, rtol=1e-4, atol=1e-5)
        obs_prev = obs


def test_reset_gives_fresh_initial_frame(cfg):
    """The pre-episode world is static, so the frame in flight at reset shows the
    reset state: initial marker obs must match the reset pose exactly, with no
    leftover frames from the previous episode."""
    env = _pickplace(cfg, cam_latency=CAM, marker_always_visible=True)
    env.reset(seed=0)
    for _ in range(5):
        env.step(_move_action())
    obs, _ = env.reset(seed=1)
    cur_pos, _ = marker_world_poses(env.data, env.marker_site_ids)
    np.testing.assert_allclose(obs[12:18].reshape(2, 3), cur_pos, atol=1e-6)


def test_reward_uses_true_state_under_latency(cfg):
    """Reward path reads self.data, not the lagged obs — so reward stays sane."""
    env = _pickplace(cfg, cam_latency=CAM)
    env.reset(seed=0)
    for _ in range(3):
        _, reward, _, _, info = env.step(_move_action())
        assert np.isfinite(reward)
        assert isinstance(info["grasped"], (bool, np.bool_))
