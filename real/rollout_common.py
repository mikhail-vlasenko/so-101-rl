"""Shared real-arm rollout core.

Everything that must behave identically across every policy rollout script
lives here: Hydra config composition, policy loading with the obs-dim check,
the common CLI arguments, the boot-time calibration sanity check, and — most
importantly — the per-tick command shaping in `ArmLoop.tick`:

    clip → EMA → prev-actions buffer → action_to_target (raw-unit quantization
    + stiction deadzone, exactly matching training) → rad_to_raw → per-tick raw
    clamp → sub-target streaming across the (possibly --slow-dilated) wall tick
    → encoder read-back → sim-time qvel.

Task scripts (rollout_lift, rollout_real) own only what genuinely differs:
observation construction, termination, and logging/plots. When a sim-to-real
shaping fix lands, it lands here once and every rollout gets it.
"""

from __future__ import annotations

import argparse
import signal
from pathlib import Path

import mujoco
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from stable_baselines3 import PPO

from src.checkpoints import resolve_model_path
from src.units import action_to_target, max_joint_speed_rad_s, max_raw_delta_per_step

from .twin.constants import (
    INTERP_HZ,
    SERVO_ACCEL,
    SERVO_POSITION_DEADZONE,
    SERVO_POSITION_KP,
    SERVO_SPEED,
)
from .twin.control import clamp_raw_delta, stream_sub_targets
from .twin.mapping import JointMaps, rad_to_raw, raw_to_rad
from .twin.servo_io import ServoBus

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CAL = REPO_ROOT / "real" / "follower_calibration.json"

# Tolerance (rad) on the boot-pose calibration check: encoder noise and
# resting-against-a-limit poses may read slightly past the XML joint range.
INIT_RANGE_SLACK = 0.05


def load_env_cfg(env_name: str) -> tuple[dict, int]:
    """Compose the full Hydra config with env=<env_name> so ${action_scale}
    (and any other top-level interpolations) resolve. Returns
    (env cfg dict, prev_actions_n); prev_actions_n must match the trained policy."""
    with initialize_config_dir(config_dir=str(REPO_ROOT / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=[f"env={env_name}"])
    return OmegaConf.to_container(cfg[f"{env_name}_env"], resolve=True), int(cfg.prev_actions_n)


def add_common_args(p: argparse.ArgumentParser, default_xml: Path,
                    default_max_steps: int) -> None:
    """CLI arguments every rollout script shares."""
    p.add_argument("--model", default="latest",
                   help="'latest' (newest checkpoint), 'best', or a path to a .zip")
    p.add_argument("--execute", action="store_true",
                   help="Actually send servo commands. Default: dry-run.")
    p.add_argument("--max-steps", type=int, default=default_max_steps)
    p.add_argument("--ema-alpha", type=float, default=1.0,
                   help="EMA smoothing on policy action: a_t = alpha*a + (1-alpha)*a_{t-1}. "
                        "1.0 = off (default), lower = smoother but laggier.")
    p.add_argument("--slow", type=float, default=1.0,
                   help="Time-dilation factor >= 1: stretch each control tick "
                        "to slow*control_dt of wall time, so the arm executes "
                        "the same per-tick position deltas at 1/slow physical "
                        "speed. Observations stay in sim-time units (qvel uses "
                        "the sim tick), so the policy sees its training "
                        "distribution — no retraining needed.")
    p.add_argument("--interp-hz", type=float, default=INTERP_HZ,
                   help="Rate at which interpolated sub-targets are written to "
                        "the bus between policy ticks. Linearly slides from the "
                        "previous commanded target to the new one across each "
                        "control period, so the servo's trapezoidal trajectory "
                        "tracks a moving setpoint instead of restarting from "
                        "rest every tick. Set to the control rate (15) to "
                        "disable.")
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--xml", default=str(default_xml))
    p.add_argument("--cal", default=str(DEFAULT_CAL))
    p.add_argument("--stream-port", type=int, default=None,
                   help="Serve the sim view as MJPEG on this port "
                        "(http://host:PORT/stream); used by the web panel.")


def load_policy(model_arg: str, log_dir: Path, expected_obs: int) -> PPO:
    """Resolve and load a checkpoint, failing loud on an obs-dim mismatch."""
    model_path = resolve_model_path(model_arg, str(log_dir))
    print(f"loading model: {model_path}")
    policy = PPO.load(model_path)
    assert policy.observation_space.shape[0] == expected_obs, (
        f"Obs dim mismatch: model expects {policy.observation_space.shape}, "
        f"env produces {expected_obs}-dim."
    )
    return policy


def install_sigint_flag() -> dict:
    """Ctrl-C sets a flag instead of raising, so the control loop exits at a
    tick boundary and the caller's `finally: bus.close()` disables torque."""
    stopped = {"flag": False}

    def stop(_sig, _frame) -> None:
        stopped["flag"] = True

    signal.signal(signal.SIGINT, stop)
    return stopped


class ArmLoop:
    """Control-loop state + per-tick command shaping for a real-arm rollout.

    Construct before `bus.connect()` (validates arguments, touches no
    hardware), then call `boot()` once inside the caller's try/finally and
    `tick(action)` once per policy step. `qpos`, `qvel` (sim-time units) and
    `prev_actions` hold the state the task script needs to build observations.
    """

    def __init__(self, model: mujoco.MjModel, n_substeps: int, jm: JointMaps,
                 bus: ServoBus, action_scale: float, prev_actions_n: int,
                 execute: bool, ema_alpha: float, slow: float, interp_hz: float):
        assert 0.0 < ema_alpha <= 1.0, f"ema_alpha must be in (0, 1], got {ema_alpha}"
        assert slow >= 1.0, f"slow must be >= 1, got {slow}"
        self.jm = jm
        self.bus = bus
        self.n_joints = len(jm.items)
        self.direction = np.ones(self.n_joints, dtype=np.int8)  # verified via twin: no inversions
        self.xml_low, self.xml_high = jm.xml_low(), jm.xml_high()
        self.action_scale = action_scale
        self.execute = execute
        self.ema_alpha = ema_alpha
        self.slow = slow

        # Control period is (timestep * n_substeps) — same definition the sim
        # env uses. The wall tick is sim time stretched by --slow; the policy
        # and all observations keep working in sim time (control_dt).
        self.control_dt = float(model.opt.timestep) * n_substeps
        self.control_hz = 1.0 / self.control_dt
        self.wall_tick = slow * self.control_dt
        self.max_raw_delta = max_raw_delta_per_step(action_scale)
        # Interpolation spans the (possibly dilated) wall tick at ~interp_hz,
        # so slower rollouts stream proportionally more, smaller sub-targets.
        self.n_interp = max(1, int(round(interp_hz * self.wall_tick)))
        self.sub_dt = self.wall_tick / self.n_interp

        self._prev_actions_n = int(prev_actions_n)
        self.prev_actions = np.zeros((self._prev_actions_n, self.n_joints), dtype=np.float32)
        self._action_ema: np.ndarray | None = None
        self.prev_raw_target: np.ndarray | None = None
        self.qpos: np.ndarray | None = None
        self.qvel: np.ndarray | None = None

    def describe(self) -> str:
        return (f"execute={self.execute} {self.control_hz:.1f}Hz "
                f"action_scale={self.action_scale} ema={self.ema_alpha} "
                f"slow={self.slow}x (max joint speed "
                f"{max_joint_speed_rad_s(self.action_scale, self.control_hz) / self.slow:.2f} rad/s, "
                f"raw clamp ±{self.max_raw_delta}/tick, "
                f"{self.n_interp} sub-targets/tick @ {1.0 / self.sub_dt:.0f} Hz)")

    def boot(self) -> np.ndarray:
        """Initial encoder read + calibration sanity check; on --execute also
        set servo gains and enable torque. Returns the boot qpos (rad)."""
        raw0 = self.bus.read_all()
        qpos = raw_to_rad(raw0, self.jm, self.direction)
        if not (np.all(qpos >= self.xml_low - INIT_RANGE_SLACK)
                and np.all(qpos <= self.xml_high + INIT_RANGE_SLACK)):
            raise SystemExit(
                "ABORT: initial qpos outside MuJoCo joint range; check calibration."
            )
        if self.execute:
            self.bus.set_position_kp(SERVO_POSITION_KP)
            self.bus.set_position_deadzone(SERVO_POSITION_DEADZONE)
            self.bus.enable_torque_all()
        self.prev_raw_target = raw0.copy()
        self.qpos = qpos
        self.qvel = np.zeros(self.n_joints, dtype=np.float64)
        return qpos

    def _write_raw(self, raw: np.ndarray) -> None:
        if self.execute:
            self.bus.write_all(raw, SERVO_SPEED, SERVO_ACCEL)

    def tick(self, action: np.ndarray) -> np.ndarray:
        """Shape and execute one policy action; paces the full wall tick via
        sub-target streaming, then reads the arm back into qpos/qvel. Returns
        the (clipped, EMA-smoothed) action actually applied — log this one."""
        action = np.clip(action.astype(np.float64), -1.0, 1.0)
        if self._action_ema is None:
            self._action_ema = action.copy()
        else:
            self._action_ema = self.ema_alpha * action + (1.0 - self.ema_alpha) * self._action_ema
        action = self._action_ema
        # Mirror sim: the (clipped, smoothed) action becomes the most recent
        # entry in the prev-actions buffer used for the NEXT obs.
        if self._prev_actions_n > 0:
            if self._prev_actions_n > 1:
                self.prev_actions[:-1] = self.prev_actions[1:]
            self.prev_actions[-1] = action.astype(np.float32)

        # Exactly matches training: raw-unit quantization + stiction deadzone.
        target_qpos = action_to_target(self.qpos, action, self.action_scale,
                                       self.xml_low, self.xml_high)
        target_raw = rad_to_raw(target_qpos, self.jm, self.direction)
        target_raw = clamp_raw_delta(self.prev_raw_target, target_raw, self.max_raw_delta)
        stream_sub_targets(self.prev_raw_target, target_raw, self.n_interp,
                           self.sub_dt, self._write_raw)
        self.prev_raw_target = target_raw

        raw = self.bus.read_all()
        new_qpos = raw_to_rad(raw, self.jm, self.direction)
        # Divide by the SIM tick, not the wall tick: under --slow the arm
        # covers the same per-tick delta in more wall time, and sim-time
        # velocities are what the policy saw in training.
        self.qvel = (new_qpos - self.qpos) / self.control_dt
        self.qpos = new_qpos
        return action
