"""Sim-side model of the Feetech servo's internal motion profile.

In position mode the STS3215 firmware never jumps its setpoint to a newly
commanded position. It runs a trapezoidal profile: the internal setpoint
accelerates at the accel-register rate toward the speed-register cap and
decelerates so as to arrive at the goal at rest; the internal P loop then
tracks that moving setpoint. With SERVO_ACCEL deliberately low (see
real/twin/constants.py) this ramp — not the P loop — is the dominant
actuator dynamic, so a sim that writes the raw target straight into
`data.ctrl` is structurally too fast and any sysid fit on top of it has to
fake the ramp with inflated joint damping/armature.

`ServoProfile` reproduces the chain the real bus sees each control tick:

    policy/sysid target (15 Hz)
      → linear sub-target interpolation across the tick
        (real: `stream_sub_targets` at INTERP_HZ; sim: per MuJoCo substep)
      → trapezoidal setpoint profile (firmware accel/speed registers)
      → `data.ctrl` of the MuJoCo <position> actuator (internal P loop)

The profile is open-loop in the MuJoCo state: like the firmware, it plans
from its own setpoint, not from the measured joint position.

Used by the training envs (src.base_env, src.reach_env) and by
sysid.replay_sim, so training, eval, and sysid replay all share one model of
the command shaping.
"""

import numpy as np

from real.twin.constants import SERVO_ACCEL, SERVO_SPEED
from src.units import SERVO_ACCEL_UNIT_RAD_S2, SERVO_SPEED_UNIT_RAD_S


class ServoProfile:
    """Vectorized trapezoidal setpoint profile for n position-mode servos."""

    def __init__(self, n_joints: int, speed: int = SERVO_SPEED, accel: int = SERVO_ACCEL):
        assert speed > 0 and accel > 0, f"speed={speed} accel={accel} must be positive"
        self.v_max = speed * SERVO_SPEED_UNIT_RAD_S
        self.a_max = accel * SERVO_ACCEL_UNIT_RAD_S2
        self.pos = np.zeros(n_joints, dtype=np.float64)
        self.vel = np.zeros(n_joints, dtype=np.float64)

    def reset(self, qpos: np.ndarray) -> None:
        """Start the profile at rest at `qpos` (servo at standstill)."""
        self.pos[:] = qpos
        self.vel[:] = 0.0

    def step(self, goal: np.ndarray, dt: float) -> np.ndarray:
        """Advance the setpoint by `dt` toward `goal`; returns the new setpoint.

        Each call re-plans from the current (pos, vel) — exactly what the
        firmware does when a fresh sub-target arrives mid-motion: velocity
        ramps at ±a_max toward the trapezoid's velocity-vs-remaining-distance
        curve, capped at v_max.
        """
        dist = goal - self.pos
        v_des = np.sign(dist) * np.minimum(self.v_max, np.sqrt(2.0 * self.a_max * np.abs(dist)))
        self.vel += np.clip(v_des - self.vel, -self.a_max * dt, self.a_max * dt)
        self.pos += self.vel * dt
        return self.pos.copy()

    def tick(self, prev_target: np.ndarray, target: np.ndarray,
             n_substeps: int, dt: float) -> np.ndarray:
        """Setpoints for one control tick, shape (n_substeps, n_joints).

        The goal slides linearly from `prev_target` to `target` across the
        tick, mirroring `stream_sub_targets` on the real bus; the profile
        chases it. Feed row k into `data.ctrl` before MuJoCo substep k.
        """
        out = np.empty((n_substeps, len(self.pos)), dtype=np.float64)
        for k in range(n_substeps):
            alpha = (k + 1) / n_substeps
            out[k] = self.step(prev_target + alpha * (target - prev_target), dt)
        return out
