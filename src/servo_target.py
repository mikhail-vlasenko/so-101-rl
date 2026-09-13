"""Real-bus previous-target limiting in simulated joint coordinates.

The policy requests a delta from the measured joint pose, while the bus limits
changes from its last commanded raw target. Convert through the same follower
mapping, encoder bias, and gravity-compliance functions as ArmLoop, clamp in raw
units, then decode the limited command for the simulated servo profile. This
also preserves real raw quantization and calibrated endpoint clipping.

Calibration files describe the deployment arm; they are not policy observation
DR. The limiter never alters the physical MjData. Call reset at each episode
start, and advance once per control tick, including when the profile is disabled.
"""

import mujoco
import numpy as np

from real.calib.calibration import load_calibration, load_compliance
from real.calib.compliance import encoder_raw_to_true, true_to_encoder_raw
from real.twin.control import clamp_raw_delta
from real.twin.mapping import FOLLOWER_CALIBRATION_PATH, load_joint_maps
from src.units import max_raw_delta_per_step


class ServoTargetLimiter:
    def __init__(self, model: mujoco.MjModel, action_scale: float):
        self.model = model
        self.data = mujoco.MjData(model)
        self.jm = load_joint_maps(model, FOLLOWER_CALIBRATION_PATH)
        self.direction = np.ones(len(self.jm.items), dtype=np.int8)
        self.qpos_bias = load_calibration()
        self.compliance = load_compliance()
        self.max_raw_delta = max_raw_delta_per_step(action_scale)
        self.previous_raw = None

    def _encode(self, target):
        return true_to_encoder_raw(
            self.model, self.data, self.jm, self.direction, target,
            self.qpos_bias, self.compliance)

    def reset(self, qpos):
        self.previous_raw = self._encode(qpos)

    def limit(self, target):
        assert self.previous_raw is not None, "reset the target limiter before stepping"
        requested_raw = self._encode(target)
        self.previous_raw = clamp_raw_delta(
            self.previous_raw, requested_raw, self.max_raw_delta)
        return encoder_raw_to_true(
            self.model, self.data, self.jm, self.direction, self.previous_raw,
            self.qpos_bias, self.compliance)
