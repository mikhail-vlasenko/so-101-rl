"""Diagnostic: is the lift failure the servo profile stalling under load, or
just honest accel-limited slowness (and is grip force the real bottleneck)?

The motion profile (src/servo_profile.py) is open-loop in the MuJoCo state: it
plans from its own setpoint `self.pos`, while the per-tick goal is recomputed
from the *measured* joint position, `target = qpos + action*scale`. So the
profile sees `dist = (qpos - self.pos) + action*scale`. Under load the P-loop
lets `qpos` lag the commanded setpoint `self.pos`; if that lag exceeds
`action_scale`, a full-forward command yields a *negative* dist and the
setpoint backs off instead of pushing — the arm would be unable to drive into
load no matter how long it's commanded.

These tests don't assert a pass/fail contract; they print the numbers that
tell us whether accel is even the right knob to tune. Run with `-s`:

    pytest tests/test_profile_stall.py -s
"""

import os

import mujoco
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.base_env import JOINT_NAMES
from src.lift_env import SO101LiftEnv
from src.units import action_to_target

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIFT_IDX = JOINT_NAMES.index("shoulder_lift")
ELBOW_IDX = JOINT_NAMES.index("elbow_flex")
GRIP_IDX = JOINT_NAMES.index("gripper")


def _make_env(action_scale=None):
    with initialize_config_dir(config_dir=os.path.join(REPO_ROOT, "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=["env=lift", "shaping=none"])
    env_cfg = OmegaConf.to_container(cfg.lift_env, resolve=True)
    if action_scale is not None:
        env_cfg["action_scale"] = action_scale
    return SO101LiftEnv(env_cfg=env_cfg,
                        xml_path=os.path.join(REPO_ROOT, "so101/scene_lift.xml"))


def _drive(env, action, n_ticks):
    """Step a constant action, returning per-tick (qpos, profile_setpoint)."""
    qpos_log, set_log = [], []
    for _ in range(n_ticks):
        env.step(action.astype(np.float32))
        qpos_log.append(env.data.qpos[env.joint_qposadr].copy())
        set_log.append(env._servo_profile.pos.copy())
    return np.array(qpos_log), np.array(set_log)


def test_profile_divergence_under_gravity_load():
    """Lift the gravity-loaded shoulder/elbow up at full command; report the
    setpoint-vs-measured lag and whether the profile dist ever goes negative."""
    env = _make_env()
    env.reset(seed=0)
    scale = env.action_scale

    # Full positive command on the two main lifters, zero elsewhere. Sign is in
    # joint-rad space; sign of "up" depends on the model, so try both and keep
    # whichever actually raises the wrist (the loaded direction).
    qpos, setp = _drive(env, _lift_action(+1.0), 60)
    lag = setp - qpos  # setpoint ahead of measured => positive when loaded

    # What the profile would see as dist on the next tick for a full command:
    #   dist = (qpos - setpoint) + scale = scale - lag
    margin_lift = scale - lag[:, LIFT_IDX]
    margin_elbow = scale - lag[:, ELBOW_IDX]

    print("\n--- profile divergence under load (shoulder_lift) ---")
    print(f"action_scale (rad)            : {scale:.4f}")
    print(f"steady setpoint-qpos lag (rad): lift={lag[-1, LIFT_IDX]:+.4f}  "
          f"elbow={lag[-1, ELBOW_IDX]:+.4f}")
    print(f"min dist-margin (scale-lag)   : lift={margin_lift.min():+.4f}  "
          f"elbow={margin_elbow.min():+.4f}   (<0 => profile reverses)")
    print(f"shoulder_lift qpos travel     : {qpos[-1, LIFT_IDX] - qpos[0, LIFT_IDX]:+.4f} rad "
          f"over {len(qpos)} ticks")
    print(f"elbow_flex   qpos travel      : {qpos[-1, ELBOW_IDX] - qpos[0, ELBOW_IDX]:+.4f} rad")


def test_achieved_vs_commanded_speed():
    """Quantify authority loss: steady per-tick joint motion vs the commanded
    action_scale, free of cube contact (gripper open, arm sweeping a joint)."""
    env = _make_env()
    env.reset(seed=1)
    scale = env.action_scale

    # Drive shoulder_pan (gravity-unloaded about vertical axis) to isolate the
    # accel ramp from gravity lag.
    action = np.zeros(env.n_joints)
    action[JOINT_NAMES.index("shoulder_pan")] = 1.0
    qpos, _ = _drive(env, action, 40)
    per_tick = np.diff(qpos[:, JOINT_NAMES.index("shoulder_pan")])

    print("\n--- achieved vs commanded per-tick motion (unloaded pan) ---")
    print(f"commanded action_scale (rad/tick): {scale:.4f}")
    print(f"steady achieved (rad/tick)       : {per_tick[-5:].mean():.4f}")
    print(f"fraction of commanded            : {per_tick[-5:].mean() / scale:.2%}")
    print(f"ticks to reach 90% of steady     : "
          f"{_ticks_to_frac(per_tick, 0.9)}")


def test_gripper_holding_force():
    """Is grip the bottleneck rather than accel? Close the gripper hard and
    report the actuator torque it can hold vs the cube weight it must support."""
    env = _make_env()
    env.reset(seed=2)

    action = np.zeros(env.n_joints)
    action[GRIP_IDX] = -1.0  # close (sign checked below via travel direction)
    qpos, setp = _drive(env, action, 60)

    grip_id = env.joint_ids[GRIP_IDX]
    kp = env.model.actuator_gainprm[GRIP_IDX, 0]  # position actuator P gain
    steady_err = abs(setp[-1, GRIP_IDX] - qpos[-1, GRIP_IDX])
    hold_torque = kp * steady_err

    cube_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_mass = env.model.body_mass[cube_id]
    cube_weight = cube_mass * abs(env.model.opt.gravity[2])

    print("\n--- gripper holding capacity ---")
    print(f"gripper kp                    : {kp:.2f}")
    print(f"steady setpoint-qpos err (rad): {steady_err:.4f}")
    print(f"=> holdable joint torque (N·m): {hold_torque:.4f}")
    print(f"cube mass / weight            : {cube_mass*1e3:.1f} g / {cube_weight:.4f} N")
    print(f"gripper travel (close)        : {qpos[-1, GRIP_IDX] - qpos[0, GRIP_IDX]:+.4f} rad")


def _drive_raw(env, action, n_ticks):
    """Old behavior: write the raw tick target straight to data.ctrl, no
    profile. Returns per-tick (qpos, ctrl)."""
    import mujoco
    qpos_log, ctrl_log = [], []
    for _ in range(n_ticks):
        current = env.data.qpos[env.joint_qposadr].copy()
        target = action_to_target(current, action.astype(np.float32),
                                  env.action_scale, env.joint_low, env.joint_high)
        env.data.ctrl[:env.n_joints] = target
        for _ in range(env.n_substeps):
            mujoco.mj_step(env.model, env.data)
        qpos_log.append(env.data.qpos[env.joint_qposadr].copy())
        ctrl_log.append(target.copy())
    return np.array(qpos_log), np.array(ctrl_log)


def test_profile_vs_raw_ctrl_ab():
    """Does the accel ramp itself change dynamics, or is the actuator fit
    (ctrl->qpos following error) the dominant slowdown? A/B the same commands
    with profile on vs. raw target straight to ctrl."""
    pan = JOINT_NAMES.index("shoulder_pan")

    env = _make_env(); env.reset(seed=7)
    q_prof, _ = _drive(env, _pan_action(), 40)
    env = _make_env(); env.reset(seed=7)
    q_raw, _ = _drive_raw(env, _pan_action(), 40)

    prof_steady = np.diff(q_prof[:, pan])[-5:].mean()
    raw_steady = np.diff(q_raw[:, pan])[-5:].mean()

    print("\n--- profile vs raw-ctrl (unloaded pan, per-tick rad) ---")
    print(f"commanded action_scale : {env.action_scale:.4f}")
    print(f"profile steady          : {prof_steady:.4f}  ({prof_steady/env.action_scale:.1%})")
    print(f"raw-ctrl steady         : {raw_steady:.4f}  ({raw_steady/env.action_scale:.1%})")
    print(f"profile / raw           : {prof_steady/raw_steady:.2f}x")

    env = _make_env(); env.reset(seed=7)
    ql_prof, _ = _drive(env, _lift_action(+1.0), 60)
    env = _make_env(); env.reset(seed=7)
    ql_raw, _ = _drive_raw(env, _lift_action(+1.0), 60)
    print("\n--- profile vs raw-ctrl (loaded lift, total travel rad/60 ticks) ---")
    print(f"shoulder_lift  profile={ql_prof[-1,LIFT_IDX]-ql_prof[0,LIFT_IDX]:+.3f}  "
          f"raw={ql_raw[-1,LIFT_IDX]-ql_raw[0,LIFT_IDX]:+.3f}")
    print(f"elbow_flex     profile={ql_prof[-1,ELBOW_IDX]-ql_prof[0,ELBOW_IDX]:+.3f}  "
          f"raw={ql_raw[-1,ELBOW_IDX]-ql_raw[0,ELBOW_IDX]:+.3f}")


def test_action_scale_sweep():
    """What does raising action_scale buy? Sweep it through the loaded-lift and
    unloaded-pan drives and report margin (scale - lag), authority %, and the
    real-arm peak-speed ceiling it implies (action_scale * control_hz)."""
    pan = JOINT_NAMES.index("shoulder_pan")
    control_hz = 15.0
    print("\n--- action_scale sweep ---")
    print(f"{'scale':>6} {'lift_lag':>9} {'margin':>8} {'pan %cmd':>9} "
          f"{'lift travel':>12} {'real v_max(rad/s)':>17}")
    for scale in (0.07, 0.10, 0.14, 0.20):
        env = _make_env(action_scale=scale); env.reset(seed=0)
        ql, sl = _drive(env, _lift_action(+1.0), 60)
        lag = (sl - ql)[-1, LIFT_IDX]

        env = _make_env(action_scale=scale); env.reset(seed=1)
        qp, _ = _drive(env, _pan_action(), 40)
        pan_pct = np.diff(qp[:, pan])[-5:].mean() / scale

        lift_travel = ql[-1, LIFT_IDX] - ql[0, LIFT_IDX]
        print(f"{scale:>6.2f} {lag:>+9.4f} {scale-lag:>+8.4f} {pan_pct:>8.1%} "
              f"{lift_travel:>+12.3f} {scale*control_hz:>17.2f}")


def _pan_action():
    action = np.zeros(len(JOINT_NAMES))
    action[JOINT_NAMES.index("shoulder_pan")] = 1.0
    return action


def _patch_old_sysid(env):
    """Overwrite the loaded model's actuator/joint params with the pre-sysid
    (a8dd00e) fit: uniform kp=77.5, kv=2.731, timeconst=0.025, damping=3.86,
    armature=0.242, frictionloss=0. Keeps geometry, profile, everything else."""
    for name in JOINT_NAMES:
        a = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        env.model.actuator_gainprm[a, 0] = 77.5
        env.model.actuator_biasprm[a, 1] = -77.5
        env.model.actuator_biasprm[a, 2] = -2.731
        env.model.actuator_dynprm[a, 0] = 0.025
    for d in env.joint_dofadr:
        env.model.dof_damping[d] = 3.86
        env.model.dof_armature[d] = 0.242
        env.model.dof_frictionloss[d] = 0.0


def test_old_sysid_values_authority():
    """Keep the profile + everything; only swap actuator params back to the
    pre-sysid fit. Does authority recover? If so the refit, not the profile,
    is what broke lift."""
    pan = JOINT_NAMES.index("shoulder_pan")

    def measure(env):
        env.reset(seed=0)
        ql, sl = _drive(env, _lift_action(+1.0), 60)
        lag = (sl - ql)[-1, LIFT_IDX]
        env.reset(seed=1)
        qp, _ = _drive(env, _pan_action(), 40)
        pan_pct = np.diff(qp[:, pan])[-5:].mean() / env.action_scale
        return lag, pan_pct, ql[-1, LIFT_IDX] - ql[0, LIFT_IDX]

    env = _make_env()
    cur = measure(env)
    env = _make_env(); _patch_old_sysid(env)
    old = measure(env)

    print("\n--- current refit vs pre-sysid actuator params (profile kept) ---")
    print(f"{'':16}{'lift_lag':>10}{'margin':>9}{'pan %cmd':>10}{'lift travel':>13}")
    for label, (lag, pct, travel) in (("current refit", cur), ("old sysid fit", old)):
        print(f"{label:16}{lag:>+10.4f}{env.action_scale-lag:>+9.4f}{pct:>9.1%}{travel:>+13.3f}")


def _lift_action(sign):
    action = np.zeros(len(JOINT_NAMES))
    action[LIFT_IDX] = sign
    action[ELBOW_IDX] = sign
    return action


def _ticks_to_frac(per_tick, frac):
    steady = per_tick[-5:].mean()
    for i, v in enumerate(per_tick):
        if v >= frac * steady:
            return i + 1
    return len(per_tick)
