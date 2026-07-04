"""Replay a sysid trajectory in MuJoCo and log measured qpos.

Same control rate and observation schema as record_real.py so the resulting
CSVs can be diff'd directly by analyze.py. Targets go through the same
command shaping as the real bus: the per-tick raw-delta clamp
(real/twin/control.clamp_raw_delta — binding whenever a trajectory commands
more than ~1.1 rad/s, e.g. the large-amplitude chirp segments), per-tick
sub-target interpolation, and the firmware's trapezoidal setpoint profile
(src/servo_profile.py).

Usage:
    python -m sysid.replay_sim --traj sweep_shoulder_pan
    python -m sysid.replay_sim --all
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from omegaconf import OmegaConf

from src.base_env import JOINT_NAMES
from src.servo_profile import ServoProfile
from src.units import max_raw_delta_per_step, raw_units_to_rad
from sysid.io import OUT_DIR_SIM, REPO_ROOT, write_log
from sysid.trajectories import SETTLE_S, SYSID_DT, SYSID_HZ, TRAJECTORIES

DEFAULT_XML = REPO_ROOT / "so101" / "scene.xml"
CONFIG_YAML = REPO_ROOT / "conf" / "config.yaml"


def max_delta_rad_from_config() -> float:
    """The per-tick target-delta clamp record_real applies on the bus, in rad."""
    action_scale = float(OmegaConf.load(CONFIG_YAML)["action_scale"])
    return float(raw_units_to_rad(max_raw_delta_per_step(action_scale)))


def clamp_traj(traj: np.ndarray, max_delta_rad: float) -> np.ndarray:
    """Rad-domain mirror of the real bus's clamp_raw_delta: limit each tick's
    commanded move to ±max_delta_rad relative to the previous (clamped)
    command. Differs from the raw-int implementation only by sub-raw-unit
    (<0.0016 rad) rounding."""
    out = np.empty_like(traj)
    cmd = traj[0].copy()
    out[0] = cmd
    for i in range(1, len(traj)):
        cmd = cmd + np.clip(traj[i] - cmd, -max_delta_rad, max_delta_rad)
        out[i] = cmd
    return out


def n_substeps_for(model: mujoco.MjModel) -> int:
    """Return substeps such that substeps * model.opt.timestep == 1/SYSID_HZ."""
    control_dt = 1.0 / SYSID_HZ
    n = int(round(control_dt / model.opt.timestep))
    err = abs(n * model.opt.timestep - control_dt)
    assert err < 1e-6, (
        f"timestep {model.opt.timestep} does not evenly divide control dt "
        f"{control_dt} (err={err}); fix so101.xml or SYSID_HZ"
    )
    return n


def run_one(model: mujoco.MjModel, data: mujoco.MjData, traj: np.ndarray,
            qposadr: np.ndarray, n_substeps: int, max_delta_rad: float) -> np.ndarray:
    """Drive actuators with the clamped traj[i] each tick through the
    servo-profile model, return measured qpos array."""
    n_joints = traj.shape[1]
    mujoco.mj_resetData(model, data)
    data.qpos[qposadr] = traj[0]
    data.ctrl[:n_joints] = traj[0]
    if model.na > 0:
        assert model.na == n_joints, f"na={model.na} but n_joints={n_joints}"
        data.act[:] = traj[0]
    mujoco.mj_forward(model, data)

    # Settle at traj[0] for SETTLE_S like record_real does on the bus: the
    # real arm's first logged tick starts from its gravity-sagged equilibrium,
    # not from traj[0] exactly, and without this hold every gravity-loaded
    # trajectory would open with a sag transient the fit mistakes for dynamics.
    for _ in range(int(round(SETTLE_S / model.opt.timestep))):
        mujoco.mj_step(model, data)

    cmds = clamp_traj(traj, max_delta_rad)
    profile = ServoProfile(n_joints)
    profile.reset(cmds[0])
    prev_cmd = cmds[0]
    pos = np.empty_like(traj)
    for i in range(len(cmds)):
        setpoints = profile.tick(prev_cmd, cmds[i], n_substeps, model.opt.timestep)
        for k in range(n_substeps):
            data.ctrl[:n_joints] = setpoints[k]
            mujoco.mj_step(model, data)
        prev_cmd = cmds[i]
        pos[i] = data.qpos[qposadr]
    return pos


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--traj", default=None,
                   help="Trajectory name to replay; omit if --all is set.")
    p.add_argument("--all", action="store_true",
                   help="Replay every trajectory in TRAJECTORIES.")
    p.add_argument("--xml", default=str(DEFAULT_XML))
    args = p.parse_args()

    if bool(args.traj) == bool(args.all):
        raise SystemExit("specify exactly one of --traj <name> or --all")

    if args.all:
        names = sorted(TRAJECTORIES.keys())
    else:
        if args.traj not in TRAJECTORIES:
            raise SystemExit(
                f"unknown trajectory {args.traj!r}; "
                f"available: {sorted(TRAJECTORIES)}"
            )
        names = [args.traj]

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
    qposadr = model.jnt_qposadr[joint_ids]
    n_substeps = n_substeps_for(model)
    max_delta_rad = max_delta_rad_from_config()

    for name in names:
        traj = TRAJECTORIES[name]
        pos = run_one(model, data, traj, qposadr, n_substeps, max_delta_rad)
        out = OUT_DIR_SIM / f"{name}.csv"
        write_log(out, target=traj, pos=pos, dt=SYSID_DT)
        print(f"sim {name}: {len(traj)} steps → {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
