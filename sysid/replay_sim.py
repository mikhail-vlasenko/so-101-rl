"""Replay a sysid trajectory in MuJoCo and log measured qpos.

Same control rate and observation schema as record_real.py so the resulting
CSVs can be diff'd directly by analyze.py.

Usage:
    python -m sysid.replay_sim --traj sweep_shoulder_pan
    python -m sysid.replay_sim --all
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from src.base_env import JOINT_NAMES
from sysid.io import OUT_DIR_SIM, REPO_ROOT, write_log
from sysid.trajectories import SYSID_DT, SYSID_HZ, TRAJECTORIES

DEFAULT_XML = REPO_ROOT / "so101" / "scene.xml"


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
            qposadr: np.ndarray, n_substeps: int) -> np.ndarray:
    """Drive actuators with traj[i] each step, return measured qpos array."""
    n_joints = traj.shape[1]
    mujoco.mj_resetData(model, data)
    data.qpos[qposadr] = traj[0]
    data.ctrl[:n_joints] = traj[0]
    if model.na > 0:
        assert model.na == n_joints, f"na={model.na} but n_joints={n_joints}"
        data.act[:] = traj[0]
    mujoco.mj_forward(model, data)

    pos = np.empty_like(traj)
    for i in range(len(traj)):
        data.ctrl[:n_joints] = traj[i]
        for _ in range(n_substeps):
            mujoco.mj_step(model, data)
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

    for name in names:
        traj = TRAJECTORIES[name]
        pos = run_one(model, data, traj, qposadr, n_substeps)
        out = OUT_DIR_SIM / f"{name}.csv"
        write_log(out, target=traj, pos=pos, dt=SYSID_DT)
        print(f"sim {name}: {len(traj)} steps → {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
