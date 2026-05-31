"""Fit MuJoCo dynamics parameters to minimize real-vs-sim joint RMSE.

Five global parameters tuned against so101.xml defaults (scales=1, tau=xml
default recovers the current sim):
  - scale_kp:           actuator position gain
  - scale_damping:      joint damping (dof_damping)
  - scale_armature:     rotor armature (dof_armature)
  - scale_frictionloss: joint frictionloss (dof_frictionloss)
  - tau:                first-order filter time constant on the actuator,
                        shared across all 6 joints (same hardware).

Requires real-arm CSVs in sysid_logs/real/ and so101.xml position actuators
with timeconst set. Writes sysid_logs/fit.json.

Usage:
    python -m sysid.fit_params
    python -m sysid.fit_params --maxiter 40 --popsize 15
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

from src.base_env import JOINT_NAMES
from sysid.io import OUT_DIR, OUT_DIR_REAL, REPO_ROOT, read_log
from sysid.replay_sim import n_substeps_for, run_one

DEFAULT_XML = REPO_ROOT / "so101" / "scene.xml"

SCALE_NAMES = ["scale_kp", "scale_damping", "scale_armature", "scale_frictionloss"]
SCALE_BOUNDS = [(0.3, 3.0), (0.2, 5.0), (0.2, 5.0), (0.0, 5.0)]
TAU_BOUNDS = (0.005, 0.300)
N_SCALES = len(SCALE_NAMES)
PARAM_NAMES = SCALE_NAMES + ["tau"]
PARAM_BOUNDS = SCALE_BOUNDS + [TAU_BOUNDS]


def snapshot_baseline(model: mujoco.MjModel, actuator_ids: np.ndarray,
                      dofadr: np.ndarray) -> dict:
    return {
        "gainprm0": model.actuator_gainprm[actuator_ids, 0].copy(),
        "biasprm1": model.actuator_biasprm[actuator_ids, 1].copy(),
        "damping":  model.dof_damping[dofadr].copy(),
        "armature": model.dof_armature[dofadr].copy(),
        "friction": model.dof_frictionloss[dofadr].copy(),
    }


def apply_params(model: mujoco.MjModel, baseline: dict, params: np.ndarray,
                 actuator_ids: np.ndarray, dofadr: np.ndarray) -> None:
    s_kp, s_damp, s_arm, s_fric, tau = params
    model.actuator_gainprm[actuator_ids, 0] = baseline["gainprm0"] * s_kp
    model.actuator_biasprm[actuator_ids, 1] = baseline["biasprm1"] * s_kp
    model.dof_damping[dofadr] = baseline["damping"] * s_damp
    model.dof_armature[dofadr] = baseline["armature"] * s_arm
    model.dof_frictionloss[dofadr] = baseline["friction"] * s_fric
    model.actuator_dynprm[actuator_ids, 0] = tau


def total_loss(params: np.ndarray, model: mujoco.MjModel, data: mujoco.MjData,
               baseline: dict, actuator_ids: np.ndarray, dofadr: np.ndarray,
               qposadr: np.ndarray, n_substeps: int,
               real_data: list[tuple[str, np.ndarray, np.ndarray]]) -> float:
    apply_params(model, baseline, params, actuator_ids, dofadr)
    total = 0.0
    for _name, target, pos_r in real_data:
        pos_s = run_one(model, data, target, qposadr, n_substeps)
        diff = pos_s - pos_r
        total += float(np.mean(diff * diff))
    return total / len(real_data)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default=str(DEFAULT_XML))
    p.add_argument("--maxiter", type=int, default=25)
    p.add_argument("--popsize", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not OUT_DIR_REAL.exists():
        raise SystemExit(f"no real logs in {OUT_DIR_REAL}; run sysid.record_real first")
    real_csvs = sorted(OUT_DIR_REAL.glob("*.csv"))
    if not real_csvs:
        raise SystemExit("no real CSVs to fit against")

    real_data: list[tuple[str, np.ndarray, np.ndarray]] = []
    for path in real_csvs:
        _t, target, pos = read_log(path)
        real_data.append((path.stem, target, pos))

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
    qposadr = model.jnt_qposadr[joint_ids]
    dofadr = model.jnt_dofadr[joint_ids]
    actuator_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINT_NAMES],
        dtype=np.int64,
    )
    # Sanity: all 6 actuators must carry a first-order filter activation, else
    # the tau parameters do nothing and the search is wasted.
    filter_types = (int(mujoco.mjtDyn.mjDYN_FILTER), int(mujoco.mjtDyn.mjDYN_FILTEREXACT))
    for aid, jname in zip(actuator_ids, JOINT_NAMES):
        assert int(model.actuator_dyntype[aid]) in filter_types, (
            f"actuator {jname!r}: dyntype must be filter/filterexact for tau "
            f"fitting; got {int(model.actuator_dyntype[aid])}"
        )
    baseline = snapshot_baseline(model, actuator_ids, dofadr)
    baseline_tau = float(model.actuator_dynprm[actuator_ids[0], 0])
    n_substeps = n_substeps_for(model)

    x0 = np.array([1.0, 1.0, 1.0, 1.0, baseline_tau])
    loss_args = (model, data, baseline, actuator_ids, dofadr,
                 qposadr, n_substeps, real_data)
    baseline_loss = total_loss(x0, *loss_args)
    print(f"baseline loss (scales=1, tau={baseline_tau:.3f}, "
          f"{len(real_data)} trajectories): {baseline_loss:.5f}")

    result = differential_evolution(
        total_loss, PARAM_BOUNDS, args=loss_args,
        maxiter=args.maxiter, popsize=args.popsize, seed=args.seed,
        polish=True, tol=1e-4, disp=True,
    )

    print(f"\nbest loss: {result.fun:.5f}  (baseline: {baseline_loss:.5f}, "
          f"improvement: {(1 - result.fun / baseline_loss) * 100:.1f}%)")
    for name, val in zip(PARAM_NAMES, result.x):
        print(f"  {name} = {val:.4f}")

    apply_params(model, baseline, result.x, actuator_ids, dofadr)
    out = {
        "joint_order": JOINT_NAMES,
        "scales": {k: float(v) for k, v in zip(SCALE_NAMES, result.x[:N_SCALES])},
        "tau": float(result.x[-1]),
        "absolute": {
            "kp_per_joint":       model.actuator_gainprm[actuator_ids, 0].tolist(),
            "damping_per_joint":  model.dof_damping[dofadr].tolist(),
            "armature_per_joint": model.dof_armature[dofadr].tolist(),
            "friction_per_joint": model.dof_frictionloss[dofadr].tolist(),
            "tau":                float(model.actuator_dynprm[actuator_ids[0], 0]),
        },
        "baseline_loss": float(baseline_loss),
        "best_loss": float(result.fun),
    }
    out_path = OUT_DIR / "fit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
