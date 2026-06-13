"""Fit MuJoCo dynamics parameters to minimize real-vs-sim joint RMSE.

Replays every real-arm trajectory through sysid.replay_sim.run_one — which
includes the per-tick raw-delta clamp and the firmware motion-profile model
(src/servo_profile.py) — so sim sees the commanded input the bus actually
executed and the fit no longer has to fake the servo's accel ramp with
inflated damping.

Parameters (scales are relative to the so101.xml values, so all-ones plus
frictionloss=0 and tau=xml recovers the current sim):
  - scale_kp@<K>:       actuator position gain, one scale per distinct real
                        servo P gain in SERVO_POSITION_KP (the xml kp baseline
                        is already proportional to the real per-joint gain)
  - scale_kv:           actuator velocity gain (motor back-EMF damping;
                        same hardware on every joint, shared)
  - scale_damping:      joint damping (dof_damping)
  - scale_armature:     rotor armature (dof_armature)
  - frictionloss:       joint dry friction, absolute N·m (the xml baseline is
                        0.0, so a scale would be a dead parameter). NOTE: with
                        the profile + tau filter modeling the lag, this fits to
                        ~0 (2026-06-13: 0.003 N·m, negligible vs ~1-3 N·m servo
                        torque) — the loss is dominated by the dynamic
                        trajectories and the quasi-static staircase that would
                        excite stiction is too small a slice to move it. Drop it
                        (fix at 0) next refit unless stiction fidelity matters,
                        in which case weight staircase/stretch_hold up instead.
  - tau:                first-order filter time constant on the actuator,
                        shared across all 6 joints (same hardware).

Requires real-arm CSVs in sysid_logs/real/ — recorded at the CURRENT servo
settings (SERVO_POSITION_KP / SERVO_SPEED / SERVO_ACCEL), since both the
profile model and the kp grouping bake those in. Writes sysid_logs/fit.json
with the settings it assumed.

Usage:
    python -m sysid.fit_params
    python -m sysid.fit_params --maxiter 40 --popsize 15
"""

import argparse
import json

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

from real.twin.constants import SERVO_ACCEL, SERVO_POSITION_KP, SERVO_SPEED
from src.base_env import JOINT_NAMES
from sysid.io import OUT_DIR, OUT_DIR_REAL, REPO_ROOT, read_log
from sysid.replay_sim import max_delta_rad_from_config, n_substeps_for, run_one

DEFAULT_XML = REPO_ROOT / "so101" / "scene.xml"

# One kp scale per distinct real-servo P gain, e.g. Kp 8 (pan, gripper) and
# Kp 32 (the rest). Joints sharing a gain share a scale.
KP_GROUPS: list[tuple[int, list[int]]] = [
    (kp, [j for j, k in enumerate(SERVO_POSITION_KP) if k == kp])
    for kp in sorted(set(SERVO_POSITION_KP))
]

KP_PARAM_NAMES = [f"scale_kp@{kp}" for kp, _ in KP_GROUPS]
SHARED_SCALE_NAMES = ["scale_kv", "scale_damping", "scale_armature"]
PARAM_NAMES = KP_PARAM_NAMES + SHARED_SCALE_NAMES + ["frictionloss", "tau"]
PARAM_BOUNDS = (
    [(0.3, 3.0)] * len(KP_GROUPS)   # kp scales
    + [(0.05, 3.0),                 # kv scale
       (0.01, 2.0),                 # damping scale
       (0.01, 2.5)]                 # armature scale
    + [(0.0, 0.5),                  # frictionloss [N·m]
       (0.005, 0.300)]              # tau [s]
)


def snapshot_baseline(model: mujoco.MjModel, actuator_ids: np.ndarray,
                      dofadr: np.ndarray) -> dict:
    return {
        "gainprm0": model.actuator_gainprm[actuator_ids, 0].copy(),
        "biasprm1": model.actuator_biasprm[actuator_ids, 1].copy(),
        "biasprm2": model.actuator_biasprm[actuator_ids, 2].copy(),
        "damping":  model.dof_damping[dofadr].copy(),
        "armature": model.dof_armature[dofadr].copy(),
    }


def apply_params(model: mujoco.MjModel, baseline: dict, params: np.ndarray,
                 actuator_ids: np.ndarray, dofadr: np.ndarray) -> None:
    kp_scales = params[:len(KP_GROUPS)]
    s_kv, s_damp, s_arm, friction, tau = params[len(KP_GROUPS):]
    for s_kp, (_kp, joints) in zip(kp_scales, KP_GROUPS):
        ids = actuator_ids[joints]
        model.actuator_gainprm[ids, 0] = baseline["gainprm0"][joints] * s_kp
        model.actuator_biasprm[ids, 1] = baseline["biasprm1"][joints] * s_kp
    model.actuator_biasprm[actuator_ids, 2] = baseline["biasprm2"] * s_kv
    model.dof_damping[dofadr] = baseline["damping"] * s_damp
    model.dof_armature[dofadr] = baseline["armature"] * s_arm
    model.dof_frictionloss[dofadr] = friction
    model.actuator_dynprm[actuator_ids, 0] = tau


def total_loss(params: np.ndarray, model: mujoco.MjModel, data: mujoco.MjData,
               baseline: dict, actuator_ids: np.ndarray, dofadr: np.ndarray,
               qposadr: np.ndarray, n_substeps: int, max_delta_rad: float,
               real_data: list[tuple[str, np.ndarray, np.ndarray]]) -> float:
    apply_params(model, baseline, params, actuator_ids, dofadr)
    total = 0.0
    for _name, target, pos_r in real_data:
        pos_s = run_one(model, data, target, qposadr, n_substeps, max_delta_rad)
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

    x0 = np.array([1.0] * len(KP_GROUPS) + [1.0, 1.0, 1.0, 0.0, baseline_tau])
    loss_args = (model, data, baseline, actuator_ids, dofadr,
                 qposadr, n_substeps, max_delta_rad_from_config(), real_data)
    baseline_loss = total_loss(x0, *loss_args)
    print(f"kp groups: {[(kp, [JOINT_NAMES[j] for j in joints]) for kp, joints in KP_GROUPS]}")
    print(f"baseline loss (scales=1, friction=0, tau={baseline_tau:.3f}, "
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
        "params": {k: float(v) for k, v in zip(PARAM_NAMES, result.x)},
        "absolute": {
            "kp_per_joint":       model.actuator_gainprm[actuator_ids, 0].tolist(),
            "kv_per_joint":       (-model.actuator_biasprm[actuator_ids, 2]).tolist(),
            "damping_per_joint":  model.dof_damping[dofadr].tolist(),
            "armature_per_joint": model.dof_armature[dofadr].tolist(),
            "friction_per_joint": model.dof_frictionloss[dofadr].tolist(),
            "tau":                float(model.actuator_dynprm[actuator_ids[0], 0]),
        },
        # Servo settings the fit assumed, via the profile model and kp groups.
        # If these no longer match real/twin/constants.py, re-record and refit.
        "servo_settings": {
            "position_kp": list(SERVO_POSITION_KP),
            "speed": SERVO_SPEED,
            "accel": SERVO_ACCEL,
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
