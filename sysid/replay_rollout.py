"""Diagnose a recorded real-arm lift rollout (rollouts/rollout_lift_*.csv)
against sim, to localize sim-to-real behavior gaps (e.g. floor pressing).

Three complementary probes, all offline (no hardware):

  replay  — open-loop: feed the CSV's recorded actions through the training env
            dynamics from the recorded start state. The obs pipeline plays no
            role, so real-vs-sim qpos divergence isolates actuation/dynamics
            mismatch: "did the same commands land in the same place?"
  policy  — closed-loop: run the checkpoint in sim from the exact recorded
            start (arm pose + cube spawn). Shows what the policy does when the
            whole observation chain is sim-consistent — the sim twin of the
            recorded episode.
  obsdiff — counterfactual: rebuild each recorded step's observation from the
            recorded encoder qpos with FK-consistent marker poses (what the
            markers *should* have read if the camera agreed with the encoder
            chain), re-predict, and diff against the recorded action. A large
            delta means the camera marker obs — not qpos/qvel/cube — is what
            changed the policy's mind on the real arm.

The cube spawn is reproduced from --seed exactly like real/rollout_lift.py
(same rng call sequence) and cross-checked against the CSV's first cube row.

Usage:
    python -m sysid.replay_rollout --csv rollouts/rollout_lift_1783245166.csv \
        --model logs/ppo_lift/stage3_v2_run246.zip --seed 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from real.rollout_common import load_policy
from src.base_env import (
    JOINT_NAMES,
    MARKER_AGE_CAP_S,
    MARKER_SITE_NAMES,
    N_MARKERS,
    marker_world_poses,
    markers_visible,
    obs_dim_for,
    sample_cube_orientation,
    tag_cam_world_pos,
)
from src.lift_env import SO101LiftEnv
from src.train import make_env

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "logs" / "ppo_lift"
LIFT_TASK_ID = 0.0

# The recorded rollout's first CSV row is the state *after* its first tick, so
# the reproduced cube spawn may have been nudged by one tick of physics; allow
# that much slack when cross-checking the seed against the CSV.
SPAWN_XY_TOL = 0.005


def compose_cfg():
    """Full Hydra config with env=lift and the default dr group (full)."""
    with initialize_config_dir(config_dir=str(REPO_ROOT / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=["env=lift"])
    return cfg


def load_rollout(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    return {
        "actions": df[[f"action_{n}" for n in JOINT_NAMES]].to_numpy(),
        "qpos": df[[f"qpos_{n}" for n in JOINT_NAMES]].to_numpy(),
        "ee": df[["ee_x", "ee_y", "ee_z"]].to_numpy(),
        "cube": df[["cube_x", "cube_y", "cube_z"]].to_numpy(),
        "marker_age_s": df["marker_age_ms"].to_numpy() * 1e-3,
    }


def reproduce_spawn(seed: int, env_cfg: dict, model: mujoco.MjModel,
                    csv_cube0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replicate real/rollout_lift.py's cube spawn rng sequence for --seed and
    cross-check it against the CSV's first cube position."""
    cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    rng = np.random.default_rng(seed)
    cube_xy = rng.uniform(np.array(env_cfg["cube_low"], dtype=np.float64),
                          np.array(env_cfg["cube_high"], dtype=np.float64))
    cube_quat, rest_half_z = sample_cube_orientation(rng, model.geom_size[cube_geom_id])
    cube_pos = np.array([cube_xy[0], cube_xy[1], rest_half_z])
    err = np.linalg.norm(cube_pos[:2] - csv_cube0[:2])
    assert err < SPAWN_XY_TOL, (
        f"seed {seed} reproduces cube spawn {cube_pos[:2]} but the CSV starts at "
        f"{csv_cube0[:2]} ({err * 1e3:.1f} mm apart) — wrong --seed for this CSV?")
    return cube_pos, cube_quat


def run_replay(env: SO101LiftEnv, rec: dict, spawn: tuple) -> dict:
    """Open-loop: recorded actions through sim dynamics from the recorded start."""
    cube_pos, cube_quat = spawn
    env.reset(seed=0, options={"qpos": rec["qpos"][0],
                               "cube_pos": cube_pos, "cube_quat": cube_quat})
    n = len(rec["actions"])
    out = {"qpos": np.empty((n, 6)), "ee": np.empty((n, 3)),
           "cube_z": np.empty(n), "floor_force": np.empty(n)}
    out["qpos"][0] = rec["qpos"][0]
    out["ee"][0] = env._get_ee_pos()
    out["cube_z"][0] = env._get_cube_pos()[2]
    out["floor_force"][0] = env._arm_floor_contact_force()
    # Row k holds the state measured after executing action k, so replay
    # actions 1..n-1 and compare state k to CSV row k.
    for k in range(1, n):
        env.step(rec["actions"][k])
        out["qpos"][k] = env._get_joint_pos()
        out["ee"][k] = env._get_ee_pos()
        out["cube_z"][k] = env._get_cube_pos()[2]
        out["floor_force"][k] = env._arm_floor_contact_force()
    return out


def run_policy(env: SO101LiftEnv, policy, rec: dict, spawn: tuple,
               max_steps: int) -> dict:
    """Closed-loop: the checkpoint in sim from the recorded start state."""
    cube_pos, cube_quat = spawn
    obs, _ = env.reset(seed=0, options={"qpos": rec["qpos"][0],
                                        "cube_pos": cube_pos, "cube_quat": cube_quat})
    out = {"qpos": [rec["qpos"][0]], "ee": [env._get_ee_pos()],
           "cube_z": [env._get_cube_pos()[2]],
           "floor_force": [env._arm_floor_contact_force()], "actions": [np.zeros(6)]}
    for _ in range(1, max_steps):
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        out["actions"].append(np.clip(action, -1.0, 1.0))
        out["qpos"].append(env._get_joint_pos())
        out["ee"].append(env._get_ee_pos())
        out["cube_z"].append(env._get_cube_pos()[2])
        out["floor_force"].append(env._arm_floor_contact_force())
        if terminated or truncated:
            break
    return {k: np.asarray(v) for k, v in out.items()}


def run_obsdiff(model: mujoco.MjModel, policy, rec: dict, control_dt: float) -> dict:
    """Counterfactual: rebuild each step's obs from the recorded qpos with
    FK-consistent markers (hold-last visibility like rollout_lift's fk branch)
    and re-predict. Starts at step 2 — the first two steps' qvel needs the
    pre-rollout boot qpos the CSV doesn't record."""
    data = mujoco.MjData(model)
    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
                for n in MARKER_SITE_NAMES]
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                 for n in JOINT_NAMES]
    qposadr = model.jnt_qposadr[joint_ids]
    cam_pos = tag_cam_world_pos(model, data)

    n = len(rec["actions"])
    held_pos = np.zeros((N_MARKERS, 3))
    held_age = np.full(N_MARKERS, MARKER_AGE_CAP_S)
    seen = np.zeros(N_MARKERS, dtype=bool)
    pred = np.full((n, 6), np.nan)
    fk_markers = np.full((n, N_MARKERS, 3), np.nan)
    for k in range(1, n):
        # Obs at step k is built from the state after tick k-1.
        q = rec["qpos"][k - 1]
        data.qpos[qposadr] = q
        mujoco.mj_kinematics(model, data)
        pos_now, _ = marker_world_poses(data, site_ids)
        vis = markers_visible(data, site_ids, cam_pos)
        held_pos[vis] = pos_now[vis]
        held_age[vis] = rec["marker_age_s"][k]
        held_age[~vis] = np.minimum(MARKER_AGE_CAP_S, held_age[~vis] + control_dt)
        seen |= vis
        held_pos[~seen] = 0.0
        held_age[~seen] = MARKER_AGE_CAP_S
        fk_markers[k] = held_pos
        if k < 2:
            continue
        qvel = (rec["qpos"][k - 1] - rec["qpos"][k - 2]) / control_dt
        obs = np.concatenate([
            q, qvel, held_pos.flatten(), held_age,
            rec["cube"][k - 1],
            np.array([0.0, 0.0, 0.0, LIFT_TASK_ID]),
            rec["actions"][k - 1],
        ]).astype(np.float32)
        action, _ = policy.predict(obs, deterministic=True)
        pred[k] = np.clip(action, -1.0, 1.0)
    return {"pred": pred, "fk_markers": fk_markers}


def plot_replay(out_path: Path, rec: dict, rep: dict, pol: dict,
                control_hz: float) -> None:
    n = len(rec["qpos"])
    t = np.arange(n) / control_hz
    t_pol = np.arange(len(pol["qpos"])) / control_hz
    fig, axes = plt.subplots(3, 3, figsize=(16, 10))
    fig.suptitle("Real rollout vs open-loop action replay (sim) vs closed-loop policy (sim)")
    for j, name in enumerate(JOINT_NAMES):
        ax = axes.flat[j]
        ax.plot(t, rec["qpos"][:, j], "C0", label="real (measured)")
        ax.plot(t, rep["qpos"][:, j], "C1", label="sim replay (same actions)")
        ax.plot(t_pol, pol["qpos"][:, j], "C2", ls="--", lw=1.0,
                label="sim policy (same start)")
        ax.set_title(name)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("rad")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8)

    ax = axes.flat[6]
    ax.plot(t, rec["ee"][:, 2], "C0", label="real ee_z (FK of measured qpos)")
    ax.plot(t, rep["ee"][:, 2], "C1", label="sim replay ee_z")
    ax.plot(t_pol, pol["ee"][:, 2], "C2", ls="--", lw=1.0, label="sim policy ee_z")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_title("end-effector height")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("m")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes.flat[7]
    ax.plot(t, rep["floor_force"], "C1", label="sim replay")
    ax.plot(t_pol, pol["floor_force"], "C2", ls="--", lw=1.0, label="sim policy")
    ax.set_title("arm–floor contact force (sim)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("N")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes.flat[8]
    ax.plot(t, rec["cube"][:, 2], "C0", label="real (lockstep sim cube)")
    ax.plot(t, rep["cube_z"], "C1", label="sim replay")
    ax.plot(t_pol, pol["cube_z"], "C2", ls="--", lw=1.0, label="sim policy")
    ax.set_title("cube height")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("m")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_obsdiff(out_path: Path, rec: dict, od: dict, control_hz: float) -> None:
    n = len(rec["actions"])
    t = np.arange(n) / control_hz
    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    fig.suptitle("Recorded action (camera-marker obs) vs counterfactual "
                 "re-prediction (FK-marker obs) — divergence = marker obs "
                 "changed the policy's mind")
    for j, name in enumerate(JOINT_NAMES):
        ax = axes.flat[j]
        ax.plot(t, rec["actions"][:, j], "C0", label="recorded (camera markers)")
        ax.plot(t, od["pred"][:, j], "C3", label="re-predicted (FK markers)")
        ax.set_title(name)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("action")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="rollouts/rollout_lift_*.csv to diagnose")
    p.add_argument("--model", default="latest",
                   help="'latest', 'best', or a checkpoint .zip path — must be "
                        "the checkpoint the rollout ran")
    p.add_argument("--seed", type=int, required=True,
                   help="--seed the rollout was recorded with (reproduces the cube spawn)")
    args = p.parse_args()

    csv_path = Path(args.csv)
    rec = load_rollout(csv_path)
    cfg = compose_cfg()
    env_cfg = OmegaConf.to_container(cfg.lift_env, resolve=True)
    prev_actions_n = int(cfg.prev_actions_n)
    marker_include_rot = bool(cfg.marker_include_rot)
    assert prev_actions_n == 1 and not marker_include_rot, (
        "obsdiff's hand-built obs assumes prev_actions_n=1, marker_include_rot=false; "
        "update run_obsdiff for the new layout")
    cam_latency = (OmegaConf.to_container(cfg.cam_latency, resolve=True)
                   if cfg.cam_latency is not None else None)

    policy = load_policy(args.model, LOG_DIR,
                         obs_dim_for(prev_actions_n, marker_include_rot))

    xml_path = str(REPO_ROOT / SO101LiftEnv.XML_PATH)
    # Open-loop replay ignores observations entirely -> fully clean env.
    # Closed-loop keeps the camera latency model (marker ages in the training
    # range) but no noise/bias/dropout: the "clean but latency-matched" sim.
    env_replay = make_env(SO101LiftEnv, env_cfg, xml_path,
                          marker_include_rot=marker_include_rot,
                          prev_actions_n=prev_actions_n)
    env_policy = make_env(SO101LiftEnv, env_cfg, xml_path, cam_latency=cam_latency,
                          marker_include_rot=marker_include_rot,
                          prev_actions_n=prev_actions_n)

    spawn = reproduce_spawn(args.seed, env_cfg, env_replay.model, rec["cube"][0])
    control_dt = env_replay._step_dt
    control_hz = 1.0 / control_dt

    rep = run_replay(env_replay, rec, spawn)
    pol = run_policy(env_policy, policy, rec, spawn, int(env_cfg["max_steps"]))
    od = run_obsdiff(env_replay.model, policy, rec, control_dt)

    qpos_err = np.abs(rep["qpos"] - rec["qpos"])
    act_diff = np.abs(od["pred"] - rec["actions"])
    print(f"rows={len(rec['qpos'])}  spawn cube=({spawn[0][0]:+.3f}, {spawn[0][1]:+.3f}, "
          f"{spawn[0][2]:+.3f})")
    print("open-loop replay |qpos_sim - qpos_real| (rad):")
    for j, name in enumerate(JOINT_NAMES):
        print(f"  {name:14s} mean={qpos_err[:, j].mean():.4f}  max={qpos_err[:, j].max():.4f}")
    print(f"ee_z min: real={rec['ee'][:, 2].min():+.4f}  replay={rep['ee'][:, 2].min():+.4f}  "
          f"policy={pol['ee'][:, 2].min():+.4f} m")
    print(f"floor force max (sim): replay={rep['floor_force'].max():.2f}  "
          f"policy={pol['floor_force'].max():.2f} N")
    print(f"sim-policy episode: {len(pol['qpos'])} steps, "
          f"final cube_z={pol['cube_z'][-1]:+.4f} m")
    print("counterfactual action |repredicted - recorded| (obsdiff, steps>=2):")
    for j, name in enumerate(JOINT_NAMES):
        v = act_diff[~np.isnan(act_diff[:, j]), j]
        print(f"  {name:14s} mean={v.mean():.3f}  p95={np.percentile(v, 95):.3f}")

    stem = csv_path.stem
    replay_png = csv_path.parent / f"replay_{stem}.png"
    obsdiff_png = csv_path.parent / f"obsdiff_{stem}.png"
    plot_replay(replay_png, rec, rep, pol, control_hz)
    plot_obsdiff(obsdiff_png, rec, od, control_hz)
    print(f"saved {replay_png} {obsdiff_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
