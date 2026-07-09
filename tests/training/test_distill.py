"""Contract for src.distill — the DAgger distillation rig.

Distillation bugs don't crash; they look like "slightly worse everywhere". So
these tests pin the pieces that must be exactly right: the aggregation buffer,
that regression actually drives the student toward the teacher's function, and
that a distilled checkpoint round-trips through the existing `resume` machinery
(the mandatory fine-tune step) with the student's own architecture.
"""

import numpy as np
import pytest
import torch
from hydra import compose, initialize
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.base_env import obs_dim_for, priv_dim_for
from src.distill import DistillBuffer, _policy_outputs, distill, regress
from src.lift_env import SO101LiftEnv
from src.train import (
    actor_obs_dim_for, build_fresh_model, obs_norm_for, resume_overrides,
    runtime_cfg_from_hydra,
)

REPO_ROOT = "."


@pytest.fixture(scope="module")
def lift_cfg():
    with initialize(config_path="../../conf", version_base=None):
        return compose(config_name="config", overrides=[
            "env=lift", "wandb.enabled=false", "train.n_envs=2",
            "ppo.n_steps=8", "train.net_arch=[64,64]",
        ])


def _lift_venv(cfg):
    return DummyVecEnv([lambda: SO101LiftEnv(
        env_cfg=cfg.lift_env, xml_path="so101/scene_lift.xml",
        cfg=runtime_cfg_from_hydra(cfg))])


# --------------------------------------------------------------------------
# Buffer
# --------------------------------------------------------------------------

def test_buffer_caps_and_overwrites_oldest():
    buf = DistillBuffer(cap=10, obs_dim=3, act_dim=2)
    buf.add(np.zeros((6, 3), np.float32), np.zeros((6, 2), np.float32),
            np.zeros((6, 1), np.float32))
    assert len(buf) == 6
    # Overflow by 6 more (total 12 > cap 10): length pins at cap, oldest evicted.
    buf.add(np.ones((6, 3), np.float32), np.ones((6, 2), np.float32),
            np.ones((6, 1), np.float32))
    assert len(buf) == 10
    # The ring wrote 12 rows into 10 slots: rows 10,11 overwrote slots 0,1.
    assert buf.obs[0, 0] == 1.0 and buf.obs[1, 0] == 1.0
    assert buf.obs[2, 0] == 0.0


def test_buffer_batches_cover_all_rows_once():
    buf = DistillBuffer(cap=100, obs_dim=1, act_dim=1)
    ids = np.arange(37, dtype=np.float32).reshape(-1, 1)
    buf.add(ids, ids, ids)
    rng = np.random.default_rng(0)
    seen = np.concatenate([b[0].ravel() for b in buf.batches(8, rng)])
    assert sorted(seen.tolist()) == list(range(37))


# --------------------------------------------------------------------------
# Regression core
# --------------------------------------------------------------------------

def test_regress_drives_student_toward_teacher(lift_cfg):
    """The heart of the rig: fitting the student on the teacher's action+value
    labels must sharply cut the action MSE between them on the label states."""
    venv = _lift_venv(lift_cfg)
    obs_norm = obs_norm_for(lift_cfg, n_substeps=10)
    actor_dim = actor_obs_dim_for(lift_cfg)
    student = build_fresh_model(lift_cfg, venv, obs_norm, [64, 64], seed=1,
                                actor_obs_dim=actor_dim, verbose=0)
    teacher = build_fresh_model(lift_cfg, venv, obs_norm, [64, 64], seed=2,
                                actor_obs_dim=actor_dim, verbose=0)
    teacher.policy.set_training_mode(False)
    device = student.device

    obs_dim = student.observation_space.shape[0]
    act_dim = student.action_space.shape[0]
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((4096, obs_dim)).astype(np.float32)
    t_mean, t_val = _policy_outputs(teacher.policy, obs, device)

    buf = DistillBuffer(8192, obs_dim, act_dim)
    buf.add(obs, t_mean, t_val)

    s_mean0, _ = _policy_outputs(student.policy, obs, device)
    err0 = float(np.mean((s_mean0 - t_mean) ** 2))

    student.policy.log_std.data.copy_(teacher.policy.log_std.data)
    params = [p for n, p in student.policy.named_parameters() if n != "log_std"]
    opt = torch.optim.Adam(params, lr=1e-3)
    for _ in range(40):
        regress(student, buf, epochs=1, batch_size=512, vf_coef=1.0,
                optimizer=opt, device=device, rng=rng)

    s_mean1, _ = _policy_outputs(student.policy, obs, device)
    err1 = float(np.mean((s_mean1 - t_mean) ** 2))
    assert err1 < 0.1 * err0, f"action MSE did not converge: {err0:.4f} -> {err1:.4f}"
    venv.close()


def test_regress_leaves_log_std_untouched(lift_cfg):
    """log_std is copied, not regressed — the optimizer must never move it."""
    venv = _lift_venv(lift_cfg)
    obs_norm = obs_norm_for(lift_cfg, n_substeps=10)
    student = build_fresh_model(lift_cfg, venv, obs_norm, [64, 64], seed=1,
                                actor_obs_dim=actor_obs_dim_for(lift_cfg), verbose=0)
    device = student.device
    obs_dim = student.observation_space.shape[0]
    act_dim = student.action_space.shape[0]

    sentinel = torch.full_like(student.policy.log_std.data, -0.5)
    student.policy.log_std.data.copy_(sentinel)
    params = [p for n, p in student.policy.named_parameters() if n != "log_std"]
    opt = torch.optim.Adam(params, lr=1e-2)

    rng = np.random.default_rng(1)
    obs = rng.standard_normal((1024, obs_dim)).astype(np.float32)
    buf = DistillBuffer(2048, obs_dim, act_dim)
    buf.add(obs, rng.standard_normal((1024, act_dim)).astype(np.float32),
            rng.standard_normal((1024, 1)).astype(np.float32))
    for _ in range(5):
        regress(student, buf, epochs=1, batch_size=256, vf_coef=1.0,
                optimizer=opt, device=device, rng=rng)
    assert torch.equal(student.policy.log_std.data, sentinel)
    venv.close()


# --------------------------------------------------------------------------
# End-to-end plumbing + resume compatibility
# --------------------------------------------------------------------------

def test_distill_end_to_end_and_resumes(lift_cfg, tmp_path):
    """One full DAgger round through the SubprocVecEnv farm (identical mode):
    the student is built with its own net_arch, trained on teacher labels, and
    saved to a checkpoint that `resume` loads and fine-tunes."""
    obs_norm = obs_norm_for(lift_cfg, n_substeps=10)
    venv = _lift_venv(lift_cfg)
    teacher = build_fresh_model(lift_cfg, venv, obs_norm, [64, 64], seed=7,
                                actor_obs_dim=actor_obs_dim_for(lift_cfg), verbose=0)
    teacher_path = tmp_path / "teacher.zip"
    teacher.save(teacher_path)
    venv.close()

    out_path = tmp_path / "distilled.zip"
    with initialize(config_path="../../conf", version_base=None):
        cfg = compose(config_name="config", overrides=[
            "env=lift", "wandb.enabled=false", "train.n_envs=2", "ppo.n_steps=8",
            "distill.teacher_obs=identical", "distill.net_arch=[64,64]",
            "distill.iterations=1", "distill.steps_per_iter=32",
            "distill.epochs=1", "distill.batch_size=32", "distill.eval_episodes=1",
            f"distill.teacher={teacher_path}", f"distill.out={out_path}",
        ])
        distill(cfg, orig_dir=".")

    assert out_path.exists(), "distillation did not write the student checkpoint"

    # The saved student carries its OWN architecture, and resume loads + trains it.
    student = PPO.load(out_path)
    assert list(student.policy_kwargs["net_arch"]) == [64, 64]

    resume_venv = _lift_venv(cfg)
    resumed = PPO.load(out_path, env=resume_venv, **resume_overrides(cfg))
    # policy_net = [TakeFirst, ObsNorm, Linear, ...] — index 2 is the first Linear.
    weight_key = "mlp_extractor.policy_net.2.weight"
    before = resumed.policy.state_dict()[weight_key].clone()
    resumed.learn(total_timesteps=16)
    after = resumed.policy.state_dict()[weight_key]
    assert not torch.equal(before, after), "resume fine-tune did not update the student"
    resume_venv.close()


def test_distill_current_mode_migrates_onto_history_taps(tmp_path):
    """teacher_obs=current — the history_taps migration path: a single-frame
    teacher supervises a lag-tapped student, queried on the [tap-0 actor block
    | priv tail] slice of the student's obs. The saved student must carry the
    tapped obs space and the widened actor slice; and feeding a NON-single-
    frame checkpoint as the teacher must fail loud."""
    with initialize(config_path="../../conf", version_base=None):
        teacher_cfg = compose(config_name="config", overrides=[
            "env=lift", "wandb.enabled=false", "train.n_envs=2",
            "ppo.n_steps=8", "train.net_arch=[32,32]",
        ])
    venv = _lift_venv(teacher_cfg)
    teacher = build_fresh_model(teacher_cfg, venv, obs_norm_for(teacher_cfg, 10),
                                [32, 32], seed=3,
                                actor_obs_dim=actor_obs_dim_for(teacher_cfg),
                                verbose=0)
    teacher_path = tmp_path / "teacher.zip"
    teacher.save(teacher_path)
    venv.close()

    out_path = tmp_path / "student.zip"
    with initialize(config_path="../../conf", version_base=None):
        cfg = compose(config_name="config", overrides=[
            "env=lift", "wandb.enabled=false", "train.n_envs=2", "ppo.n_steps=8",
            "history_taps=[0,2]",
            "distill.teacher_obs=current", "distill.net_arch=[32,32]",
            "distill.iterations=1", "distill.steps_per_iter=32",
            "distill.epochs=1", "distill.batch_size=32", "distill.eval_episodes=1",
            f"distill.teacher={teacher_path}", f"distill.out={out_path}",
        ])
        distill(cfg, orig_dir=".")

        student = PPO.load(out_path)
        a = obs_dim_for(int(cfg.prev_actions_n), bool(cfg.marker_include_rot))
        p = priv_dim_for(bool(cfg.marker_include_rot))
        assert student.observation_space.shape == (2 * a + p,)
        assert student.policy_kwargs["actor_obs_dim"] == 2 * a

        # The tapped student itself is not a valid current-mode teacher.
        cfg2 = compose(config_name="config", overrides=[
            "env=lift", "wandb.enabled=false", "train.n_envs=2", "ppo.n_steps=8",
            "history_taps=[0,2]", "distill.teacher_obs=current",
            f"distill.teacher={out_path}", f"distill.out={tmp_path / 'x.zip'}",
        ])
        with pytest.raises(AssertionError, match="single-frame teacher"):
            distill(cfg2, orig_dir=".")
