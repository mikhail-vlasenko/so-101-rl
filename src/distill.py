"""DAgger distillation: migrate a trained teacher onto a fresh student across an
architecture or observation-layout change, without a curriculum restart.

Every net-width, history-feature, or obs-layout change invalidates checkpoints,
and the only same-obs recovery is `widen_checkpoint` (function-preserving, one
case). Distillation turns the general "retrain from scratch" into a supervised
problem: regress a fresh student onto the teacher's deterministic action *and*
value on the states an (annealed teacher->student) rollout visits under the
student's own deployment DR. DAgger — not plain behavior cloning — because the
covariate shift that BC suffers is worst exactly in the states the still-bad
student visits, which is where it needs the teacher's supervision.

Two regimes, selected by `distill.teacher_obs` (conf/config.yaml):

  identical  — the teacher sees the student's own observation (architecture-only
               change, e.g. a wider net or, later, a history-augmented net whose
               extra inputs are useless for predicting a memoryless teacher and
               get trained toward zero weight). Requires matching obs spaces.
  privileged — the teacher sees the always-fresh-tag privileged view
               (SO101BaseEnv.privileged_obs: what a marker_always_visible=true
               policy sees) while the student trains on the camera obs with
               dropout and hold-last-pose. Here imitation *actively teaches* the
               student to infer the hidden tag pose from its own obs.

The value function is distilled alongside the actor: a warm actor paired with a
fresh critic is the classic PPO-resume trap — the noisy critic yields garbage
advantages that wreck the distilled actor in the first fine-tune updates. The
Gaussian log_std is state-independent, so it is copied, not regressed.

Output is a `resume`-compatible checkpoint. Distillation gets competence;
*always* follow it with a short PPO fine-tune (`python -m src.train resume=...`)
to adapt it and close the residual gap in states where the student's obs makes
the teacher's action only partially inferable.

Usage:
    python -m src.distill env=lift distill.teacher=logs/ppo_lift/best_model.zip
    python -m src.distill env=lift distill.teacher=old.zip distill.net_arch=[512,512,512,512]
    python -m src.distill env=lift distill.teacher=priv.zip distill.teacher_obs=privileged
then fine-tune:
    python -m src.train env=lift resume=distilled.zip
"""

import os

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

from src.train import (
    ALGORITHM_REGISTRY, actor_obs_dim_for, build_fresh_model, env_specs,
    obs_norm_for, runtime_cfg_from_hydra,
)


class DistillBuffer:
    """Aggregated DAgger dataset: a fixed-capacity ring of
    (student_obs, teacher_action, teacher_value) transitions. Old rounds are
    kept (that is the "aggregate" in DAgger) until the FIFO cap is hit; the
    budget is generous because a few million transitions is cheap next to the
    RL steps distillation replaces."""

    def __init__(self, cap: int, obs_dim: int, act_dim: int):
        self.cap = cap
        self.obs = np.zeros((cap, obs_dim), dtype=np.float32)
        self.act = np.zeros((cap, act_dim), dtype=np.float32)
        self.val = np.zeros((cap, 1), dtype=np.float32)
        self._pos = 0
        self._added = 0

    def add(self, obs, act, val):
        n = len(obs)
        idx = (self._pos + np.arange(n)) % self.cap
        self.obs[idx] = obs
        self.act[idx] = act
        self.val[idx] = val
        self._pos = (self._pos + n) % self.cap
        self._added += n

    def __len__(self):
        return min(self._added, self.cap)

    def batches(self, batch_size, rng):
        order = rng.permutation(len(self))
        for start in range(0, len(self), batch_size):
            b = order[start:start + batch_size]
            yield self.obs[b], self.act[b], self.val[b]


def _policy_outputs(policy, obs_np, device):
    """(deterministic mean action, value estimate) for a batch of raw obs, no
    grad. The Gaussian mean is the teacher's deterministic action (widen-style),
    unclipped — the env clips at execution and the student regresses the raw
    target."""
    obs_t = torch.as_tensor(np.asarray(obs_np, dtype=np.float32), device=device)
    with torch.no_grad():
        mean = policy.get_distribution(obs_t).distribution.mean
        value = policy.predict_values(obs_t)
    return mean.cpu().numpy(), value.cpu().numpy()


def _teacher_obs(venv, student_obs, teacher_obs_mode):
    """The observation to query the teacher on at the current (pre-step) state:
    the student's own obs (identical) or the always-fresh privileged view pulled
    from each subprocess env (privileged). Aligned with student_obs because both
    read the env before the upcoming step."""
    if teacher_obs_mode == "identical":
        return student_obs
    return np.stack(venv.env_method("privileged_obs")).astype(np.float32)


def collect(venv, teacher, student, buf, n_steps, beta, teacher_obs_mode,
            device, rng):
    """One DAgger round: roll the env for n_steps ticks under a per-env
    Bernoulli(beta) teacher/student action mix, labeling each visited state with
    the teacher's action+value on its teacher-obs. Stores the *student's* obs as
    the regression input."""
    n_envs = venv.num_envs
    n_loops = int(np.ceil(n_steps / n_envs))
    obs = venv.reset()
    for _ in range(n_loops):
        t_obs = _teacher_obs(venv, obs, teacher_obs_mode)
        t_mean, t_value = _policy_outputs(teacher.policy, t_obs, device)
        buf.add(obs.astype(np.float32), t_mean, t_value)

        t_action = np.clip(t_mean, -1.0, 1.0)
        if beta >= 1.0:
            action = t_action
        else:
            s_action, _ = student.predict(obs, deterministic=True)
            use_teacher = rng.random(n_envs) < beta
            action = np.where(use_teacher[:, None], t_action, s_action)
        obs, _, _, _ = venv.step(action)


def regress(student, buf, epochs, batch_size, vf_coef, optimizer, device, rng):
    """Fit the student to the aggregated dataset: MSE on the teacher's mean
    action plus vf_coef * MSE on the teacher's value. Returns (action_mse,
    value_mse) averaged over the last epoch's batches."""
    student.policy.set_training_mode(True)
    action_mse = value_mse = 0.0
    n_batches = 0
    for epoch in range(epochs):
        action_mse = value_mse = 0.0
        n_batches = 0
        for b_obs, b_act, b_val in buf.batches(batch_size, rng):
            obs_t = torch.as_tensor(b_obs, device=device)
            act_t = torch.as_tensor(b_act, device=device)
            val_t = torch.as_tensor(b_val, device=device)
            dist = student.policy.get_distribution(obs_t)
            pred_mean = dist.distribution.mean
            pred_val = student.policy.predict_values(obs_t)
            loss_a = F.mse_loss(pred_mean, act_t)
            loss_v = F.mse_loss(pred_val, val_t)
            loss = loss_a + vf_coef * loss_v
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            action_mse += loss_a.item()
            value_mse += loss_v.item()
            n_batches += 1
    return action_mse / n_batches, value_mse / n_batches


def eval_policy(eval_venv, policy, n_episodes, is_teacher, teacher_obs_mode,
                seed):
    """Roll out n_episodes deterministically on a single-env vec env and return
    (mean_return, success_rate). The teacher is fed its teacher-obs so both
    policies are scored on the same task under the same DR — a distillation bug
    shows up here as "slightly worse everywhere" long before it crashes."""
    eval_venv.seed(seed)
    obs = eval_venv.reset()
    returns, successes = [], []
    ep_return = 0.0
    while len(returns) < n_episodes:
        if is_teacher and teacher_obs_mode == "privileged":
            pol_obs = np.stack(eval_venv.env_method("privileged_obs")).astype(np.float32)
        else:
            pol_obs = obs
        action, _ = policy.predict(pol_obs, deterministic=True)
        obs, reward, dones, infos = eval_venv.step(action)
        ep_return += float(reward[0])
        if dones[0]:
            returns.append(ep_return)
            ep_return = 0.0
            info = infos[0]
            if "lift_success" in info:
                successes.append(float(info["lift_success"]))
            elif "placed" in info:
                successes.append(float(info["placed"]))
    success_rate = float(np.mean(successes)) if successes else float("nan")
    return float(np.mean(returns)), success_rate


def distill(cfg: DictConfig, orig_dir: str):
    os.chdir(orig_dir)

    dcfg = cfg.distill
    assert dcfg.teacher is not None, "distill.teacher (path to teacher .zip) is required"
    teacher_obs_mode = dcfg.teacher_obs
    assert teacher_obs_mode in ("identical", "privileged"), \
        f"distill.teacher_obs must be 'identical' or 'privileged', got {teacher_obs_mode!r}"

    frame_stack = int(cfg.frame_stack)
    # Privileged obs is a single-frame view pulled live from each subprocess env;
    # feeding a frame-stacked teacher would need a stacked history of it, which
    # only a history-augmented student needs — build that alongside the history
    # wrapper, not here. Fail loud rather than silently mis-stack.
    assert not (teacher_obs_mode == "privileged" and frame_stack > 1), \
        "privileged distillation supports frame_stack=1 only (see distill.py)"

    algo_cls, _, _ = ALGORITHM_REGISTRY[cfg.algorithm]
    seed = int(cfg.seed) if cfg.seed is not None else None
    if seed is not None:
        set_random_seed(seed)
    rng = np.random.default_rng(seed)

    n_envs = cfg.train.n_envs
    runtime_cfg = runtime_cfg_from_hydra(cfg)
    env_fns, eval_inner, n_substeps = env_specs(cfg, orig_dir, runtime_cfg, n_envs)

    vec_env = SubprocVecEnv(env_fns)
    if frame_stack > 1:
        vec_env = VecFrameStack(vec_env, n_stack=frame_stack)
    if seed is not None:
        vec_env.seed(seed)

    obs_norm = obs_norm_for(cfg, n_substeps)
    net_arch = list(dcfg.net_arch) if dcfg.net_arch is not None else list(cfg.train.net_arch)
    student = build_fresh_model(cfg, vec_env, obs_norm, net_arch, seed,
                                actor_obs_dim=actor_obs_dim_for(cfg),
                                tensorboard_log=None, verbose=0)
    device = student.device

    teacher_path = dcfg.teacher
    if not os.path.isabs(teacher_path):
        teacher_path = os.path.join(orig_dir, teacher_path)
    teacher = algo_cls.load(teacher_path, device=device)
    teacher.policy.set_training_mode(False)

    student_obs_dim = student.observation_space.shape[0]
    teacher_obs_dim = teacher.observation_space.shape[0]
    if teacher_obs_mode == "identical":
        assert teacher_obs_dim == student_obs_dim, (
            f"identical distillation needs matching obs dims, "
            f"teacher {teacher_obs_dim} != student {student_obs_dim}")
    else:
        # Privileged teacher shares the single-frame marker layout with the
        # (frame_stack=1) student — same dims, only always-fresh vs held tags.
        assert teacher_obs_dim == student_obs_dim, (
            f"privileged distillation needs the same obs layout, "
            f"teacher {teacher_obs_dim} != student {student_obs_dim}")
    assert teacher.action_space.shape == student.action_space.shape, (
        teacher.action_space, student.action_space)

    # State-independent Gaussian spread: copy, don't regress (an MSE on the mean
    # carries no gradient to it, and a fresh log_std would mis-scale fine-tune
    # exploration).
    student.policy.log_std.data.copy_(teacher.policy.log_std.data)

    # Adam over every student parameter except the copied log_std.
    params = [p for name, p in student.policy.named_parameters() if name != "log_std"]
    optimizer = torch.optim.Adam(params, lr=float(dcfg.lr))

    buf = DistillBuffer(int(dcfg.buffer_cap), student_obs_dim,
                        student.action_space.shape[0])

    eval_venv = DummyVecEnv([lambda: eval_inner])
    if frame_stack > 1:
        eval_venv = VecFrameStack(eval_venv, n_stack=frame_stack)
    eval_seed = (seed + 10_000) if seed is not None else 12_345
    n_eval = int(dcfg.eval_episodes)

    run = None
    if cfg.wandb.enabled:
        import wandb
        run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity,
                         config=OmegaConf.to_container(cfg, resolve=True),
                         job_type="distill")

    t_return, t_success = eval_policy(eval_venv, teacher, n_eval, True,
                                      teacher_obs_mode, eval_seed)
    print(f"teacher eval: return={t_return:.2f} success={t_success:.3f}")

    iterations = int(dcfg.iterations)
    for it in range(iterations):
        beta = (float(dcfg.beta_start) if iterations == 1 else
                float(dcfg.beta_start) + (float(dcfg.beta_end) - float(dcfg.beta_start))
                * it / (iterations - 1))
        collect(vec_env, teacher, student, buf, int(dcfg.steps_per_iter), beta,
                teacher_obs_mode, device, rng)
        action_mse, value_mse = regress(student, buf, int(dcfg.epochs),
                                        int(dcfg.batch_size), float(dcfg.vf_coef),
                                        optimizer, device, rng)
        s_return, s_success = eval_policy(eval_venv, student, n_eval, False,
                                          teacher_obs_mode, eval_seed)
        print(f"iter {it + 1}/{iterations}  beta={beta:.2f}  "
              f"buffer={len(buf)}  action_mse={action_mse:.4f}  "
              f"value_mse={value_mse:.4f}  student: return={s_return:.2f} "
              f"success={s_success:.3f}")
        if run is not None:
            run.log({"distill/iter": it + 1, "distill/beta": beta,
                     "distill/buffer": len(buf), "distill/action_mse": action_mse,
                     "distill/value_mse": value_mse, "distill/student_return": s_return,
                     "distill/student_success": s_success,
                     "distill/teacher_return": t_return,
                     "distill/teacher_success": t_success})

    out_path = dcfg.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(orig_dir, out_path)
    student.save(out_path)
    print(f"saved distilled student -> {out_path}\n"
          f"fine-tune with:  python -m src.train env={cfg.env_name} resume={dcfg.out}")

    if run is not None:
        run.finish()
    vec_env.close()
    eval_venv.close()


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    distill(cfg, hydra.utils.get_original_cwd())


if __name__ == "__main__":
    main()
