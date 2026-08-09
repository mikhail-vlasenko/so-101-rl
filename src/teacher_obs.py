"""Observation adapters for evaluating and distilling migration teachers.

Modern BPS checkpoints consume the full asymmetric-critic observation.  The
last pre-BPS lift policy family was symmetric instead: its checkpoint space is
only the 37-value tag actor block.  Both :mod:`src.eval` and
:mod:`src.distill` must query these teachers through the same exact adapter so
an evaluation cannot accidentally validate a different input than training
uses for supervision.
"""

import numpy as np

from src.bps import BPS_OBS_DIM


TEACHER_OBS_MODES = ("identical", "current", "privileged", "legacy_tag")


def validate_teacher_obs_dim(teacher_obs_mode: str, teacher_obs_dim: int,
                             student_obs_dim: int, single_actor_dim: int,
                             priv_dim: int, legacy_actor_dim: int) -> None:
    """Fail loudly unless a checkpoint matches its requested teacher view."""
    assert teacher_obs_mode in TEACHER_OBS_MODES, (
        f"teacher_obs must be one of {TEACHER_OBS_MODES}, "
        f"got {teacher_obs_mode!r}")
    if teacher_obs_mode == "identical":
        assert teacher_obs_dim == student_obs_dim, (
            "identical teacher view needs matching obs dims, "
            f"teacher {teacher_obs_dim} != student {student_obs_dim}")
    elif teacher_obs_mode == "legacy_tag":
        valid_dims = (legacy_actor_dim, legacy_actor_dim + priv_dim)
        assert teacher_obs_dim in valid_dims, (
            "legacy_tag teacher view needs either the symmetric actor-only "
            "tag layout or its later asymmetric-critic form: "
            f"teacher {teacher_obs_dim} not in {valid_dims}")
    else:
        assert teacher_obs_dim == single_actor_dim + priv_dim, (
            f"{teacher_obs_mode} teacher view needs a single-frame teacher "
            f"BPS checkpoint: teacher {teacher_obs_dim} != "
            f"{single_actor_dim} + {priv_dim}")


def teacher_observation(venv, student_obs, teacher_obs_mode: str,
                        state_dim: int, priv_dim: int, legacy_actor_dim: int,
                        teacher_obs_dim: int):
    """Build the checkpoint-shaped view of the current environment state."""
    if teacher_obs_mode == "identical":
        return student_obs
    if teacher_obs_mode == "current":
        return np.concatenate(
            [student_obs[:, :state_dim],
             student_obs[:, student_obs.shape[1] - priv_dim - BPS_OBS_DIM:
                         student_obs.shape[1] - priv_dim],
             student_obs[:, student_obs.shape[1] - priv_dim:]], axis=1)

    method = "legacy_tag_obs" if teacher_obs_mode == "legacy_tag" \
        else "privileged_obs"
    obs = np.stack(venv.env_method(method)).astype(np.float32)
    if teacher_obs_mode == "legacy_tag" and teacher_obs_dim == legacy_actor_dim:
        return obs[:, :legacy_actor_dim]
    return obs
