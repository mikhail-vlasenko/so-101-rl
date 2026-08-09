"""Custom networks with LayerNorm + GELU for SB3 policies.

Provides drop-in policy replacements for PPO and SAC that use
LayerNorm + GELU MLPs with fully separate actor/critic networks, fronted by a
fixed affine input normalization (ObsNorm) with physically-derived constants
from src/obs_norm.py.
"""

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy, ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import Actor, SACPolicy


class TakeFirst(nn.Module):
    """Pass through only the first n input dims.

    The asymmetric critic's actor slice: the env obs is [actor block |
    privileged tail] and this module, prepended to the actor MLP, makes it
    structurally impossible for the policy to read the tail. Living in the
    network graph (rather than in calling convention), the constraint rides
    through checkpoints and every predict/rollout path untouched.
    """

    def __init__(self, n: int):
        super().__init__()
        self.n = int(n)

    def forward(self, x):
        return x[..., :self.n]


class ObsNorm(nn.Module):
    """Fixed affine input normalization: (x - center) / scale.

    center/scale are physically-derived constants (src/obs_norm.py) stored as
    non-trainable buffers: no estimated statistics, so the transform is
    identical in train and eval mode, immune to obs-distribution shifts
    (marker dropout rates, curriculum stages), and rides along in the
    checkpoint like any weight.
    """

    def __init__(self, center, scale):
        super().__init__()
        center = torch.as_tensor(np.asarray(center), dtype=torch.float32)
        scale = torch.as_tensor(np.asarray(scale), dtype=torch.float32)
        assert center.shape == scale.shape and center.dim() == 1, \
            (center.shape, scale.shape)
        assert bool((scale > 0).all()), "obs_norm scale must be strictly positive"
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)

    def forward(self, x):
        return (x - self.center) / self.scale


def build_mlp(
    input_dim: int, hidden_dims: list[int], output_dim: int | None = None,
    obs_norm: tuple | None = None,
) -> nn.Sequential:
    """Build MLP with LayerNorm + GELU activation.

    Each hidden layer is Linear → LayerNorm → GELU.
    If obs_norm is given as (center, scale) sequences covering input_dim dims,
    prepends the fixed ObsNorm affine on the raw input.
    If output_dim is given, appends a bare Linear projection (no norm/activation).
    """
    layers: list[nn.Module] = []
    if obs_norm is not None:
        norm = ObsNorm(*obs_norm)
        assert norm.center.shape[0] == input_dim, (norm.center.shape, input_dim)
        layers.append(norm)
    prev = input_dim
    for dim in hidden_dims:
        layers.extend([nn.Linear(prev, dim), nn.LayerNorm(dim), nn.GELU()])
        prev = dim
    if output_dim is not None:
        layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# PPO / A2C — custom MlpExtractor + policy
# ---------------------------------------------------------------------------


class LayerNormMlpExtractor(nn.Module):
    """MLP extractor with LayerNorm + GELU, fully separate actor and critic.

    Same interface as SB3's MlpExtractor so it can be swapped in.
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: list[int] | dict[str, list[int]],
        activation_fn: type[nn.Module],  # ignored — always LayerNorm + GELU
        device: str = "auto",
        obs_norm: tuple | None = None,
        actor_obs_dim: int | None = None,
    ):
        super().__init__()
        if isinstance(net_arch, dict):
            pi_arch = net_arch["pi"]
            vf_arch = net_arch["vf"]
        else:
            pi_arch = list(net_arch)
            vf_arch = list(net_arch)

        if actor_obs_dim is None:
            # Symmetric: both heads consume the full observation.
            self.policy_net = build_mlp(feature_dim, pi_arch, obs_norm=obs_norm)
        else:
            # Asymmetric critic: the obs is [actor block | privileged tail];
            # the actor slices its block off before its first layer (its
            # ObsNorm covers only those dims), while the value net below
            # consumes the full vector, privileged tail included.
            assert 0 < actor_obs_dim < feature_dim, (actor_obs_dim, feature_dim)
            actor_norm = None
            if obs_norm is not None:
                actor_norm = (np.asarray(obs_norm[0])[:actor_obs_dim],
                              np.asarray(obs_norm[1])[:actor_obs_dim])
            self.policy_net = nn.Sequential(
                TakeFirst(actor_obs_dim),
                *build_mlp(actor_obs_dim, pi_arch, obs_norm=actor_norm))
        self.value_net = build_mlp(feature_dim, vf_arch, obs_norm=obs_norm)
        self.latent_dim_pi = pi_arch[-1]
        self.latent_dim_vf = vf_arch[-1]

    def forward(self, features):
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)


class LayerNormActorCriticPolicy(ActorCriticPolicy):
    """ActorCriticPolicy (PPO/A2C) using LayerNorm + GELU MLPs.

    actor_obs_dim (asymmetric critic): when set, the actor consumes only the
    first actor_obs_dim obs dims (TakeFirst) while the critic sees the full
    vector including the privileged tail. Rides in policy_kwargs, so
    checkpoints restore the slice with the weights."""

    def __init__(self, *args, obs_norm: tuple | None = None,
                 actor_obs_dim: int | None = None,
                 bps_fingerprint: str | None = None, **kwargs):
        self._obs_norm = obs_norm
        self._actor_obs_dim = actor_obs_dim
        # Persisted through policy_kwargs. Runtime loaders validate it before
        # inference or resume; the network itself never interprets the value.
        self.bps_fingerprint = bps_fingerprint
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = LayerNormMlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
            obs_norm=self._obs_norm,
            actor_obs_dim=self._actor_obs_dim,
        )


# ---------------------------------------------------------------------------
# SAC — custom Actor, Critic, and policy
# ---------------------------------------------------------------------------


class LayerNormActor(Actor):
    """SAC Actor using LayerNorm + GELU MLP."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        obs_norm: tuple | None = None,
        **kwargs,
    ):
        super().__init__(
            observation_space, action_space, net_arch,
            features_extractor, features_dim, **kwargs,
        )
        self.latent_pi = build_mlp(features_dim, net_arch, obs_norm=obs_norm)


class LayerNormContinuousCritic(ContinuousCritic):
    """SAC/TD3 Critic using LayerNorm + GELU MLP."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        obs_norm: tuple | None = None,
        **kwargs,
    ):
        super().__init__(
            observation_space, action_space, net_arch,
            features_extractor, features_dim, **kwargs,
        )
        action_dim = get_action_dim(action_space)
        if obs_norm is not None:
            # Critic input is [obs, action]; actions are already in [-1, 1],
            # so extend the obs constants with an identity map for them.
            obs_norm = (np.concatenate([np.asarray(obs_norm[0]), np.zeros(action_dim)]),
                        np.concatenate([np.asarray(obs_norm[1]), np.ones(action_dim)]))
        self.q_networks = []
        for idx in range(self.n_critics):
            q_net = build_mlp(features_dim + action_dim, net_arch, output_dim=1, obs_norm=obs_norm)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)


class LayerNormSACPolicy(SACPolicy):
    """SACPolicy using LayerNorm + GELU MLPs for both actor and critic."""

    def __init__(self, *args, obs_norm: tuple | None = None, **kwargs):
        self._obs_norm = obs_norm
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor: BaseFeaturesExtractor | None = None) -> LayerNormActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        actor_kwargs["obs_norm"] = self._obs_norm
        return LayerNormActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: BaseFeaturesExtractor | None = None) -> LayerNormContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        critic_kwargs["obs_norm"] = self._obs_norm
        return LayerNormContinuousCritic(**critic_kwargs).to(self.device)
