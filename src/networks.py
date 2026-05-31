"""Custom networks with LayerNorm + GELU for SB3 policies.

Provides drop-in policy replacements for PPO and SAC that use
LayerNorm + GELU MLPs with fully separate actor/critic networks.
"""

import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy, ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import Actor, SACPolicy


def build_mlp(
    input_dim: int, hidden_dims: list[int], output_dim: int | None = None,
    input_batchnorm: bool = False,
) -> nn.Sequential:
    """Build MLP with LayerNorm + GELU activation.

    Each hidden layer is Linear → LayerNorm → GELU.
    If input_batchnorm is True, prepends BatchNorm1d on the raw input to
    normalize observation scales (running stats are saved with the model).
    If output_dim is given, appends a bare Linear projection (no norm/activation).
    """
    layers: list[nn.Module] = []
    if input_batchnorm:
        layers.append(nn.BatchNorm1d(input_dim))
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
        input_batchnorm: bool = False,
    ):
        super().__init__()
        if isinstance(net_arch, dict):
            pi_arch = net_arch["pi"]
            vf_arch = net_arch["vf"]
        else:
            pi_arch = list(net_arch)
            vf_arch = list(net_arch)

        self.policy_net = build_mlp(feature_dim, pi_arch, input_batchnorm=input_batchnorm)
        self.value_net = build_mlp(feature_dim, vf_arch, input_batchnorm=input_batchnorm)
        self.latent_dim_pi = pi_arch[-1]
        self.latent_dim_vf = vf_arch[-1]

    def forward(self, features):
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)


class LayerNormActorCriticPolicy(ActorCriticPolicy):
    """ActorCriticPolicy (PPO/A2C) using LayerNorm + GELU MLPs."""

    def __init__(self, *args, input_batchnorm: bool = False, **kwargs):
        self._input_batchnorm = input_batchnorm
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = LayerNormMlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
            input_batchnorm=self._input_batchnorm,
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
        input_batchnorm: bool = False,
        **kwargs,
    ):
        super().__init__(
            observation_space, action_space, net_arch,
            features_extractor, features_dim, **kwargs,
        )
        self.latent_pi = build_mlp(features_dim, net_arch, input_batchnorm=input_batchnorm)


class LayerNormContinuousCritic(ContinuousCritic):
    """SAC/TD3 Critic using LayerNorm + GELU MLP."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        input_batchnorm: bool = False,
        **kwargs,
    ):
        super().__init__(
            observation_space, action_space, net_arch,
            features_extractor, features_dim, **kwargs,
        )
        action_dim = get_action_dim(action_space)
        self.q_networks = []
        for idx in range(self.n_critics):
            q_net = build_mlp(features_dim + action_dim, net_arch, output_dim=1, input_batchnorm=input_batchnorm)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)


class LayerNormSACPolicy(SACPolicy):
    """SACPolicy using LayerNorm + GELU MLPs for both actor and critic."""

    def __init__(self, *args, input_batchnorm: bool = False, **kwargs):
        self._input_batchnorm = input_batchnorm
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor: BaseFeaturesExtractor | None = None) -> LayerNormActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        actor_kwargs["input_batchnorm"] = self._input_batchnorm
        return LayerNormActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: BaseFeaturesExtractor | None = None) -> LayerNormContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        critic_kwargs["input_batchnorm"] = self._input_batchnorm
        return LayerNormContinuousCritic(**critic_kwargs).to(self.device)
