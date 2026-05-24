from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from ..utils.decorators import logged_call, validate_tensor_output


@dataclass(slots=True)
class AdapterParameters:
    gammas: list[torch.Tensor]
    betas: list[torch.Tensor]

    def for_expert(self, expert_idx: int) -> dict[int, dict[str, torch.Tensor]]:
        return {
            stage_idx: {"gamma": gamma[expert_idx], "beta": beta[expert_idx]}
            for stage_idx, (gamma, beta) in enumerate(zip(self.gammas, self.betas))
        }


class PrototypeHyperNetwork(nn.Module):
    """Generate stable FiLM parameters from prototype vectors."""

    def __init__(
        self,
        prototype_dim: int,
        decoder_channels: Sequence[int],
        hidden_dim: int = 256,
        gamma_scale: float = 0.1,
        beta_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.decoder_channels = list(decoder_channels)
        self.gamma_scale = gamma_scale
        self.beta_scale = beta_scale
        self.stage_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(prototype_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, 2 * channel),
                )
                for channel in self.decoder_channels
            ]
        )
        for mlp in self.stage_mlps:
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)

    @logged_call()
    @validate_tensor_output
    def forward(self, prototypes: torch.Tensor) -> AdapterParameters:
        gammas: list[torch.Tensor] = []
        betas: list[torch.Tensor] = []
        for mlp in self.stage_mlps:
            raw = mlp(prototypes)
            raw_gamma, raw_beta = raw.chunk(2, dim=-1)
            gammas.append(1.0 + self.gamma_scale * torch.tanh(raw_gamma))
            betas.append(self.beta_scale * torch.tanh(raw_beta))
        return AdapterParameters(gammas=gammas, betas=betas)


class StaticPrototypeConditioner(nn.Module):
    """Static per-prototype FiLM parameters for no-hypernetwork ablations."""

    def __init__(self, max_prototypes: int, decoder_channels: Sequence[int]) -> None:
        super().__init__()
        self.decoder_channels = list(decoder_channels)
        self.gamma_params = nn.ParameterList(
            [nn.Parameter(torch.ones(max_prototypes, channel)) for channel in self.decoder_channels]
        )
        self.beta_params = nn.ParameterList(
            [nn.Parameter(torch.zeros(max_prototypes, channel)) for channel in self.decoder_channels]
        )

    @logged_call()
    @validate_tensor_output
    def forward(self, prototypes: torch.Tensor) -> AdapterParameters:
        num_active = prototypes.shape[0]
        gammas = [parameter[:num_active] for parameter in self.gamma_params]
        betas = [parameter[:num_active] for parameter in self.beta_params]
        return AdapterParameters(gammas=gammas, betas=betas)


HypernetworkAdapterGenerator = PrototypeHyperNetwork
StaticAdapterGenerator = StaticPrototypeConditioner
