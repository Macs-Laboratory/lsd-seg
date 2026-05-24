from __future__ import annotations

import torch
from torch import nn

from ..utils.decorators import logged_call, validate_tensor_output


class PrototypeRouter(nn.Module):
    def __init__(self, tau: float = 0.07, learnable_tau: bool = False, min_tau: float = 1e-3) -> None:
        super().__init__()
        self.learnable_tau = learnable_tau
        self.min_tau = min_tau
        if learnable_tau:
            self.log_tau = nn.Parameter(torch.log(torch.tensor(float(tau))))
        else:
            self.register_buffer("tau_buffer", torch.tensor(float(tau)))

    def tau_value(self) -> torch.Tensor:
        if self.learnable_tau:
            return self.log_tau.exp().clamp_min(self.min_tau)
        return self.tau_buffer.clamp_min(self.min_tau)

    @logged_call()
    @validate_tensor_output
    def forward(self, descriptor: torch.Tensor, prototypes: torch.Tensor) -> dict[str, torch.Tensor]:
        if descriptor.numel() == 0 or prototypes.numel() == 0:
            raise ValueError("Router requires non-empty descriptors and prototypes.")
        similarities = descriptor @ prototypes.t()
        tau = self.tau_value()
        logits = similarities / tau
        weights = torch.softmax(logits, dim=-1)
        return {
            "similarities": similarities,
            "cosine_logits": similarities,
            "logits": logits,
            "weights": weights,
            "tau": tau,
        }


class UncertaintyTemperedRouter(nn.Module):
    def __init__(
        self,
        base_temperature: float = 0.07,
        uncertainty_scale: float = 4.0,
        use_uncertainty_tempering: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.base_temperature = base_temperature
        self.uncertainty_scale = uncertainty_scale
        self.use_uncertainty_tempering = use_uncertainty_tempering
        self.eps = eps

    def _entropy(self, weights: torch.Tensor, dim: int, keepdim: bool) -> torch.Tensor:
        safe = weights.clamp_min(self.eps)
        return -(safe * safe.log()).sum(dim=dim, keepdim=keepdim)

    @logged_call()
    @validate_tensor_output
    def forward(
        self,
        similarities: torch.Tensor,
        expert_probs: torch.Tensor,
        mode: str = "sample",
    ) -> dict[str, torch.Tensor]:
        if mode not in {"sample", "pixel"}:
            raise ValueError(f"Invalid routing mode '{mode}'. Expected 'sample' or 'pixel'.")
        if similarities.dim() != 2:
            raise ValueError(f"Expected similarities [B,K], got {tuple(similarities.shape)}")
        if expert_probs.dim() != 5:
            raise ValueError(f"Expected expert_probs [B,K,1,H,W], got {tuple(expert_probs.shape)}")

        base_logits = similarities / max(self.base_temperature, self.eps)
        base_weights = torch.softmax(base_logits, dim=1)
        mean_prob = expert_probs.mean(dim=1)
        uncertainty_map = ((expert_probs - mean_prob[:, None]) ** 2).mean(dim=1)

        if not self.use_uncertainty_tempering:
            return {
                "weights": base_weights,
                "base_weights": base_weights,
                "temperature": torch.full_like(base_weights[:, :1], self.base_temperature),
                "routing_entropy": self._entropy(base_weights, dim=1, keepdim=False),
                "uncertainty": uncertainty_map,
                "mean_expert_prob": mean_prob,
            }

        if mode == "sample":
            u = uncertainty_map.mean(dim=(1, 2, 3))
            tau_eff = self.base_temperature * (1.0 + self.uncertainty_scale * u)
            weights = torch.softmax(similarities / tau_eff[:, None].clamp_min(self.eps), dim=1)
            routing_entropy = self._entropy(weights, dim=1, keepdim=False)
            temperature = tau_eff[:, None]
        else:
            tau_eff = self.base_temperature * (1.0 + self.uncertainty_scale * uncertainty_map)
            logits_xy = similarities[:, :, None, None] / tau_eff.clamp_min(self.eps)
            weights = torch.softmax(logits_xy, dim=1)
            routing_entropy = self._entropy(weights, dim=1, keepdim=True)
            temperature = tau_eff

        return {
            "weights": weights,
            "base_weights": base_weights,
            "temperature": temperature,
            "routing_entropy": routing_entropy,
            "uncertainty": uncertainty_map,
            "mean_expert_prob": mean_prob,
        }


UncertaintyTemperedMixer = UncertaintyTemperedRouter
