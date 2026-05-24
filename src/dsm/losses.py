from __future__ import annotations

import torch
from torch import nn

from .utils.decorators import logged_call, validate_tensor_output


def soft_dice_loss(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    intersection = (prob * target).sum(dim=(1, 2, 3))
    denominator = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


class LSDSegLoss(nn.Module):
    def __init__(
        self,
        expert_weight: float = 0.2,
        ortho_weight: float = 0.01,
        balance_weight: float = 0.01,
        proto_weight: float = 0.0,
        prompt_weight: float = 0.1,
        lambda_expert: float | None = None,
        lambda_ortho: float | None = None,
        lambda_balance: float | None = None,
    ) -> None:
        super().__init__()
        self.expert_weight = lambda_expert if lambda_expert is not None else expert_weight
        self.ortho_weight = lambda_ortho if lambda_ortho is not None else ortho_weight
        self.balance_weight = lambda_balance if lambda_balance is not None else balance_weight
        self.proto_weight = proto_weight
        self.prompt_weight = prompt_weight
        self.bce = nn.BCEWithLogitsLoss()

    def _seg_components(self, logits: torch.Tensor, prob: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.bce(logits, target), soft_dice_loss(prob, target)

    def _expert_loss(self, expert_logits: torch.Tensor, expert_probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if expert_logits.shape[1] == 0:
            return target.new_tensor(0.0)
        losses = []
        for expert_index in range(expert_logits.shape[1]):
            bce, dice = self._seg_components(expert_logits[:, expert_index], expert_probs[:, expert_index], target)
            losses.append(bce + dice)
        return torch.stack(losses).mean()

    def _prototype_orthogonality(self, prototypes: torch.Tensor) -> torch.Tensor:
        if prototypes.shape[0] <= 1:
            return prototypes.new_tensor(0.0)
        normalized = torch.nn.functional.normalize(prototypes, dim=-1, eps=1e-6)
        sim = normalized @ normalized.t()
        off_diag = sim - torch.eye(sim.shape[0], device=sim.device, dtype=sim.dtype)
        return off_diag.pow(2).mean()

    def _routing_balance(self, routing_weights: torch.Tensor) -> torch.Tensor:
        if routing_weights.shape[1] <= 1:
            return routing_weights.new_tensor(0.0)
        if routing_weights.dim() == 2:
            mean_w = routing_weights.mean(dim=0)
        elif routing_weights.dim() == 4:
            mean_w = routing_weights.mean(dim=(0, 2, 3))
        else:
            raise ValueError(f"Unsupported routing weight shape: {tuple(routing_weights.shape)}")
        mean_w = mean_w.clamp_min(1e-6)
        uniform = torch.full_like(mean_w, 1.0 / mean_w.numel())
        return torch.sum(mean_w * (mean_w / uniform).log())

    @logged_call()
    @validate_tensor_output
    def forward(self, outputs: dict[str, torch.Tensor | int], target: torch.Tensor) -> dict[str, torch.Tensor]:
        final_logits = outputs["logits"]
        final_prob = outputs["probabilities"] if "probabilities" in outputs else outputs["prob"]
        loss_bce, loss_dice = self._seg_components(final_logits, final_prob, target)
        loss_seg = loss_bce + loss_dice

        expert_logits = outputs["expert_logits"]
        expert_probs = outputs["expert_probabilities"] if "expert_probabilities" in outputs else outputs["expert_probs"]
        loss_expert = self._expert_loss(expert_logits, expert_probs, target)

        prototypes = outputs["prototype_embeddings"] if "prototype_embeddings" in outputs else outputs["prototypes"]
        loss_ortho = self._prototype_orthogonality(prototypes)

        routing_weights = outputs["routing_weights"]
        loss_balance = self._routing_balance(routing_weights)

        loss_proto = final_logits.new_tensor(0.0)
        prompt_logits = outputs.get("prompt_logits", outputs.get("prompt_coarse_logits"))
        if isinstance(prompt_logits, torch.Tensor):
            prompt_logits = torch.nn.functional.interpolate(prompt_logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
            prompt_prob = torch.sigmoid(prompt_logits)
            prompt_bce, prompt_dice = self._seg_components(prompt_logits, prompt_prob, target)
            loss_prompt = prompt_bce + prompt_dice
        else:
            loss_prompt = final_logits.new_tensor(0.0)
        total = (
            loss_seg
            + self.expert_weight * loss_expert
            + self.ortho_weight * loss_ortho
            + self.balance_weight * loss_balance
            + self.proto_weight * loss_proto
            + self.prompt_weight * loss_prompt
        )
        return {
            "loss": total,
            "loss_seg": loss_seg,
            "loss_bce": loss_bce,
            "loss_dice": loss_dice,
            "loss_expert": loss_expert,
            "loss_ortho": loss_ortho,
            "loss_balance": loss_balance,
            "loss_proto": loss_proto,
            "loss_prompt": loss_prompt,
        }
