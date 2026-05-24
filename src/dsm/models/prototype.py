from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..utils.decorators import logged_call, validate_tensor_output


@dataclass(slots=True)
class PrototypeState:
    embeddings: torch.Tensor
    counts: torch.Tensor
    active_mask: torch.Tensor
    last_update: torch.Tensor
    step: torch.Tensor


class PrototypeMemoryBank(nn.Module):
    """Dynamic latent prototype memory stored as registered buffers."""

    def __init__(
        self,
        descriptor_dim: int | None = None,
        dim: int | None = None,
        max_prototypes: int = 16,
        novelty_threshold: float = 0.65,
        init_threshold: float | None = None,
        merge_threshold: float = 0.95,
        ema_momentum: float = 0.95,
        min_support: int = 3,
        merge_interval: int = 100,
        warmup_steps: int = 0,
        dynamic_prototypes: bool = True,
        fixed_k: int | None = None,
        merge_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.descriptor_dim = int(descriptor_dim if descriptor_dim is not None else dim if dim is not None else 256)
        self.max_prototypes = max_prototypes
        self.novelty_threshold = init_threshold if init_threshold is not None else novelty_threshold
        self.merge_threshold = merge_threshold
        self.ema_momentum = ema_momentum
        self.min_support = min_support
        self.merge_interval = merge_interval
        self.warmup_steps = warmup_steps
        self.dynamic_prototypes = dynamic_prototypes
        self.fixed_k = fixed_k
        self.merge_enabled = merge_enabled and fixed_k is None

        self.register_buffer("embeddings", torch.zeros(max_prototypes, self.descriptor_dim))
        self.register_buffer("counts", torch.zeros(max_prototypes, dtype=torch.long))
        self.register_buffer("active_mask", torch.zeros(max_prototypes, dtype=torch.bool))
        self.register_buffer("last_update", torch.full((max_prototypes,), -1, dtype=torch.long))
        self.register_buffer("step", torch.zeros((), dtype=torch.long))

    @property
    def prototype_embeddings(self) -> torch.Tensor:
        return self.embeddings

    @property
    def prototype_counts(self) -> torch.Tensor:
        return self.counts

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return F.normalize(tensor, dim=-1, eps=1e-6)

    def get_active_indices(self) -> torch.Tensor:
        return torch.nonzero(self.active_mask, as_tuple=False).flatten()

    def get_active_state(self) -> PrototypeState:
        return PrototypeState(
            embeddings=self.get_active_embeddings(),
            counts=self.counts[self.active_mask],
            active_mask=self.active_mask.clone(),
            last_update=self.last_update[self.active_mask],
            step=self.step.clone(),
        )

    def get_active_embeddings(self) -> torch.Tensor:
        indices = self.get_active_indices()
        if indices.numel() == 0:
            return self.embeddings.new_zeros((0, self.descriptor_dim))
        return self.embeddings.index_select(0, indices)

    def num_active(self) -> int:
        return int(self.active_mask.sum().item())

    @logged_call()
    @validate_tensor_output
    def assign(self, descriptors: torch.Tensor) -> dict[str, torch.Tensor]:
        if descriptors.dim() != 2 or descriptors.shape[1] != self.descriptor_dim:
            raise ValueError(f"Expected descriptors [B,{self.descriptor_dim}], got {tuple(descriptors.shape)}")
        active_embeddings = self.get_active_embeddings()
        if active_embeddings.numel() == 0:
            raise RuntimeError("No active prototypes found. Run training or load prototype memory first.")
        normalized = self._normalize(descriptors)
        similarities = normalized @ active_embeddings.t()
        max_similarity, assignments = similarities.max(dim=1)
        return {
            "assignments": assignments,
            "assigned_ids": assignments,
            "similarities": similarities,
            "similarity": similarities,
            "max_similarity": max_similarity,
        }

    @torch.no_grad()
    def maybe_create(self, descriptor: torch.Tensor) -> int | None:
        if self.num_active() >= self.max_prototypes:
            return None
        inactive = torch.nonzero(~self.active_mask, as_tuple=False).flatten()
        if inactive.numel() == 0:
            return None
        global_index = int(inactive[0].item())
        self.embeddings[global_index] = self._normalize(descriptor.unsqueeze(0)).squeeze(0)
        self.counts[global_index] = 1
        self.active_mask[global_index] = True
        self.last_update[global_index] = self.step
        return global_index

    @torch.no_grad()
    def update(self, descriptors: torch.Tensor, assigned_ids: torch.Tensor) -> None:
        if descriptors.numel() == 0:
            return
        normalized = self._normalize(descriptors)
        active_indices = self.get_active_indices()
        for local_index, global_index in enumerate(active_indices.tolist()):
            mask = assigned_ids == local_index
            if not mask.any():
                continue
            mean_descriptor = normalized[mask].mean(dim=0)
            updated = self.ema_momentum * self.embeddings[global_index] + (1.0 - self.ema_momentum) * mean_descriptor
            self.embeddings[global_index] = self._normalize(updated.unsqueeze(0)).squeeze(0)
            self.counts[global_index] += int(mask.sum().item())
            self.last_update[global_index] = self.step

    @torch.no_grad()
    def maybe_merge(self) -> None:
        if not self.merge_enabled:
            return
        if self.step.item() < self.warmup_steps:
            return
        if self.merge_interval <= 0 or self.step.item() % self.merge_interval != 0:
            return
        active_indices = self.get_active_indices()
        if active_indices.numel() <= 1:
            return
        active_embeddings = self.get_active_embeddings()
        pairwise = active_embeddings @ active_embeddings.t()
        for i in range(active_indices.numel()):
            for j in range(i + 1, active_indices.numel()):
                if pairwise[i, j].item() <= self.merge_threshold:
                    continue
                gi = int(active_indices[i].item())
                gj = int(active_indices[j].item())
                if not self.active_mask[gi] or not self.active_mask[gj]:
                    continue
                keep, remove = (gi, gj) if self.counts[gi] >= self.counts[gj] else (gj, gi)
                total = self.counts[keep] + self.counts[remove]
                merged = (
                    self.counts[keep].float() * self.embeddings[keep]
                    + self.counts[remove].float() * self.embeddings[remove]
                ) / total.float().clamp_min(1.0)
                self.embeddings[keep] = self._normalize(merged.unsqueeze(0)).squeeze(0)
                self.counts[keep] = total
                self.last_update[keep] = self.step
                self.active_mask[remove] = False
                self.counts[remove] = 0
                self.last_update[remove] = -1

    @logged_call()
    @validate_tensor_output
    def forward(self, descriptors: torch.Tensor, update: bool = True) -> dict[str, torch.Tensor | int]:
        if descriptors.dim() != 2:
            raise ValueError(f"Expected descriptors [B,D], got {tuple(descriptors.shape)}")
        descriptors = self._normalize(descriptors)
        should_update = bool(update and self.training)

        if should_update:
            with torch.no_grad():
                self.step += 1
                seed_start = 0
                if self.num_active() == 0:
                    self.maybe_create(descriptors[0])
                    seed_start = 1

                if self.fixed_k is not None:
                    for descriptor in descriptors[seed_start:]:
                        if self.num_active() >= min(self.fixed_k, self.max_prototypes):
                            break
                        self.maybe_create(descriptor)
                else:
                    for descriptor in descriptors:
                        assigned = self.assign(descriptor.unsqueeze(0))
                        if self.dynamic_prototypes and assigned["max_similarity"].item() < self.novelty_threshold:
                            self.maybe_create(descriptor)

                assigned = self.assign(descriptors)
                self.update(descriptors, assigned["assignments"])
                self.maybe_merge()
        else:
            if self.num_active() == 0:
                raise RuntimeError("No active prototypes found. Run training or load prototype memory first.")
            assigned = self.assign(descriptors)

        active_embeddings = self.get_active_embeddings()
        return {
            "embeddings": active_embeddings,
            "prototypes": active_embeddings,
            "assignments": assigned["assignments"],
            "assigned_ids": assigned["assignments"],
            "similarities": assigned["similarities"],
            "similarity": assigned["similarities"],
            "max_similarity": assigned["max_similarity"],
            "num_active": self.num_active(),
            "counts": self.counts[self.active_mask],
        }

    @torch.no_grad()
    def summary(self) -> dict[str, torch.Tensor | list[float] | int]:
        active = self.get_active_embeddings()
        if active.numel() == 0:
            return {"num_active": 0, "support_counts": [], "prototype_norms": [], "pairwise_similarity": []}
        return {
            "num_active": self.num_active(),
            "support_counts": self.counts[self.active_mask].tolist(),
            "prototype_norms": active.norm(dim=1).tolist(),
            "pairwise_similarity": (active @ active.t()).cpu(),
        }

    def prototype_summary(self) -> dict[str, torch.Tensor | list[float] | int]:
        return self.summary()


DynamicPrototypeMemory = PrototypeMemoryBank
