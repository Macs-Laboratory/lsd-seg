from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from tqdm import tqdm

from ..losses import LSDSegLoss
from ..metrics import dice_score
from ..utils.decorators import ensure_directory, logged_call, timed
from ..utils.reproducibility import RunMetadata


@dataclass(slots=True)
class Trainer:
    model: torch.nn.Module
    config: dict[str, Any]
    device: torch.device
    criterion: LSDSegLoss | None = None
    runtime_metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.criterion is None:
            loss_cfg = self.config.get("loss", {})
            self.criterion = LSDSegLoss(
                expert_weight=loss_cfg.get("expert_weight", loss_cfg.get("lambda_expert", 0.2)),
                ortho_weight=loss_cfg.get("ortho_weight", loss_cfg.get("lambda_ortho", 0.01)),
                balance_weight=loss_cfg.get("balance_weight", loss_cfg.get("lambda_balance", 0.01)),
                proto_weight=loss_cfg.get("proto_weight", 0.0),
                prompt_weight=loss_cfg.get("prompt_weight", 0.1),
            )

    @logged_call()
    @ensure_directory(lambda self, train_loader, val_loader, output_dir: output_dir)
    @timed("fit_seconds")
    def fit(self, train_loader, val_loader, output_dir: str) -> RunMetadata:
        output_path = Path(output_dir)
        trainable = [(name, parameter) for name, parameter in self.model.named_parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable parameters were found before optimizer creation.")
        model_cfg = self.config.get("model", {})
        if model_cfg.get("encoder_type") == "sam2" and model_cfg.get("sam2_projection_trainable", True):
            projection_trainable = any("projection" in name or "projections" in name for name, _ in trainable)
            if not projection_trainable:
                raise RuntimeError("SAM2 projection_trainable=True but no trainable projection parameters were found.")
        optimizer = AdamW(
            (parameter for _, parameter in trainable),
            lr=self.config["optimizer"]["lr"],
            weight_decay=self.config["optimizer"]["weight_decay"],
        )
        best_val = float("-inf")
        metadata = RunMetadata(config=self.config)

        for epoch in range(self.config["optimizer"]["epochs"]):
            epoch_stats = self._train_epoch(train_loader, optimizer)
            val_stats = self._validate_epoch(val_loader)
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "config": self.config,
                "prototype_summary": self.model.prototype_summary(),
            }
            if val_stats["dice"] > best_val:
                best_val = val_stats["dice"]
                torch.save(checkpoint, output_path / "best_model.pt")

            for key, value in epoch_stats.items():
                metadata.metrics[f"epoch_{epoch + 1}_{key}"] = value
            metadata.metrics[f"val_dice_epoch_{epoch + 1}"] = val_stats["dice"]
        metadata.metrics["best_val_dice"] = best_val
        return metadata

    @logged_call()
    @timed("train_epoch_seconds")
    def _train_epoch(self, train_loader, optimizer: torch.optim.Optimizer) -> dict[str, float]:
        self.model.train()
        aggregates = {
            "loss": 0.0,
            "loss_seg": 0.0,
            "loss_bce": 0.0,
            "loss_dice": 0.0,
            "loss_expert": 0.0,
            "loss_ortho": 0.0,
            "loss_balance": 0.0,
            "loss_proto": 0.0,
            "loss_prompt": 0.0,
            "dice": 0.0,
            "routing_entropy": 0.0,
            "uncertainty": 0.0,
            "max_similarity": 0.0,
            "steps": 0,
        }
        progress = tqdm(train_loader, desc="train", leave=False)
        for batch in progress:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            outputs = self.model(images, targets=masks, training=True, update_prototypes=True)
            loss_dict = self.criterion(outputs, masks)
            optimizer.zero_grad(set_to_none=True)
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["optimizer"]["grad_clip_norm"])
            optimizer.step()

            batch_dice = dice_score(outputs["probabilities"].detach(), masks.detach())
            routing_entropy = outputs["routing_entropy"].float().mean().item()
            uncertainty = outputs["uncertainty"].float().mean().item()
            max_similarity = outputs["similarities"].max(dim=1).values.mean().item()
            for key in ["loss", "loss_seg", "loss_bce", "loss_dice", "loss_expert", "loss_ortho", "loss_balance", "loss_proto", "loss_prompt"]:
                value = loss_dict.get(key)
                if value is None:
                    continue
                aggregates[key] += float(value.detach().cpu())
            aggregates["dice"] += batch_dice
            aggregates["routing_entropy"] += routing_entropy
            aggregates["uncertainty"] += uncertainty
            aggregates["max_similarity"] += max_similarity
            aggregates["steps"] += 1
            progress.set_postfix(
                loss=float(loss_dict["loss"].detach().cpu()),
                dice=batch_dice,
                K=int(outputs["prototype_count"]),
            )

        steps = max(int(aggregates["steps"]), 1)
        summary = {
            "loss": aggregates["loss"] / steps,
            "loss_seg": aggregates["loss_seg"] / steps,
            "loss_bce": aggregates["loss_bce"] / steps,
            "loss_dice": aggregates["loss_dice"] / steps,
            "loss_expert": aggregates["loss_expert"] / steps,
            "loss_ortho": aggregates["loss_ortho"] / steps,
            "loss_balance": aggregates["loss_balance"] / steps,
            "loss_proto": aggregates["loss_proto"] / steps,
            "loss_prompt": aggregates["loss_prompt"] / steps,
            "dice": aggregates["dice"] / steps,
            "mean_routing_entropy": aggregates["routing_entropy"] / steps,
            "mean_uncertainty": aggregates["uncertainty"] / steps,
            "mean_max_prototype_similarity": aggregates["max_similarity"] / steps,
            "num_active_prototypes": int(self.model.prototype_summary()["num_active"]),
        }
        print(
            "epoch summary:",
            {
                **summary,
                "support_counts": self.model.prototype_summary()["support_counts"],
            },
        )
        return summary

    @logged_call()
    @timed("validate_epoch_seconds")
    def _validate_epoch(self, val_loader) -> dict[str, float]:
        self.model.eval()
        scores = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                outputs = self.model(images, targets=masks, training=False, update_prototypes=False)
                scores.append(dice_score(outputs["probabilities"], masks))
        return {"dice": float(sum(scores) / max(len(scores), 1))}
