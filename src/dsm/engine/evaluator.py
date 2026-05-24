from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..metrics import aggregate_subdomain_metrics, assd_score, boundary_f1_score, dice_score, expected_calibration_error, hd95_score, iou_score
from ..utils.decorators import ensure_directory, logged_call, timed


@dataclass(slots=True)
class Evaluator:
    model: torch.nn.Module
    device: torch.device
    config: dict[str, Any] | None = None
    runtime_metrics: dict[str, float] = field(default_factory=dict)

    def _logit_stats(self, logits: torch.Tensor | None, index: int) -> tuple[float, float]:
        if not isinstance(logits, torch.Tensor):
            return float("nan"), float("nan")
        prob = torch.sigmoid(logits[index])
        return float(prob.mean().item()), float(prob.std().item())

    def _maybe_save_artifacts(self, output_path: Path, sample_id: str, outputs: dict[str, torch.Tensor | int], index: int) -> None:
        if not self.config or not self.config.get("evaluation", {}).get("save_predictions", False):
            return
        artifact_dir = output_path / "sample_artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        routing_entropy = outputs["routing_entropy"][index]
        payload = {
            "final_probability": outputs["probabilities"][index].detach().cpu().numpy(),
            "uncertainty": outputs["uncertainty"][index].detach().cpu().numpy(),
            "routing_weights": outputs["routing_weights"][index].detach().cpu().numpy(),
            "routing_entropy": routing_entropy.detach().cpu().numpy() if torch.is_tensor(routing_entropy) else routing_entropy,
            "assigned_prototype": int(outputs["assignments"][index].item()),
            "prototype_count": int(outputs["prototype_count"]),
            "max_similarity": float(outputs["max_similarity"][index].item()),
            "sam_native_prior_used": float(outputs["sam_native_prior_used"].item()),
            "sam_decoder_fallback_used": float(outputs["sam_decoder_fallback_used"].item()),
            "sam_mask_prior_available": float(outputs["sam_mask_prior_available"].item()),
        }
        prompt_prior_logits = outputs.get("prompt_prior_logits")
        decoder_prior_logits = outputs.get("decoder_prior_logits")
        native_prior_logits = outputs.get("sam_native_prior_logits", outputs.get("sam_mask_prior_logits"))
        if isinstance(prompt_prior_logits, torch.Tensor):
            payload["prompt_prior_probability"] = torch.sigmoid(prompt_prior_logits[index]).detach().cpu().numpy()
        if isinstance(decoder_prior_logits, torch.Tensor):
            payload["decoder_prior_probability"] = torch.sigmoid(decoder_prior_logits[index]).detach().cpu().numpy()
        if isinstance(native_prior_logits, torch.Tensor):
            native_prob = torch.sigmoid(native_prior_logits[index]).detach().cpu().numpy()
            payload["sam_native_prior_probability"] = native_prob
            payload["sam_mask_prior_probability"] = native_prob
        top_k = min(2, outputs["expert_probabilities"].shape[1])
        if outputs["routing_weights"].dim() == 2:
            ranked = outputs["routing_weights"][index].argsort(descending=True)[:top_k]
        else:
            ranked = outputs["routing_weights"][index].mean(dim=(1, 2)).argsort(descending=True)[:top_k]
        for rank, expert_id in enumerate(ranked.tolist()):
            payload[f"expert_probability_top{rank + 1}"] = outputs["expert_probabilities"][index, expert_id].detach().cpu().numpy()
        np.savez_compressed(artifact_dir / f"{sample_id}.npz", **payload)

    @logged_call()
    @ensure_directory(lambda self, loader, output_dir: output_dir)
    @timed("evaluate_seconds")
    def evaluate(self, loader, output_dir: str) -> dict[str, float]:
        output_path = Path(output_dir)
        rows: list[dict[str, float | str]] = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc="evaluate", leave=False):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                outputs = self.model(images, training=False, update_prototypes=False)
                for index in range(images.shape[0]):
                    sample_id = str(batch["sample_id"][index])
                    prob = outputs["probabilities"][index]
                    target = masks[index]
                    routing_entropy = outputs["routing_entropy"][index]
                    uncertainty = outputs["uncertainty"][index]
                    prompt_prior_mean, prompt_prior_std = self._logit_stats(outputs.get("prompt_prior_logits"), index)
                    decoder_prior_mean, decoder_prior_std = self._logit_stats(outputs.get("decoder_prior_logits"), index)
                    sam_native_prior_mean, sam_native_prior_std = self._logit_stats(outputs.get("sam_native_prior_logits"), index)
                    prior_available = float(outputs["sam_mask_prior_available"].item())
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "subdomain_id": str(batch["subdomain_id"][index]),
                            "assigned_prototype": int(outputs["assignments"][index].item()),
                            "max_similarity": float(outputs["max_similarity"][index].item()),
                            "routing_entropy_mean": float(routing_entropy.float().mean().item()),
                            "uncertainty_mean": float(uncertainty.float().mean().item()),
                            "sam_native_prior_used": float(outputs["sam_native_prior_used"].item()),
                            "sam_decoder_fallback_used": float(outputs["sam_decoder_fallback_used"].item()),
                            "sam_mask_prior_available": prior_available,
                            "sam_native_prior_mean": sam_native_prior_mean,
                            "sam_native_prior_std": sam_native_prior_std,
                            "decoder_prior_mean": decoder_prior_mean,
                            "decoder_prior_std": decoder_prior_std,
                            "prompt_prior_mean": prompt_prior_mean,
                            "prompt_prior_std": prompt_prior_std,
                            "sam_mask_prior_mean": sam_native_prior_mean,
                            "sam_mask_prior_std": sam_native_prior_std,
                            "dice": dice_score(prob, target),
                            "iou": iou_score(prob, target),
                            "hd95": hd95_score(prob, target),
                            "assd": assd_score(prob, target),
                            "boundary_f1": boundary_f1_score(prob, target),
                            "ece": expected_calibration_error(prob, target),
                            "runtime_seconds": float(self.model.runtime_metrics.get("forward_seconds", 0.0)),
                            "peak_gpu_memory_mb": float(self.model.runtime_metrics.get("peak_gpu_memory_mb", 0.0)),
                            "prototype_count": float(outputs["prototype_count"]),
                        }
                    )
                    self._maybe_save_artifacts(output_path, sample_id, outputs, index)
        df = pd.DataFrame(rows)
        df.to_csv(output_path / "per_sample_metrics.csv", index=False)
        summary = {
            "dice": float(df["dice"].mean()),
            "iou": float(df["iou"].mean()),
            "hd95": float(df["hd95"].dropna().mean()),
            "assd": float(df["assd"].dropna().mean()),
            "boundary_f1": float(df["boundary_f1"].mean()),
            "ece": float(df["ece"].mean()),
            "runtime_seconds": float(df["runtime_seconds"].mean()),
            "peak_gpu_memory_mb": float(df["peak_gpu_memory_mb"].mean()),
            "sam_native_prior_used_rate": float(df["sam_native_prior_used"].mean()),
            "sam_decoder_fallback_rate": float(df["sam_decoder_fallback_used"].mean()),
            "sam_mask_prior_available_rate": float(df["sam_mask_prior_available"].mean()),
        }
        summary.update(aggregate_subdomain_metrics(rows))
        pd.DataFrame([summary]).to_csv(output_path / "summary_metrics.csv", index=False)
        return summary
