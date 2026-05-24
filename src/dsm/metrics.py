from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

try:
    from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
except Exception:  # pragma: no cover - optional dependency fallback
    binary_dilation = None
    binary_erosion = None
    distance_transform_edt = None


def _binarize(prediction: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (prediction > threshold).float()


def _surface_mask(mask: np.ndarray) -> np.ndarray:
    if binary_erosion is None:
        raise RuntimeError("scipy is required for surface-based boundary metrics.")
    return mask ^ binary_erosion(mask)


def dice_score(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    prediction = _binarize(prediction)
    target = _binarize(target)
    intersection = (prediction * target).sum().item()
    denominator = prediction.sum().item() + target.sum().item()
    return float((2.0 * intersection + eps) / (denominator + eps))


def iou_score(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    prediction = _binarize(prediction)
    target = _binarize(target)
    intersection = (prediction * target).sum().item()
    union = prediction.sum().item() + target.sum().item() - intersection
    return float((intersection + eps) / (union + eps))


def boundary_f1_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    tolerance: float = 2.0,
    spacing: Sequence[float] | None = None,
) -> float:
    if binary_erosion is None or distance_transform_edt is None:
        return float("nan")
    pred = _binarize(prediction).squeeze().cpu().numpy().astype(bool)
    gt = _binarize(target).squeeze().cpu().numpy().astype(bool)
    pred_boundary = _surface_mask(pred)
    gt_boundary = _surface_mask(gt)
    if pred_boundary.sum() == 0 and gt_boundary.sum() == 0:
        return 1.0
    if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
        return 0.0
    gt_dt = distance_transform_edt(~gt_boundary, sampling=spacing)
    pred_dt = distance_transform_edt(~pred_boundary, sampling=spacing)
    precision = float((gt_dt[pred_boundary] <= tolerance).mean()) if pred_boundary.any() else 0.0
    recall = float((pred_dt[gt_boundary] <= tolerance).mean()) if gt_boundary.any() else 0.0
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def hd95_score(prediction: torch.Tensor, target: torch.Tensor, spacing: Sequence[float] | None = None) -> float:
    if binary_erosion is None or distance_transform_edt is None:
        return float("nan")
    pred = _binarize(prediction).squeeze().cpu().numpy().astype(bool)
    gt = _binarize(target).squeeze().cpu().numpy().astype(bool)
    pred_boundary = _surface_mask(pred)
    gt_boundary = _surface_mask(gt)
    if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
        return float("nan")
    dt_pred = distance_transform_edt(~pred_boundary, sampling=spacing)
    dt_gt = distance_transform_edt(~gt_boundary, sampling=spacing)
    distances = np.concatenate([dt_gt[pred_boundary], dt_pred[gt_boundary]])
    return float(np.percentile(distances, 95))


def assd_score(prediction: torch.Tensor, target: torch.Tensor, spacing: Sequence[float] | None = None) -> float:
    if binary_erosion is None or distance_transform_edt is None:
        return float("nan")
    pred = _binarize(prediction).squeeze().cpu().numpy().astype(bool)
    gt = _binarize(target).squeeze().cpu().numpy().astype(bool)
    pred_boundary = _surface_mask(pred)
    gt_boundary = _surface_mask(gt)
    if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
        return float("nan")
    dt_pred = distance_transform_edt(~pred_boundary, sampling=spacing)
    dt_gt = distance_transform_edt(~gt_boundary, sampling=spacing)
    distances = np.concatenate([dt_gt[pred_boundary], dt_pred[gt_boundary]])
    return float(distances.mean())


def expected_calibration_error(prob: torch.Tensor, target: torch.Tensor, n_bins: int = 15) -> float:
    prob_flat = prob.detach().flatten().cpu()
    target_flat = target.detach().flatten().cpu()
    pred_flat = (prob_flat > 0.5).float()
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = torch.tensor(0.0)
    for bin_idx in range(n_bins):
        left, right = bin_edges[bin_idx], bin_edges[bin_idx + 1]
        mask = (prob_flat >= left) & (prob_flat < right if bin_idx < n_bins - 1 else prob_flat <= right)
        if not mask.any():
            continue
        conf = prob_flat[mask].mean()
        acc = (pred_flat[mask] == target_flat[mask]).float().mean()
        ece = ece + mask.float().mean() * torch.abs(conf - acc)
    return float(ece.item())


def aggregate_subdomain_metrics(rows: list[dict[str, float | str]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["subdomain_id"]), []).append(float(row["dice"]))
    means = {key: float(np.mean(values)) for key, values in grouped.items()}
    if not means:
        return {"worst_subdomain_dice": float("nan"), "subdomain_variance": float("nan")}
    return {
        "worst_subdomain_dice": min(means.values()),
        "subdomain_variance": float(np.var(list(means.values()))),
    }


def predictive_entropy(prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = prob.clamp(eps, 1.0 - eps)
    return -(prob * prob.log() + (1.0 - prob) * (1.0 - prob).log())
