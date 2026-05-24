from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..utils.decorators import logged_call, validate_tensor_output


@dataclass(slots=True)
class PromptBundle:
    """Prompt tensors in original image coordinates unless explicitly rescaled for SAM/SAM2."""

    boxes: torch.Tensor
    points: torch.Tensor
    point_labels: torch.Tensor
    coarse_logits: torch.Tensor
    mask_input: torch.Tensor | None = None
    original_size: tuple[int, int] | None = None
    source_size: tuple[int, int] | None = None


def _validate_prompt_bundle(prompts: PromptBundle) -> None:
    if prompts.boxes.dim() != 2 or prompts.boxes.shape[1] != 4:
        raise ValueError(f"PromptBundle.boxes must have shape [B,4], got {tuple(prompts.boxes.shape)}")
    if prompts.points.dim() != 3 or prompts.points.shape[-1] != 2:
        raise ValueError(f"PromptBundle.points must have shape [B,N,2], got {tuple(prompts.points.shape)}")
    if prompts.point_labels.dim() != 2:
        raise ValueError(f"PromptBundle.point_labels must have shape [B,N], got {tuple(prompts.point_labels.shape)}")
    if prompts.points.shape[:2] != prompts.point_labels.shape:
        raise ValueError(
            "PromptBundle.points and point_labels must agree on [B,N]. "
            f"Got points {tuple(prompts.points.shape)} and labels {tuple(prompts.point_labels.shape)}."
        )
    if prompts.mask_input is not None:
        if prompts.mask_input.dim() != 4 or prompts.mask_input.shape[1] != 1:
            raise ValueError(f"PromptBundle.mask_input must have shape [B,1,Hm,Wm], got {tuple(prompts.mask_input.shape)}")
        if prompts.mask_input.shape[0] != prompts.boxes.shape[0]:
            raise ValueError("PromptBundle.mask_input batch size must match prompt boxes batch size.")


def scale_prompts_to_sam_input(
    prompts: PromptBundle,
    original_size: tuple[int, int],
    sam_image_size: int,
    mask_input_size: int | None = None,
) -> PromptBundle:
    """Scale boxes/points to SAM image coordinates and resize mask_input to low-res prompt-mask size."""

    _validate_prompt_bundle(prompts)
    source_h, source_w = original_size
    scale_x = sam_image_size / max(source_w, 1)
    scale_y = sam_image_size / max(source_h, 1)
    boxes = prompts.boxes.clone()
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y
    points = prompts.points.clone()
    points[..., 0] = points[..., 0] * scale_x
    points[..., 1] = points[..., 1] * scale_y
    mask_input = None
    if prompts.mask_input is not None:
        resolved_mask_size = int(mask_input_size or 256)
        mask_input = F.interpolate(prompts.mask_input, size=(resolved_mask_size, resolved_mask_size), mode="bilinear", align_corners=False)
    scaled = PromptBundle(
        boxes=boxes,
        points=points,
        point_labels=prompts.point_labels.clone(),
        coarse_logits=prompts.coarse_logits,
        mask_input=mask_input,
        original_size=prompts.original_size,
        source_size=(sam_image_size, sam_image_size),
    )
    _validate_prompt_bundle(scaled)
    return scaled


class AutomaticPromptGenerator(nn.Module):
    def __init__(self, in_channels: int, prompt_channels: int, num_points: int, fallback_to_center: bool = True) -> None:
        super().__init__()
        self.num_points = num_points
        self.fallback_to_center = fallback_to_center
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, prompt_channels, kernel_size=1),
        )

    def _scale_box_to_original(self, box: torch.Tensor, source_hw: tuple[int, int], target_hw: tuple[int, int]) -> torch.Tensor:
        source_h, source_w = source_hw
        target_h, target_w = target_hw
        scale_x = target_w / max(source_w, 1)
        scale_y = target_h / max(source_h, 1)
        scaled = box.clone()
        scaled[[0, 2]] = scaled[[0, 2]] * scale_x
        scaled[[1, 3]] = scaled[[1, 3]] * scale_y
        return scaled

    def _scale_points_to_original(self, points: torch.Tensor, source_hw: tuple[int, int], target_hw: tuple[int, int]) -> torch.Tensor:
        source_h, source_w = source_hw
        target_h, target_w = target_hw
        scale_x = target_w / max(source_w, 1)
        scale_y = target_h / max(source_h, 1)
        scaled = points.clone()
        scaled[..., 0] = scaled[..., 0] * scale_x
        scaled[..., 1] = scaled[..., 1] * scale_y
        return scaled

    @logged_call()
    @validate_tensor_output
    def forward(self, features: torch.Tensor, original_size: tuple[int, int] | None = None) -> PromptBundle:
        coarse_logits = self.head(features)
        coarse_mask = (torch.sigmoid(coarse_logits) > 0.5).float()
        source_size = tuple(int(value) for value in coarse_mask.shape[-2:])
        target_size = original_size or source_size
        boxes = self._boxes_from_mask(coarse_mask, source_size=source_size, target_size=target_size)
        points, point_labels = self._points_from_mask(coarse_mask, source_size=source_size, target_size=target_size)
        mask_input = F.interpolate(coarse_logits[:, :1], size=target_size, mode="bilinear", align_corners=False)
        prompts = PromptBundle(
            boxes=boxes,
            points=points,
            point_labels=point_labels,
            coarse_logits=coarse_logits[:, :1],
            mask_input=mask_input,
            original_size=target_size,
            source_size=source_size,
        )
        _validate_prompt_bundle(prompts)
        return prompts

    def _boxes_from_mask(
        self,
        mask: torch.Tensor,
        source_size: tuple[int, int],
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        boxes = []
        for sample in mask[:, 0]:
            coords = torch.nonzero(sample > 0.5, as_tuple=False)
            if coords.numel() == 0:
                box = torch.tensor(
                    [0.0, 0.0, float(max(source_size[1] - 1, 1)), float(max(source_size[0] - 1, 1))],
                    device=mask.device,
                )
            else:
                y0, x0 = coords.min(dim=0).values
                y1, x1 = coords.max(dim=0).values
                box = torch.stack([x0, y0, x1, y1]).float()
            boxes.append(self._scale_box_to_original(box, source_size, target_size))
        return torch.stack(boxes, dim=0)

    def _points_from_mask(
        self,
        mask: torch.Tensor,
        source_size: tuple[int, int],
        target_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        points = []
        labels = []
        center_point = torch.tensor([[source_size[1] / 2.0, source_size[0] / 2.0]], device=mask.device)
        for sample in mask[:, 0]:
            coords = torch.nonzero(sample > 0.5, as_tuple=False)
            if coords.numel() == 0:
                chosen = center_point if self.fallback_to_center else torch.zeros((self.num_points, 2), device=mask.device)
                chosen = chosen.repeat(self.num_points, 1)[: self.num_points]
                point_labels = torch.ones(self.num_points, device=mask.device)
            else:
                step = max(1, coords.shape[0] // self.num_points)
                chosen = coords[::step][: self.num_points][:, [1, 0]].float()
                if chosen.shape[0] < self.num_points:
                    pad = chosen[-1:].repeat(self.num_points - chosen.shape[0], 1)
                    chosen = torch.cat([chosen, pad], dim=0)
                point_labels = torch.ones(self.num_points, device=mask.device)
            points.append(self._scale_points_to_original(chosen, source_size, target_size))
            labels.append(point_labels)
        return torch.stack(points, dim=0), torch.stack(labels, dim=0)
