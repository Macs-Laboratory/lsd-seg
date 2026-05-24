from __future__ import annotations

from collections.abc import Sequence
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import roi_align

from ..prompts.auto_prompt import PromptBundle
from ..utils.decorators import logged_call, validate_tensor_output


class SubdomainDescriptorHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        descriptor_dim: int,
        use_roi: bool = True,
        normalize: bool = True,
        descriptor_source: str = "deepest",
        pyramid_channels: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if descriptor_source not in {"deepest", "pyramid"}:
            raise ValueError(f"Unsupported descriptor_source '{descriptor_source}'.")
        self.use_roi = use_roi
        self.normalize = normalize
        self.descriptor_source = descriptor_source
        self.pyramid_channels = list(pyramid_channels) if pyramid_channels is not None else [feature_dim]
        global_dim = feature_dim if descriptor_source == "deepest" else int(sum(self.pyramid_channels))
        input_dim = global_dim + (feature_dim if use_roi else 0)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, descriptor_dim),
            nn.GELU(),
            nn.Linear(descriptor_dim, descriptor_dim),
        )

    def _full_image_boxes(self, features: torch.Tensor) -> torch.Tensor:
        b, _, h, w = features.shape
        boxes = []
        for index in range(b):
            boxes.append(
                torch.tensor(
                    [index, 0.0, 0.0, float(max(w - 1, 1)), float(max(h - 1, 1))],
                    device=features.device,
                    dtype=features.dtype,
                )
            )
        return torch.stack(boxes, dim=0)

    def _scale_boxes_to_feature_map(
        self,
        boxes: torch.Tensor,
        original_size: tuple[int, int] | None,
        feature_size: tuple[int, int],
    ) -> torch.Tensor:
        if original_size is None:
            warnings.warn("Original image size was not provided for ROI scaling. Falling back to full-image ROI.", stacklevel=2)
            return torch.empty(0, device=boxes.device, dtype=boxes.dtype)
        original_h, original_w = original_size
        feature_h, feature_w = feature_size
        scale_x = feature_w / max(original_w, 1)
        scale_y = feature_h / max(original_h, 1)
        scaled = boxes.clone()
        scaled[:, [0, 2]] = scaled[:, [0, 2]] * scale_x
        scaled[:, [1, 3]] = scaled[:, [1, 3]] * scale_y
        return scaled

    def _sanitize_boxes(
        self,
        features: torch.Tensor,
        prompts: PromptBundle | None,
        original_size: tuple[int, int] | None,
    ) -> torch.Tensor:
        if prompts is None or prompts.boxes.numel() == 0:
            return self._full_image_boxes(features)
        _, _, h, w = features.shape
        scaled_boxes = self._scale_boxes_to_feature_map(prompts.boxes.to(features.device, dtype=features.dtype), original_size, (h, w))
        if scaled_boxes.numel() == 0:
            return self._full_image_boxes(features)
        boxes = []
        for index, box in enumerate(scaled_boxes):
            if box.numel() != 4:
                boxes.append(self._full_image_boxes(features)[index])
                continue
            x0, y0, x1, y1 = box
            x0 = x0.clamp(0, max(w - 1, 1))
            y0 = y0.clamp(0, max(h - 1, 1))
            x1 = x1.clamp(0, max(w - 1, 1))
            y1 = y1.clamp(0, max(h - 1, 1))
            if (x1 - x0).abs() < 1 or (y1 - y0).abs() < 1:
                boxes.append(self._full_image_boxes(features)[index])
            else:
                boxes.append(torch.tensor([index, x0, y0, x1, y1], device=features.device, dtype=features.dtype))
        return torch.stack(boxes, dim=0)

    def _global_descriptor(self, features: torch.Tensor, pyramid: list[torch.Tensor] | None) -> torch.Tensor:
        if self.descriptor_source == "deepest" or not pyramid:
            return features.mean(dim=(2, 3))
        pooled = [level.mean(dim=(2, 3)) for level in pyramid]
        return torch.cat(pooled, dim=1)

    @logged_call()
    @validate_tensor_output
    def forward(
        self,
        features: torch.Tensor,
        prompts: PromptBundle | None = None,
        pyramid: list[torch.Tensor] | None = None,
        original_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        global_descriptor = self._global_descriptor(features, pyramid)
        if not self.use_roi:
            descriptor = self.mlp(global_descriptor)
            return F.normalize(descriptor, dim=-1, eps=1e-6) if self.normalize else descriptor

        boxes = self._sanitize_boxes(features, prompts, original_size=original_size)
        aligned = roi_align(
            features,
            boxes,
            output_size=(2, 2),
            spatial_scale=1.0,
            aligned=True,
        )
        roi_descriptor = aligned.mean(dim=(2, 3))
        descriptor = self.mlp(torch.cat([global_descriptor, roi_descriptor], dim=1))
        return F.normalize(descriptor, dim=-1, eps=1e-6) if self.normalize else descriptor
