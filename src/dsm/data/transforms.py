from __future__ import annotations

from dataclasses import dataclass

import torch
import torchvision.transforms.functional as tvf


@dataclass(slots=True)
class SegmentationTransform:
    image_size: int

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if image.shape[0] > 3:
            image = image[:3]
        image = tvf.resize(image, [self.image_size, self.image_size], antialias=True)
        mask = tvf.resize(mask, [self.image_size, self.image_size], antialias=False)
        image = tvf.convert_image_dtype(image, torch.float32)
        mask = (mask > 0.5).float()
        return image, mask
