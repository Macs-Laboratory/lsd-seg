from __future__ import annotations

from pathlib import Path

import torchvision.io as io
from torch import Tensor

from .base import BaseSegmentationDataset
from .transforms import SegmentationTransform


class ManifestSegmentationDataset(BaseSegmentationDataset):
    def __init__(self, manifest_path: str | Path, split: str, image_size: int) -> None:
        super().__init__(manifest_path=manifest_path, split=split)
        self.transform = SegmentationTransform(image_size=image_size)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        record = self.records[index]
        if not Path(record.mask_path).exists():
            raise FileNotFoundError(f"Missing mask file: {record.mask_path}")
        image = io.read_image(record.image_path)
        mask = io.read_image(record.mask_path)[:1]
        image, mask = self.transform(image, mask)
        return {
            "sample_id": record.sample_id,
            "image": image,
            "mask": mask,
            "subdomain_id": record.subdomain_id or "unknown",
        }
