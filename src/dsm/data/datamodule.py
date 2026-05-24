from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import DataLoader

from .datasets import ManifestSegmentationDataset
from ..utils.decorators import logged_call


@dataclass(slots=True)
class SegmentationDataModule:
    manifest_path: str
    image_size: int
    batch_size: int
    num_workers: int

    @logged_call()
    def train_dataloader(self) -> DataLoader:
        dataset = ManifestSegmentationDataset(self.manifest_path, "train", self.image_size)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    @logged_call()
    def val_dataloader(self) -> DataLoader:
        dataset = ManifestSegmentationDataset(self.manifest_path, "val", self.image_size)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    @logged_call()
    def test_dataloader(self) -> DataLoader:
        dataset = ManifestSegmentationDataset(self.manifest_path, "test", self.image_size)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
