from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import Dataset

from .schemas import SampleRecord


class BaseSegmentationDataset(Dataset):
    def __init__(self, manifest_path: str | Path, split: str) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")
        records = json.loads(self.manifest_path.read_text())
        self.records = [SampleRecord(**item) for item in records if item["split"] == split]
        if len(self.records) == 0:
            raise ValueError(f"Empty dataset for split '{split}' in manifest {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.records)
