from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..config import Config
from ..data.datamodule import SegmentationDataModule
from ..models.full_model import DynamicSubdomainModel
from ..utils.decorators import logged_call
from ..utils.reproducibility import seed_everything


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(slots=True)
class ExperimentContext:
    name: str
    config: dict[str, Any]
    run_dir: Path
    device: torch.device


class DataModuleFactory:
    @logged_call()
    def build(self, run_config: dict[str, Any]) -> SegmentationDataModule:
        dataset = run_config["dataset"]
        return SegmentationDataModule(
            manifest_path=dataset["manifest_path"],
            image_size=dataset["image_size"],
            batch_size=dataset["batch_size"],
            num_workers=dataset["num_workers"],
        )


class ModelFactory:
    @logged_call()
    def build(self, run_config: dict[str, Any], device: torch.device) -> DynamicSubdomainModel:
        return DynamicSubdomainModel(**run_config["model"]).to(device)


@dataclass(slots=True)
class ExperimentSpec:
    name: str
    kind: str
    description: str = ""
    overrides: dict[str, Any] | None = None

    def merged_config(self, base: Config) -> dict[str, Any]:
        return deep_update(base.raw, self.overrides or {})


class BaseExperiment(ABC):
    def __init__(self, spec: ExperimentSpec) -> None:
        self.spec = spec

    @logged_call()
    def create_context(self, base_config: Config, device: torch.device) -> ExperimentContext:
        run_config = self.spec.merged_config(base_config)
        run_dir = Path(run_config["output_dir"]) / self.spec.name
        run_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(run_config["seed"])
        return ExperimentContext(
            name=self.spec.name,
            config=run_config,
            run_dir=run_dir,
            device=device,
        )

    @abstractmethod
    def run(self, context: ExperimentContext) -> dict[str, Any]:
        raise NotImplementedError
