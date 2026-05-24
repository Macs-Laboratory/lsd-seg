from __future__ import annotations

import torch

from ..config import Config
from ..utils.decorators import logged_call
from .base import BaseExperiment
from .external import ExternalBaselineExperiment
from .native import NativeTrainingExperiment
from .specs import PAPER_EXPERIMENT_SPECS


class ExperimentRunner:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._registry: dict[str, BaseExperiment] = self._build_registry()

    @logged_call()
    def _build_registry(self) -> dict[str, BaseExperiment]:
        registry: dict[str, BaseExperiment] = {}
        for name, spec in PAPER_EXPERIMENT_SPECS.items():
            if spec.kind == "native":
                registry[name] = NativeTrainingExperiment(spec)
            else:
                registry[name] = ExternalBaselineExperiment(spec)
        return registry

    @logged_call()
    def run(self, experiment_name: str) -> dict:
        if experiment_name not in self._registry:
            available = ", ".join(sorted(self._registry))
            raise KeyError(f"Unknown experiment '{experiment_name}'. Available experiments: {available}")
        experiment = self._registry[experiment_name]
        context = experiment.create_context(self.config, self.device)
        return experiment.run(context)
