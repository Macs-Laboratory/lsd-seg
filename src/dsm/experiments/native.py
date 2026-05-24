from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..engine.evaluator import Evaluator
from ..engine.trainer import Trainer
from ..utils.decorators import logged_call, timed
from .base import BaseExperiment, DataModuleFactory, ExperimentContext, ExperimentSpec, ModelFactory


@dataclass(slots=True)
class NativeTrainingResult:
    experiment: str
    summary: dict[str, float]


class NativeTrainingExperiment(BaseExperiment):
    def __init__(
        self,
        spec: ExperimentSpec,
        data_factory: DataModuleFactory | None = None,
        model_factory: ModelFactory | None = None,
    ) -> None:
        super().__init__(spec)
        self.data_factory = data_factory or DataModuleFactory()
        self.model_factory = model_factory or ModelFactory()

    @logged_call()
    @timed("native_experiment_seconds")
    def run(self, context: ExperimentContext) -> dict[str, Any]:
        datamodule = self.data_factory.build(context.config)
        model = self.model_factory.build(context.config, context.device)
        trainer = Trainer(model=model, config=context.config, device=context.device)
        metadata = trainer.fit(datamodule.train_dataloader(), datamodule.val_dataloader(), str(context.run_dir))
        model.load_artifacts(torch.load(context.run_dir / "best_model.pt", map_location=context.device))
        evaluator = Evaluator(model=model, device=context.device, config=context.config)
        summary = evaluator.evaluate(datamodule.test_dataloader(), str(context.run_dir))
        metadata.metrics.update(summary)
        metadata.save(context.run_dir)
        return NativeTrainingResult(experiment=context.name, summary=summary).__dict__
