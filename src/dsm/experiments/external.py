from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..models.baselines import ExternalBaselineAdapter, PAPER_BASELINES
from ..utils.decorators import ensure_directory, logged_call
from .base import BaseExperiment, ExperimentContext, ExperimentSpec


@dataclass(slots=True)
class ExternalExperimentResult:
    experiment: str
    kind: str
    command: str
    repository: str


class ExternalBaselineExperiment(BaseExperiment):
    def __init__(self, spec: ExperimentSpec) -> None:
        super().__init__(spec)
        self.adapter = ExternalBaselineAdapter(PAPER_BASELINES[spec.name])

    @logged_call()
    @ensure_directory(lambda self, context: context.run_dir)
    def run(self, context: ExperimentContext) -> dict[str, Any]:
        command = self.adapter.command(context.config["dataset"]["manifest_path"], str(context.run_dir))
        (context.run_dir / "external_command.txt").write_text(command + "\n")
        return ExternalExperimentResult(
            experiment=context.name,
            kind="external",
            command=command,
            repository=PAPER_BASELINES[context.name].repository,
        ).__dict__
