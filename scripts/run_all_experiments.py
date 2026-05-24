from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from dsm.config import load_config
from dsm.experiments.runner import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run every experiment listed in configs/experiments.yaml.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--experiment-list", default="configs/experiments.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_names = yaml.safe_load(Path(args.experiment_list).read_text())["paper_experiments"]
    runner = ExperimentRunner(config)
    summaries = {}
    for experiment_name in experiment_names:
        summaries[experiment_name] = runner.run(experiment_name)
    output_path = Path(config["output_dir"]) / "paper_experiment_index.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
