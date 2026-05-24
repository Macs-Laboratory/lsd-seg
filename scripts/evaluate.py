from __future__ import annotations

import argparse

from dsm.config import load_config
from dsm.experiments.runner import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a configured experiment.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    runner = ExperimentRunner(config)
    print(runner.run(args.experiment))


if __name__ == "__main__":
    main()
