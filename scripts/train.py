from __future__ import annotations

import argparse

from dsm.config import load_config
from dsm.experiments.runner import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSD-Seg or an ablation specified by the experiment registry.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--experiment", default="ours_full")
    args = parser.parse_args()
    config = load_config(args.config)
    runner = ExperimentRunner(config)
    runner.run(args.experiment)


if __name__ == "__main__":
    main()
