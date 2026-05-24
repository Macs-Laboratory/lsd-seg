from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dsm.experiments.analyses import plot_sensitivity, plot_win_loss_heatmap, wilcoxon_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate results and produce paper analyses.")
    parser.add_argument("--result-csv", nargs="+", required=True)
    parser.add_argument("--metric", default="dice")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--sensitivity-csv", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wilcoxon_table(args.result_csv, args.metric, str(output_dir / "wilcoxon.csv"))
    if len(args.result_csv) >= 2:
        frames = [pd.read_csv(path) for path in args.result_csv]
        deltas = pd.concat(
            [
                (frames[0][args.metric] - frame[args.metric]).rename(f"baseline_{index}")
                for index, frame in enumerate(frames[1:], start=1)
            ],
            axis=1,
        )
        plot_win_loss_heatmap(deltas.T, str(output_dir / "win_loss_heatmap.png"))
    if args.sensitivity_csv:
        sensitivity = pd.read_csv(args.sensitivity_csv)
        required = {"parameter", "value", args.metric}
        if not required.issubset(sensitivity.columns):
            raise ValueError(f"Sensitivity CSV must contain columns: {sorted(required)}")
        plot_sensitivity(sensitivity.rename(columns={args.metric: "dice"}), str(output_dir / "sensitivity.png"))
    else:
        print("No sensitivity CSV provided; skipping hyperparameter sensitivity plot.")


if __name__ == "__main__":
    main()
