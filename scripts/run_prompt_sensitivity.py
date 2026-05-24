from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _to_markdown(frame: pd.DataFrame) -> str:
    text_frame = frame.fillna("").astype(str)
    header = "| " + " | ".join(text_frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in text_frame.columns) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in text_frame.to_numpy()]
    return "\n".join([header, divider, *rows]) + "\n"


METRIC_COLUMNS = [
    "perturbation",
    "level",
    "dice",
    "boundary_f1",
    "routing_entropy",
    "assigned_prototype_change_rate",
    "mean_max_prototype_similarity",
    "sam_decoder_fallback_rate",
    "status",
]


def _parse_levels(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _write_template(args: argparse.Namespace) -> pd.DataFrame:
    rows = [
        {
            "perturbation": args.perturbation,
            "level": level,
            "dice": pd.NA,
            "boundary_f1": pd.NA,
            "routing_entropy": pd.NA,
            "assigned_prototype_change_rate": pd.NA,
            "mean_max_prototype_similarity": pd.NA,
            "sam_decoder_fallback_rate": pd.NA,
            "status": "template_only_run_perturbed_evaluation_to_fill",
        }
        for level in _parse_levels(args.levels)
    ]
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def _aggregate_input(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    required = {"perturbation", "level"}
    if not required.issubset(frame.columns):
        raise ValueError("Prompt sensitivity CSV must contain perturbation and level columns.")
    metric_columns = [column for column in METRIC_COLUMNS if column in frame.columns and column not in {"perturbation", "level", "status"}]
    grouped = frame.groupby(["perturbation", "level"], dropna=False)[metric_columns].mean(numeric_only=True).reset_index()
    grouped["status"] = "computed_from_input_csv"
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or aggregate prompt sensitivity analyses.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--perturbation", default="box_jitter")
    parser.add_argument("--levels", default="0,0.05,0.10,0.20,0.30")
    parser.add_argument("--input-csv", type=Path, default=None, help="Optional CSV from perturbed evaluations to aggregate.")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if args.input_csv:
        output = _aggregate_input(args.input_csv)
    else:
        output = _write_template(args)
        print(
            "No perturbed evaluation CSV was provided. "
            "Wrote a template so results can be filled after running prompt-perturbed evaluation."
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    print(_to_markdown(output))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
