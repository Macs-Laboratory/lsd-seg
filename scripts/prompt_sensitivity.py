from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TEMPLATE_COLUMNS = [
    "prompt_error_level",
    "perturbation",
    "dice",
    "boundary_f1",
    "routing_entropy",
    "assigned_prototype_change_rate",
    "mean_max_prototype_similarity",
    "sam_decoder_fallback_rate",
    "status",
]


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "prompt_error_level": pd.NA,
                "perturbation": pd.NA,
                "dice": pd.NA,
                "boundary_f1": pd.NA,
                "routing_entropy": pd.NA,
                "assigned_prototype_change_rate": pd.NA,
                "mean_max_prototype_similarity": pd.NA,
                "sam_decoder_fallback_rate": pd.NA,
                "status": "template_only_no_experiment_values",
            }
        ],
        columns=TEMPLATE_COLUMNS,
    )


def _summarize_input(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if "level" in frame.columns and "prompt_error_level" not in frame.columns:
        frame = frame.rename(columns={"level": "prompt_error_level"})
    required = {"prompt_error_level", "perturbation"}
    if not required.issubset(frame.columns):
        raise ValueError("Input CSV must contain prompt_error_level or level, plus perturbation.")
    metric_columns = [
        column
        for column in TEMPLATE_COLUMNS
        if column in frame.columns and column not in {"prompt_error_level", "perturbation", "status"}
    ]
    summary = frame.groupby(["prompt_error_level", "perturbation"], dropna=False)[metric_columns].mean(numeric_only=True).reset_index()
    summary["status"] = "computed_from_input_csv"
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a prompt-sensitivity report from real CSVs or an NA template.")
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--output", default="reports/prompt_sensitivity.json")
    parser.add_argument("--template-only", action="store_true")
    args = parser.parse_args()

    if args.input_csv and args.template_only:
        raise ValueError("Use either --input-csv or --template-only, not both.")
    if args.input_csv:
        frame = _summarize_input(args.input_csv)
    else:
        if not args.template_only:
            print("No input CSV supplied; writing NA template. Pass --input-csv to summarize actual results.")
        frame = _template()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path.with_suffix(".csv"), index=False)
    output_path.write_text(json.dumps(frame.where(pd.notna(frame), None).to_dict(orient="records"), indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"Wrote {output_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
