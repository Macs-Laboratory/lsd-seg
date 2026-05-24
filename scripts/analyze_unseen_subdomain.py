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


OUTPUT_COLUMNS = [
    "dataset",
    "heldout_subdomain",
    "analysis_type",
    "dice",
    "boundary_f1",
    "max_similarity",
    "routing_entropy",
    "uncertainty",
    "top1_assignment_distribution",
    "status",
]


def _from_csv(path: Path, dataset: str, heldout: str, analysis_type: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    filtered = frame.copy()
    if "subdomain_id" in filtered.columns:
        filtered = filtered[filtered["subdomain_id"].astype(str) == heldout]
    metrics = {
        "dice": "dice",
        "boundary_f1": "boundary_f1",
        "max_similarity": "max_similarity",
        "routing_entropy": "routing_entropy_mean",
        "uncertainty": "uncertainty_mean",
    }
    row: dict[str, object] = {
        "dataset": dataset,
        "heldout_subdomain": heldout,
        "analysis_type": analysis_type,
        "top1_assignment_distribution": "",
        "status": "computed_from_input_csv",
    }
    for output_name, input_name in metrics.items():
        row[output_name] = float(filtered[input_name].mean()) if input_name in filtered.columns and not filtered.empty else pd.NA
    if "assigned_prototype" in filtered.columns and not filtered.empty:
        counts = filtered["assigned_prototype"].value_counts(normalize=True).sort_index()
        row["top1_assignment_distribution"] = ";".join(f"{idx}:{value:.4f}" for idx, value in counts.items())
    return pd.DataFrame([row], columns=OUTPUT_COLUMNS)


def _template(dataset: str, heldout: str, analysis_type: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": dataset,
                "heldout_subdomain": heldout,
                "analysis_type": analysis_type,
                "dice": pd.NA,
                "boundary_f1": pd.NA,
                "max_similarity": pd.NA,
                "routing_entropy": pd.NA,
                "uncertainty": pd.NA,
                "top1_assignment_distribution": pd.NA,
                "status": "template_only_train_leave_one_subdomain_out_to_fill",
            }
        ],
        columns=OUTPUT_COLUMNS,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize leave-one-subdomain-out behavior.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--heldout-subdomain", required=True)
    parser.add_argument("--input-csv", type=Path, default=None, help="Optional per-sample metrics CSV for the held-out subdomain.")
    parser.add_argument("--analysis-type", choices=["metadata", "pseudo"], default="metadata")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    output_dir = args.out if args.out.suffix == "" else args.out.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "unseen_subdomain_summary.csv" if args.out.suffix == "" else args.out

    if args.input_csv:
        output = _from_csv(args.input_csv, args.dataset, args.heldout_subdomain, args.analysis_type)
    else:
        output = _template(args.dataset, args.heldout_subdomain, args.analysis_type)
        print("No held-out per-sample CSV was provided. Wrote a protocol template without fabricated values.")
    output.to_csv(out_path, index=False)
    print(_to_markdown(output))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
