from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def _to_markdown(frame: pd.DataFrame) -> str:
    text_frame = frame.fillna("").astype(str)
    header = "| " + " | ".join(text_frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in text_frame.columns) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in text_frame.to_numpy()]
    return "\n".join([header, divider, *rows]) + "\n"


VARIANTS = [
    "full_lsdseg",
    "fixed_k_experts",
    "random_routing",
    "frozen_assignment",
    "global_only_descriptor",
    "static_adapters",
    "uniform_ensemble_control",
]


def _template(dataset: str, metadata_key: str | None) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": dataset,
                "variant": variant,
                "metadata_key": metadata_key or "",
                "dice": pd.NA,
                "boundary_f1": pd.NA,
                "worst_subdomain_dice": pd.NA,
                "routing_entropy": pd.NA,
                "prototype_purity": pd.NA,
                "nmi": pd.NA,
                "ari": pd.NA,
                "expert_specialization_score": pd.NA,
                "status": "template_only_run_capacity_controls_to_fill",
            }
            for variant in VARIANTS
        ]
    )


def _purity(assignments: pd.Series, labels: pd.Series) -> float:
    frame = pd.DataFrame({"assignment": assignments, "label": labels}).dropna()
    if frame.empty:
        return float("nan")
    total = 0
    for _, group in frame.groupby("assignment"):
        total += int(group["label"].value_counts().max())
    return float(total / len(frame))


def _from_csv(path: Path, dataset: str, metadata_key: str | None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "variant" not in frame.columns:
        raise ValueError("Capacity analysis CSV must contain a variant column.")
    rows = []
    for variant, group in frame.groupby("variant"):
        row = {
            "dataset": dataset,
            "variant": variant,
            "metadata_key": metadata_key or "",
            "dice": float(group["dice"].mean()) if "dice" in group.columns else pd.NA,
            "boundary_f1": float(group["boundary_f1"].mean()) if "boundary_f1" in group.columns else pd.NA,
            "worst_subdomain_dice": float(group["worst_subdomain_dice"].mean()) if "worst_subdomain_dice" in group.columns else pd.NA,
            "routing_entropy": float(group["routing_entropy_mean"].mean()) if "routing_entropy_mean" in group.columns else pd.NA,
            "prototype_purity": pd.NA,
            "nmi": pd.NA,
            "ari": pd.NA,
            "expert_specialization_score": float(group["expert_specialization_score"].mean()) if "expert_specialization_score" in group.columns else pd.NA,
            "status": "computed_from_input_csv",
        }
        if metadata_key and metadata_key in group.columns and "assigned_prototype" in group.columns:
            clean = group[[metadata_key, "assigned_prototype"]].dropna()
            if not clean.empty:
                row["prototype_purity"] = _purity(clean["assigned_prototype"], clean[metadata_key])
                row["nmi"] = float(normalized_mutual_info_score(clean[metadata_key], clean["assigned_prototype"]))
                row["ari"] = float(adjusted_rand_score(clean[metadata_key], clean["assigned_prototype"]))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze sub-domain discovery versus generic capacity controls.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--metadata-key", default=None)
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    out_dir = args.out if args.out.suffix == "" else args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "subdomain_capacity_summary.csv" if args.out.suffix == "" else args.out

    if args.input_csv:
        output = _from_csv(args.input_csv, args.dataset, args.metadata_key)
    else:
        output = _template(args.dataset, args.metadata_key)
        print("No capacity-control CSV was provided. Wrote a template without fabricated values.")
    output.to_csv(out_path, index=False)
    print(_to_markdown(output))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
