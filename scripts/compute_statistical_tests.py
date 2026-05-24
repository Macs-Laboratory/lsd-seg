from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._\n"
    text_frame = frame.fillna("").astype(str)
    header = "| " + " | ".join(text_frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in text_frame.columns) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in text_frame.to_numpy()]
    return "\n".join([header, divider, *rows]) + "\n"


def _require_scipy():
    try:
        from scipy.stats import wilcoxon
    except ImportError as exc:
        raise RuntimeError(
            "Statistical testing requires scipy. Please install scipy or export paired differences and test externally."
        ) from exc
    return wilcoxon


def _load_metric_frame(csv_path: Path, metric: str) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    required_wide = {"dataset", "method", metric}
    required_long = {"dataset", "metric", "method", "value"}
    if required_wide.issubset(frame.columns):
        return frame[["dataset", "method", metric]].rename(columns={metric: "value"})
    if required_long.issubset(frame.columns):
        subset = frame[frame["metric"].astype(str).str.lower() == metric.lower()]
        return subset[["dataset", "method", "value"]]
    raise ValueError(
        "Input CSV must be wide format with columns dataset,method,<metric> "
        "or long format with columns dataset,metric,method,value."
    )


def _holm_bonferroni(pvalues: list[float]) -> list[float]:
    m = len(pvalues)
    order = sorted(range(m), key=lambda idx: pvalues[idx])
    adjusted = [1.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        raw = min(1.0, (m - rank) * pvalues[idx])
        running_max = max(running_max, raw)
        adjusted[idx] = running_max
    return adjusted


def _rank_biserial_from_diffs(diffs: np.ndarray) -> float:
    nonzero = diffs[diffs != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = pd.Series(np.abs(nonzero)).rank(method="average").to_numpy()
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    total = ranks.sum()
    return float((positive - negative) / total) if total else 0.0


def compute_tests(csv_path: Path, ours: str, metric: str) -> pd.DataFrame:
    wilcoxon = _require_scipy()
    metric_frame = _load_metric_frame(csv_path, metric)
    pivot = metric_frame.pivot_table(index="dataset", columns="method", values="value", aggfunc="mean")
    if ours not in pivot.columns:
        raise ValueError(f"Ours method '{ours}' was not found. Available methods: {list(pivot.columns)}")

    rows: list[dict[str, float | int | str]] = []
    for method in pivot.columns:
        if method == ours:
            continue
        paired = pivot[[ours, method]].dropna()
        if paired.empty:
            continue
        diffs = (paired[ours] - paired[method]).to_numpy(dtype=float)
        if np.allclose(diffs, 0):
            statistic = 0.0
            pvalue = 1.0
        else:
            statistic, pvalue = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        rows.append(
            {
                "baseline": method,
                "metric": metric,
                "n_datasets": int(len(diffs)),
                "mean_difference": float(np.mean(diffs)),
                "median_difference": float(np.median(diffs)),
                "std_difference": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
                "wins": int(np.sum(diffs > 0)),
                "ties": int(np.sum(np.isclose(diffs, 0))),
                "losses": int(np.sum(diffs < 0)),
                "wilcoxon_statistic": float(statistic),
                "p_value": float(pvalue),
                "rank_biserial": _rank_biserial_from_diffs(diffs),
            }
        )
    output = pd.DataFrame(rows)
    if not output.empty:
        output["holm_bonferroni_p"] = _holm_bonferroni(output["p_value"].tolist())
        output = output.sort_values("mean_difference", ascending=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute paired Wilcoxon tests from per-dataset result CSVs.")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--ours", default="Ours (Full)")
    parser.add_argument("--metric", default="dice")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    output = compute_tests(args.csv, ours=args.ours, metric=args.metric)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    md_path = args.out.with_suffix(".md")
    markdown = _to_markdown(output)
    md_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Wrote {args.out}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
