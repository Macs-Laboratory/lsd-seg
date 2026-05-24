from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._\n"
    text_frame = frame.fillna("").astype(str)
    header = "| " + " | ".join(text_frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in text_frame.columns) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in text_frame.to_numpy()]
    return "\n".join([header, divider, *rows]) + "\n"


AGGREGATE_ROWS = [
    ("nnU-Net (2D)", "80.4 ± 1.7", "12.9 ± 0.3", "74.8 ± 1.7", "75.3 ± 6.5"),
    ("MedNeXt (2D)", "82.3 ± 1.8", "12.7 ± 0.3", "76.7 ± 1.8", "77.3 ± 1.3"),
    ("VM-UNet", "81.3 ± 1.8", "12.8 ± 0.3", "75.7 ± 1.8", "75.8 ± 1.2"),
    ("U-Mamba (2D)", "82.4 ± 1.9", "12.6 ± 0.3", "76.8 ± 1.9", "75.9 ± 3.8"),
    ("SegMamba (slice-wise 2D)", "84.0 ± 1.8", "12.4 ± 0.3", "78.4 ± 1.8", "78.6 ± 4.0"),
    ("SAM2 + APG", "82.5 ± 1.8", "12.6 ± 0.3", "76.9 ± 1.8", "76.4 ± 9.1"),
    ("MedSAM + APG", "84.2 ± 1.6", "12.4 ± 0.2", "78.6 ± 1.6", "78.2 ± 1.7"),
    ("SAM-Med2D + APG", "84.4 ± 2.0", "12.3 ± 0.3", "78.8 ± 2.0", "78.6 ± 8.9"),
    ("MedSAM2 + APG", "84.8 ± 1.8", "12.3 ± 0.3", "79.2 ± 1.8", "78.8 ± 9.2"),
    ("Ours (Fixed-K experts)", "84.6 ± 1.8", "12.3 ± 0.3", "79.0 ± 1.8", "78.9 ± 8.4"),
    ("Ours (w/o hypernetwork; static adapters)", "82.8 ± 1.8", "12.6 ± 0.3", "77.2 ± 1.8", "76.4 ± 9.3"),
    ("Ours (w/o uncertainty-tempered routing)", "83.1 ± 1.8", "12.5 ± 0.3", "77.5 ± 1.8", "77.6 ± 6.0"),
    ("Ours (global-only descriptor; w/o ROI)", "83.9 ± 1.8", "12.4 ± 0.3", "78.3 ± 1.8", "77.5 ± 5.0"),
    ("Ours (Full)", "87.9 ± 1.7", "11.8 ± 0.3", "82.3 ± 1.7", "82.2 ± 7.3"),
]


def _aggregate_table() -> pd.DataFrame:
    return pd.DataFrame(AGGREGATE_ROWS, columns=["Method", "Dice", "HD95", "Boundary-F1", "Worst Dice"])


def _write_per_dataset(csv_path: Path, out_dir: Path, ours: str) -> None:
    frame = pd.read_csv(csv_path)
    table_path = out_dir / "per_dataset_results.md"
    table_path.write_text(_to_markdown(frame), encoding="utf-8")
    if {"dataset", "method", "dice"}.issubset(frame.columns):
        pivot = frame.pivot_table(index="dataset", columns="method", values="dice", aggfunc="mean")
        if ours in pivot.columns:
            rows = []
            for method in pivot.columns:
                if method == ours:
                    continue
                diff = pivot[ours] - pivot[method]
                rows.append(
                    {
                        "baseline": method,
                        "wins": int((diff > 0).sum()),
                        "ties": int((diff == 0).sum()),
                        "losses": int((diff < 0).sum()),
                        "mean_delta_dice": float(diff.mean()),
                    }
                )
            win_loss = pd.DataFrame(rows)
            win_loss.to_csv(out_dir / "win_loss_counts.csv", index=False)
            (out_dir / "win_loss_counts.md").write_text(_to_markdown(win_loss), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reviewer-facing Markdown tables.")
    parser.add_argument("--per-dataset-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("results/reviewer_tables"))
    parser.add_argument("--ours", default="Ours (Full)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    aggregate = _aggregate_table()
    aggregate.to_csv(args.out_dir / "paper_aggregate_results.csv", index=False)
    markdown = _to_markdown(aggregate)
    (args.out_dir / "paper_aggregate_results.md").write_text(markdown, encoding="utf-8")
    print(markdown)

    if args.per_dataset_csv:
        _write_per_dataset(args.per_dataset_csv, args.out_dir, args.ours)
    else:
        template = pd.DataFrame(columns=["dataset", "method", "dice", "hd95", "boundary_f1", "worst_dice"])
        template.to_csv(args.out_dir / "per_dataset_results_template.csv", index=False)
        print("No per-dataset CSV provided. Wrote per_dataset_results_template.csv.")


if __name__ == "__main__":
    main()
