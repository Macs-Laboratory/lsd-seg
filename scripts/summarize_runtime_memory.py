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


def _infer_method_dataset(path: Path) -> tuple[str, str]:
    parts = path.parts
    method = path.parent.name
    dataset = path.parent.parent.name if len(parts) >= 2 else "unknown"
    return method, dataset


def _load_runtime_rows(path: Path) -> pd.DataFrame | None:
    frame = pd.read_csv(path)
    if "runtime_seconds" not in frame.columns and "peak_gpu_memory_mb" not in frame.columns:
        return None
    method, dataset = _infer_method_dataset(path)
    if "method" not in frame.columns:
        frame["method"] = method
    if "dataset" not in frame.columns:
        frame["dataset"] = dataset
    if "runtime_seconds" not in frame.columns:
        frame["runtime_seconds"] = pd.NA
    if "peak_gpu_memory_mb" not in frame.columns:
        frame["peak_gpu_memory_mb"] = pd.NA
    return frame[["method", "dataset", "runtime_seconds", "peak_gpu_memory_mb"]]


def summarize(results_root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(results_root.rglob("*.csv")):
        if path.name not in {"summary_metrics.csv", "per_sample_metrics.csv"}:
            continue
        loaded = _load_runtime_rows(path)
        if loaded is not None:
            frames.append(loaded)
    if not frames:
        raise RuntimeError(
            "No runtime_seconds or peak_gpu_memory_mb columns found. "
            "Re-run evaluation with evaluation.compute_runtime=true and evaluation.compute_memory=true."
        )
    combined = pd.concat(frames, ignore_index=True)
    grouped = combined.groupby(["method", "dataset"], dropna=False)
    output = grouped.agg(
        mean_runtime_ms_per_image=("runtime_seconds", lambda x: float(pd.to_numeric(x, errors="coerce").mean() * 1000.0)),
        std_runtime_ms_per_image=("runtime_seconds", lambda x: float(pd.to_numeric(x, errors="coerce").std(ddof=1) * 1000.0)),
        peak_gpu_memory_mb=("peak_gpu_memory_mb", lambda x: float(pd.to_numeric(x, errors="coerce").max())),
        n_images=("runtime_seconds", "count"),
    ).reset_index()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize runtime and peak GPU memory from evaluation CSVs.")
    parser.add_argument("--results-root", default="results", type=Path)
    parser.add_argument("--out", default="results/runtime_memory_summary.csv", type=Path)
    args = parser.parse_args()

    output = summarize(args.results_root)
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
