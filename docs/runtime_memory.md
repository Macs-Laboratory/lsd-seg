# Runtime and Peak Memory

Reviewer concern: runtime and peak GPU memory were mentioned as reported quantities, but the page-limited manuscript did not include the detailed table.

Runtime is measured as forward-pass wall time per image when available in evaluator outputs. Peak memory is measured by the evaluator when GPU memory statistics are available.

Runtime and memory depend strongly on encoder mode. Report them separately for:

- `resnet` fallback,
- `sam2 + hybrid`,
- ablations when available.

## Reproduce Summary Table

```bash
uv run python scripts/summarize_runtime_memory.py \
  --results-root results \
  --out results/runtime_memory_summary.csv
```

The script writes:

- `results/runtime_memory_summary.csv`
- `results/runtime_memory_summary.md`

Expected evaluator columns:

```text
runtime_seconds
peak_gpu_memory_mb
```

If no `runtime_seconds` or `peak_gpu_memory_mb` columns are found, re-run evaluation with:

```yaml
evaluation:
  compute_runtime: true
  compute_memory: true
```

## Table Placeholder

The table below is intentionally a placeholder until `scripts/summarize_runtime_memory.py` is run on evaluator outputs. The repository does not include fabricated runtime/memory numbers.

| Method | Dataset | Runtime (ms/image) | Peak GPU memory (MB) | N images |
| --- | --- | ---: | ---: | ---: |

The paper focuses on segmentation robustness, but deployment constraints motivated reporting runtime and peak memory. The repository provides scripts to regenerate these numbers from evaluation logs without inventing values.
