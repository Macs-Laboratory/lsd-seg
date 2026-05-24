# Extended Results

This page records paper-reported aggregate results and explains how to regenerate reviewer-facing per-dataset tables from result CSVs. Do not manually edit computed p-values, win/loss counts, runtime, or per-dataset values into this file unless they come from the provided scripts.

## Official Aggregate Results

Main results averaged over 20 datasets, mean ± 95% CI.

These values are transcribed from the camera-ready manuscript Table 1. If the manuscript table is updated, this table must be updated in the same commit.

| Method | Dice ↑ | HD95 ↓ | Boundary-F1 ↑ | Worst Dice ↑ |
| --- | ---: | ---: | ---: | ---: |
| nnU-Net (2D) | 80.4 ± 1.7 | 12.9 ± 0.3 | 74.8 ± 1.7 | 75.3 ± 6.5 |
| MedNeXt (2D) | 82.3 ± 1.8 | 12.7 ± 0.3 | 76.7 ± 1.8 | 77.3 ± 1.3 |
| VM-UNet | 81.3 ± 1.8 | 12.8 ± 0.3 | 75.7 ± 1.8 | 75.8 ± 1.2 |
| U-Mamba (2D) | 82.4 ± 1.9 | 12.6 ± 0.3 | 76.8 ± 1.9 | 75.9 ± 3.8 |
| SegMamba (slice-wise 2D) | 84.0 ± 1.8 | 12.4 ± 0.3 | 78.4 ± 1.8 | 78.6 ± 4.0 |
| SAM2 + APG | 82.5 ± 1.8 | 12.6 ± 0.3 | 76.9 ± 1.8 | 76.4 ± 9.1 |
| MedSAM + APG | 84.2 ± 1.6 | 12.4 ± 0.2 | 78.6 ± 1.6 | 78.2 ± 1.7 |
| SAM-Med2D + APG | 84.4 ± 2.0 | 12.3 ± 0.3 | 78.8 ± 2.0 | 78.6 ± 8.9 |
| MedSAM2 + APG | 84.8 ± 1.8 | 12.3 ± 0.3 | 79.2 ± 1.8 | 78.8 ± 9.2 |
| Ours (Fixed-K experts) | 84.6 ± 1.8 | 12.3 ± 0.3 | 79.0 ± 1.8 | 78.9 ± 8.4 |
| Ours (w/o hypernetwork; static adapters) | 82.8 ± 1.8 | 12.6 ± 0.3 | 77.2 ± 1.8 | 76.4 ± 9.3 |
| Ours (w/o uncertainty-tempered routing) | 83.1 ± 1.8 | 12.5 ± 0.3 | 77.5 ± 1.8 | 77.6 ± 6.0 |
| Ours (global-only descriptor; w/o ROI) | 83.9 ± 1.8 | 12.4 ± 0.3 | 78.3 ± 1.8 | 77.5 ± 5.0 |
| Ours (Full) | 87.9 ± 1.7 | 11.8 ± 0.3 | 82.3 ± 1.7 | 82.2 ± 7.3 |

## Interpretation

Ours Full achieves the best Dice, HD95, Boundary-F1, and Worst Dice in the reported aggregate table.

- Ours Full improves Dice by +3.9 over SegMamba, the strongest non-foundation baseline in the reported table.
- Ours Full improves Dice by +3.1 over MedSAM2 + APG.
- Ours Full improves Boundary-F1 to 82.3.
- Ours Full improves Worst Dice to 82.2.

Boundary-F1 and Worst Dice are emphasized because the method targets boundary reliability and hidden sub-domain robustness, not only mean overlap.

## Per-dataset Results

Per-dataset values should be regenerated from evaluation outputs rather than copied by hand.

```bash
uv run python scripts/make_reviewer_tables.py \
  --per-dataset-csv results/per_dataset_results.csv \
  --out-dir results/reviewer_tables
```

If `results/per_dataset_results.csv` is absent, run the evaluation pipeline first and then run `scripts/make_reviewer_tables.py` to populate this section. The expected CSV schema is:

```text
dataset,method,dice,hd95,boundary_f1,worst_dice
```

## Win/loss Heatmap

Figure 2(b) reports Ours minus Baseline Dice across 20 datasets and 13 comparison methods. Exact per-dataset win/tie/loss counts should be regenerated from CSV rather than manually edited.

```bash
uv run python scripts/make_reviewer_tables.py \
  --per-dataset-csv results/per_dataset_results.csv \
  --out-dir results/reviewer_tables
```
