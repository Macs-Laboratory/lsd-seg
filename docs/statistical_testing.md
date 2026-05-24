# Statistical Testing

Reviewer concern: per-dataset improvements should be accompanied by paired significance testing rather than only aggregate mean ± 95% CI.

## Protocol

Paired tests are appropriate because each method is evaluated on the same datasets. The primary test is a Wilcoxon signed-rank test on per-dataset metric differences:

```text
delta_i = metric_i(Ours Full) - metric_i(Baseline)
```

For Dice, Boundary-F1, and Worst Dice, positive deltas favor Ours. For HD95, lower is better, so interpret deltas accordingly or multiply by `-1` before testing if using a higher-is-better convention.

## Multiple Comparisons

The repository script applies Holm-Bonferroni correction across all baseline comparisons for the selected metric.

Reported quantities:

- number of paired datasets,
- mean difference,
- median difference,
- standard deviation,
- wins / ties / losses,
- Wilcoxon signed-rank p-value,
- Holm-Bonferroni adjusted p-value,
- rank-biserial effect size.

## Command

```bash
uv run python scripts/compute_statistical_tests.py \
  --csv results/per_dataset_results.csv \
  --ours "Ours (Full)" \
  --metric dice \
  --out results/statistical_tests_dice.csv
```

The manuscript table reports aggregate mean ± 95% CI. Formal statistical tests are computed from per-dataset paired results and are intentionally kept in the repository to avoid overcrowding the MICCAI page-limited manuscript.

