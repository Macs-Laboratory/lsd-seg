# Reproducibility Notes

This page records the repository-side protocol for reproducing LSD-Seg experiments and reviewer-facing supplementary analyses.

## Dataset Manifest

The training pipeline expects manifest records with image/mask paths, split information, sample IDs, and optional subdomain metadata.

```json
{
  "sample_id": "case001_slice000",
  "image_path": "/absolute/path/to/image.png",
  "mask_path": "/absolute/path/to/mask.png",
  "split": "train",
  "patient_id": "case001",
  "subdomain_id": "center_a",
  "metadata": {
    "center": "center_a"
  }
}
```

Use patient-level splits whenever possible to prevent leakage.

## Metadata vs Pseudo-subdomains

To avoid circular evaluation:

1. Use metadata subdomains when available.
2. If using pseudo-subdomains, fit clustering only on training descriptors.
3. Assign validation/test samples to the nearest training cluster center.
4. Do not refit clusters on test labels or test descriptors.
5. Report whether each analysis uses metadata or pseudo-subdomains.

Pseudo-clustered subdomains are diagnostic, not a replacement for external metadata.

## Fixed Defaults

The main protocol uses fixed defaults in `configs/default.yaml`: `init_threshold=0.65`, `merge_threshold=0.95`, `routing_tau=0.07`, `uncertainty_alpha=4.0`, `ema_momentum=0.95`, `max_prototypes=12`, `min_support=3`, and `merge_interval=100`.

## Command Sequence

```bash
uv run python scripts/smoke_test.py
uv run python scripts/train.py --config configs/default.yaml --experiment ours_full
uv run python scripts/evaluate.py --config configs/default.yaml --experiment ours_full
uv run python scripts/compute_statistical_tests.py --csv results/per_dataset_results.csv --ours "Ours (Full)" --metric dice --out results/statistical_tests_dice.csv
uv run python scripts/summarize_runtime_memory.py --results-root results --out results/runtime_memory_summary.csv
uv run python scripts/run_prompt_sensitivity.py --config configs/default.yaml --perturbation box_jitter --levels 0,0.05,0.10,0.20,0.30 --out results/prompt_sensitivity.csv
uv run python scripts/analyze_subdomain_capacity.py --config configs/default.yaml --dataset brats --metadata-key center --out results/subdomain_capacity/brats
```

For SAM2 experiments, run `scripts/inspect_sam2_features.py` first and verify `sam_native_prior_used_rate` after evaluation.

