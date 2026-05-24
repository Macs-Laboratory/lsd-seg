# Sub-domain Discovery vs Generic Capacity

Reviewer concern: whether gains come from meaningful latent sub-domain modeling or simply from extra MoE capacity.

## Evidence Plan

- Fixed-K experts test whether dynamic discovery matters.
- Static adapters test whether hypernetwork-conditioned expert fields matter.
- Global-only descriptor tests whether prompt-aligned local descriptor matters.
- Random or uniform routing controls for generic ensemble capacity.
- Metadata alignment metrics, when available, test whether assignments correspond to external subdomain factors.
- Expert specialization score tests whether experts are functionally specialized.

## Paper Aggregate Ablations

| Variant | Dice | Boundary-F1 | Worst Dice |
| --- | ---: | ---: | ---: |
| Fixed-K experts | 84.6 | 79.0 | 78.9 |
| w/o hypernetwork; static adapters | 82.8 | 77.2 | 76.4 |
| w/o uncertainty-tempered routing | 83.1 | 77.5 | 77.6 |
| global-only descriptor; w/o ROI | 83.9 | 78.3 | 77.5 |
| Ours Full | 87.9 | 82.3 | 82.2 |

Interpretation from paper-reported aggregate values:

- Full model exceeds Fixed-K by +3.3 Dice.
- Full model exceeds static adapters by +5.1 Dice.
- Full model exceeds no uncertainty routing by +4.8 Dice.
- Full model exceeds global-only descriptor by +4.0 Dice.

These gaps argue against a pure capacity-only explanation, but further metadata-alignment and random-routing controls should be computed with the repository script before claiming stronger conclusions.

## Command

```bash
uv run python scripts/analyze_subdomain_capacity.py \
  --config configs/default.yaml \
  --dataset brats \
  --metadata-key center \
  --out results/subdomain_capacity/brats
```

If metadata labels are unavailable, report pseudo-subdomain metrics separately and mark them as descriptor-clustered pseudo-labels rather than external ground truth.

