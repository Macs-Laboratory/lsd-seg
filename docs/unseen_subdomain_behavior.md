# Test-time Unseen Sub-domain Behavior

Reviewer concern: prototypes are fixed during inference, so a genuinely unseen mode cannot instantiate a new prototype.

LSD-Seg does not create new prototypes during inference. Evaluation is parameter-free: prototype memory is fixed, no gradient updates are performed, and no test-time expert is added.

Expected behavior for a genuinely new or out-of-support subdomain:

- lower max prototype similarity,
- higher routing entropy,
- smoother uncertainty-tempered mixture,
- higher uncertainty map values,
- possible degradation if the mode is far outside training support.

This is a known limitation and a deployment diagnostic, not a hidden adaptation mechanism.

## Leave-one-subdomain-out Protocol

```bash
uv run python scripts/analyze_unseen_subdomain.py \
  --config configs/default.yaml \
  --dataset brats \
  --heldout-subdomain center_3 \
  --out results/unseen_subdomain/brats_center_3
```

Recommended reporting:

| Held-out subdomain | Dice | Boundary-F1 | Max similarity | Routing entropy | Uncertainty |
| --- | ---: | ---: | ---: | ---: | ---: |
| To be generated | To be generated | To be generated | To be generated | To be generated | To be generated |

If metadata is unavailable, pseudo-subdomain holdout may be used, but it must be labeled as pseudo-subdomain analysis.

