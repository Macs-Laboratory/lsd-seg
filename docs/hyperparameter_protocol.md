# Hyperparameter Protocol

Reviewer concern: whether tau, rho, and alpha were fixed across datasets or tuned separately.

The main protocol uses fixed defaults exposed in `configs/default.yaml` and `configs/sam2.example.yaml`. Dataset-specific tuning should not be required for the main reported setting unless explicitly stated in an experiment config.

| Parameter | Meaning | Default |
| --- | --- | ---: |
| tau / `init_threshold` | spawn new prototype if max similarity is below threshold | 0.65 |
| rho / `merge_threshold` | merge prototypes above cosine similarity | 0.95 |
| T0 / `routing_tau` | base routing temperature | 0.07 |
| alpha / `uncertainty_alpha` | strength of uncertainty-tempered smoothing | 4.0 |
| eta / `ema_momentum` | prototype EMA update momentum | 0.95 |
| `max_prototypes` | upper bound on active prototypes | 12 |
| `min_support` | support threshold for prototype maintenance | 3 |
| `merge_interval` | merge frequency in training steps | 100 |

## Notation Note

Reviewer confusion can arise because tau is used in the paper for novelty while `routing_tau` controls softmax routing temperature in code. The code uses explicit names:

- `init_threshold`: novelty threshold for creating a prototype,
- `merge_threshold`: similarity threshold for merging prototypes,
- `routing_tau`: base softmax temperature for routing.

Figure 3(c) in the paper reports stable performance across a broad range of tau, rho, and alpha. Repository-side sensitivity analyses should be generated from CSVs or run logs rather than manually edited.

