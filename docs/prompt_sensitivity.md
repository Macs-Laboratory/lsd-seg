# Prompt Sensitivity

Reviewer concern: poor automatic prompts may corrupt the descriptor, collapse routing, or assign samples to incorrect prototypes.

This document defines the prompt-sensitivity protocol. It does not report prompt-sensitivity numbers unless generated from actual perturbed evaluation outputs.

The descriptor `r` uses global pooled features and prompt-aligned ROI cues. Prompt perturbation evaluates whether routing degrades gradually when prompts become noisy, or whether assignments collapse abruptly.

## Perturbation Modes

- box jitter: 0%, 5%, 10%, 20%, 30% of box size,
- point jitter: 0 px, 4 px, 8 px, 16 px,
- coarse mask degradation: erosion/dilation radius 0, 2, 4, 8,
- coarse mask dropout: 0%, 10%, 25%,
- missing prompt prior or global-only descriptor when supported by config.

## Metrics

- Dice,
- Boundary-F1,
- routing entropy,
- assigned prototype change rate,
- mean max prototype similarity,
- SAM prior fallback rate when native SAM/SAM2 prior is enabled.

## Command

Required input schema for computed summaries:

```text
perturbation,level,dice,boundary_f1,routing_entropy,assigned_prototype_change_rate,mean_max_prototype_similarity,sam_decoder_fallback_rate
```

Template:

```bash
uv run python scripts/run_prompt_sensitivity.py \
  --perturbation box_jitter \
  --levels 0,0.05,0.10,0.20,0.30 \
  --out results/prompt_sensitivity_template.csv
```

Aggregate actual results:

```bash
uv run python scripts/run_prompt_sensitivity.py \
  --input-csv results/prompt_sensitivity_raw.csv \
  --out results/prompt_sensitivity_summary.csv
```

Report generator:

```bash
uv run python scripts/prompt_sensitivity.py \
  --input-csv results/prompt_sensitivity_raw.csv \
  --output reports/prompt_sensitivity.json
```

Template-only report:

```bash
uv run python scripts/prompt_sensitivity.py \
  --output reports/prompt_sensitivity.json \
  --template-only
```

Robust behavior should show gradual degradation and no abrupt assignment collapse. A failure mode would be large assignment changes, lower max similarity, and increased entropy under severe prompt corruption.

The script can also aggregate an existing sensitivity CSV. If raw perturbed evaluation outputs are unavailable, it writes a clearly labeled template instead of fake results.
