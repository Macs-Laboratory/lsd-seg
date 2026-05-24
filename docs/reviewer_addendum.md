# Reviewer-driven Supplementary Evidence

This document collects additional implementation details, analysis protocols, and reproducibility notes that could not fit into the main MICCAI manuscript. It is organized around the main reviewer concerns: statistical testing, runtime/memory, prompt robustness, hyperparameter protocol, unseen sub-domain behavior, and disentangling latent sub-domain discovery from generic MoE capacity.

The repository supplement is intentionally evidence-oriented. Paper-reported aggregate values are listed exactly where available. Missing per-dataset values, runtime/memory measurements, p-values, or sensitivity results must be regenerated from result CSVs with the scripts in this repository.

## Reviewer Concern Map

| Reviewer concern | GitHub response |
| --- | --- |
| Per-dataset breakdown and Wilcoxon tests | `scripts/compute_statistical_tests.py`, `docs/statistical_testing.md` |
| Runtime and peak memory missing | `scripts/summarize_runtime_memory.py`, `docs/runtime_memory.md` |
| Prompt sensitivity | `scripts/run_prompt_sensitivity.py`, `docs/prompt_sensitivity.md` |
| tau, rho, alpha tuning protocol | `docs/hyperparameter_protocol.md` |
| Test-time new subdomains | `docs/unseen_subdomain_behavior.md` |
| Sub-domain discovery vs generic MoE | `scripts/analyze_subdomain_capacity.py`, `docs/subdomain_vs_capacity.md` |
| Pseudo-subdomain circularity | `docs/reproducibility.md`, pseudo-subdomain protocol |
| Architecture under-specified | README method-to-code map plus implementation files under `src/dsm/models/` |
| 2D limitation | `docs/limitations.md` |

## Repository Supplement Index

| Topic | Document |
| --- | --- |
| Official aggregate table | `docs/extended_results.md` |
| Statistical testing protocol | `docs/statistical_testing.md` |
| Runtime and memory reporting | `docs/runtime_memory.md` |
| Prompt sensitivity protocol | `docs/prompt_sensitivity.md` |
| Hyperparameter protocol | `docs/hyperparameter_protocol.md` |
| Unseen sub-domain behavior | `docs/unseen_subdomain_behavior.md` |
| Sub-domain discovery vs capacity | `docs/subdomain_vs_capacity.md` |
| Reproducibility and circularity precautions | `docs/reproducibility.md` |
| Known limitations | `docs/limitations.md` |

## Scope

The main manuscript reports LSD-Seg over 20 public datasets spanning endoscopy, dermoscopy, fundus, histopathology, ultrasound, MRI, CT, and PET/CT. This supplement does not introduce new unsupported claims. It provides:

- exact paper-reported aggregate values,
- scripts to compute additional reviewer-requested statistics from CSVs,
- templates for missing per-dataset or runtime tables,
- protocol notes for future reruns and auditability.

## Data Availability and Generated Tables

Reviewer-facing scripts are designed to regenerate tables from evaluation CSVs. When CSVs are unavailable, scripts create templates with missing values rather than fabricated numbers. Paper aggregate values are the only manually transcribed values and are explicitly labeled as coming from Table 1 of the camera-ready manuscript.
