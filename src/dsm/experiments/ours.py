from __future__ import annotations

from .base import ExperimentSpec


OURS_FULL = ExperimentSpec(
    name="ours_full",
    kind="native",
    description="Full dynamic sub-domain model from the paper.",
)

OURS_FIXED_K = ExperimentSpec(
    name="ours_fixed_k",
    kind="native",
    description="Fixed-K expert ablation.",
    overrides={"model": {"fixed_k": 5}},
)

OURS_NO_HYPERNETWORK = ExperimentSpec(
    name="ours_no_hypernetwork",
    kind="native",
    description="Static-adapter ablation.",
    overrides={"model": {"hypernetwork_enabled": False}},
)

OURS_NO_UNCERTAINTY_TEMPERING = ExperimentSpec(
    name="ours_no_uncertainty_tempering",
    kind="native",
    description="Routing-temperature ablation.",
    overrides={"model": {"uncertainty_tempering_enabled": False}},
)

OURS_GLOBAL_ONLY_DESCRIPTOR = ExperimentSpec(
    name="ours_global_only_descriptor",
    kind="native",
    description="Descriptor ablation without ROI-conditioned prompt features.",
    overrides={"model": {"use_roi_descriptor": False}},
)

OURS_NO_PROTOTYPE_MERGE = ExperimentSpec(
    name="ours_no_prototype_merge",
    kind="native",
    description="Prototype merge ablation.",
    overrides={"model": {"merge_enabled": False}},
)

OURS_NO_EXPERT_LOSS = ExperimentSpec(
    name="ours_no_expert_loss",
    kind="native",
    description="No expert auxiliary loss.",
    overrides={"loss": {"expert_weight": 0.0, "lambda_expert": 0.0}},
)

OURS_NO_ORTHOGONALITY = ExperimentSpec(
    name="ours_no_orthogonality",
    kind="native",
    description="No prototype orthogonality regularization.",
    overrides={"loss": {"ortho_weight": 0.0, "lambda_ortho": 0.0}},
)

OURS_NO_BALANCE = ExperimentSpec(
    name="ours_no_balance",
    kind="native",
    description="No routing balance regularization.",
    overrides={"loss": {"balance_weight": 0.0, "lambda_balance": 0.0}},
)
