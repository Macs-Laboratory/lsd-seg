from __future__ import annotations

from .base import ExperimentSpec
from .baselines import (
    MEDNEXT_2D,
    MEDSAM2_APG,
    MEDSAM_APG,
    NNUNET_2D,
    SAM2_APG,
    SAM_MED2D_APG,
    SEGMAMBA_2D,
    U_MAMBA_2D,
    VM_UNET,
)
from .ours import (
    OURS_FIXED_K,
    OURS_FULL,
    OURS_GLOBAL_ONLY_DESCRIPTOR,
    OURS_NO_BALANCE,
    OURS_NO_EXPERT_LOSS,
    OURS_NO_HYPERNETWORK,
    OURS_NO_ORTHOGONALITY,
    OURS_NO_PROTOTYPE_MERGE,
    OURS_NO_UNCERTAINTY_TEMPERING,
)


PAPER_EXPERIMENT_SPECS: dict[str, ExperimentSpec] = {
    spec.name: spec
    for spec in [
        OURS_FULL,
        OURS_FIXED_K,
        OURS_NO_HYPERNETWORK,
        OURS_NO_UNCERTAINTY_TEMPERING,
        OURS_GLOBAL_ONLY_DESCRIPTOR,
        OURS_NO_PROTOTYPE_MERGE,
        OURS_NO_EXPERT_LOSS,
        OURS_NO_ORTHOGONALITY,
        OURS_NO_BALANCE,
        NNUNET_2D,
        MEDNEXT_2D,
        VM_UNET,
        U_MAMBA_2D,
        SEGMAMBA_2D,
        SAM2_APG,
        MEDSAM_APG,
        SAM_MED2D_APG,
        MEDSAM2_APG,
    ]
}
