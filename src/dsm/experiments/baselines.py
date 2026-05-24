from __future__ import annotations

from .base import ExperimentSpec


NNUNET_2D = ExperimentSpec(name="nnunet_2d", kind="external", description="nnU-Net (2D)")
MEDNEXT_2D = ExperimentSpec(name="mednext_2d", kind="external", description="MedNeXt (2D)")
VM_UNET = ExperimentSpec(name="vm_unet", kind="external", description="VM-UNet")
U_MAMBA_2D = ExperimentSpec(name="u_mamba_2d", kind="external", description="U-Mamba (2D)")
SEGMAMBA_2D = ExperimentSpec(name="segmamba_2d", kind="external", description="SegMamba (slice-wise 2D)")
SAM2_APG = ExperimentSpec(name="sam2_apg", kind="external", description="SAM2 + APG")
MEDSAM_APG = ExperimentSpec(name="medsam_apg", kind="external", description="MedSAM + APG")
SAM_MED2D_APG = ExperimentSpec(name="sam_med2d_apg", kind="external", description="SAM-Med2D + APG")
MEDSAM2_APG = ExperimentSpec(name="medsam2_apg", kind="external", description="MedSAM2 + APG")
