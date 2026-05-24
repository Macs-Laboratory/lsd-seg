from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ExternalBaselineSpec:
    name: str
    repository: str
    entrypoint: str
    checkpoint_env: str | None = None
    notes: list[str] = field(default_factory=list)


class ExternalBaselineAdapter:
    def __init__(self, spec: ExternalBaselineSpec) -> None:
        self.spec = spec

    def command(self, dataset_manifest: str, output_dir: str) -> str:
        checkpoint = f"${self.spec.checkpoint_env}" if self.spec.checkpoint_env else ""
        return (
            f"python {self.spec.entrypoint} "
            f"--manifest {dataset_manifest} --output-dir {output_dir} "
            f"{f'--checkpoint {checkpoint}' if checkpoint else ''}".strip()
        )


PAPER_BASELINES: dict[str, ExternalBaselineSpec] = {
    "nnunet_2d": ExternalBaselineSpec(
        name="nnU-Net (2D)",
        repository="https://github.com/MIC-DKFZ/nnUNet",
        entrypoint="external/nnunet/run_baseline.py",
        checkpoint_env="NNUNET_CHECKPOINT",
    ),
    "mednext_2d": ExternalBaselineSpec(
        name="MedNeXt (2D)",
        repository="https://github.com/MIC-DKFZ/MedNeXt",
        entrypoint="external/mednext/run_baseline.py",
        checkpoint_env="MEDNEXT_CHECKPOINT",
    ),
    "vm_unet": ExternalBaselineSpec(
        name="VM-UNet",
        repository="https://github.com/JCruan519/VM-UNet",
        entrypoint="external/vmunet/run_baseline.py",
    ),
    "u_mamba_2d": ExternalBaselineSpec(
        name="U-Mamba (2D)",
        repository="https://github.com/bowang-lab/U-Mamba",
        entrypoint="external/umamba/run_baseline.py",
    ),
    "segmamba_2d": ExternalBaselineSpec(
        name="SegMamba (slice-wise 2D)",
        repository="https://github.com/ge-xing/SegMamba",
        entrypoint="external/segmamba/run_baseline.py",
    ),
    "sam2_apg": ExternalBaselineSpec(
        name="SAM2 + APG",
        repository="https://github.com/facebookresearch/segment-anything-2",
        entrypoint="external/sam2/run_promptable_baseline.py",
    ),
    "medsam_apg": ExternalBaselineSpec(
        name="MedSAM + APG",
        repository="https://github.com/bowang-lab/MedSAM",
        entrypoint="external/medsam/run_promptable_baseline.py",
    ),
    "sam_med2d_apg": ExternalBaselineSpec(
        name="SAM-Med2D + APG",
        repository="https://github.com/OpenGVLab/SAM-Med2D",
        entrypoint="external/sam_med2d/run_promptable_baseline.py",
    ),
    "medsam2_apg": ExternalBaselineSpec(
        name="MedSAM2 + APG",
        repository="https://github.com/bowang-lab/MedSAM2",
        entrypoint="external/medsam2/run_promptable_baseline.py",
    ),
}
