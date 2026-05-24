from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SampleRecord:
    sample_id: str
    image_path: str
    mask_path: str
    split: str
    patient_id: str | None = None
    subdomain_id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
