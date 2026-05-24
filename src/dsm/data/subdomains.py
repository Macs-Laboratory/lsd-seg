from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from ..utils.decorators import logged_call


@dataclass(slots=True)
class SubdomainAssignments:
    sample_ids: list[str]
    labels: list[str]

    def save(self, path: str | Path) -> None:
        df = pd.DataFrame({"sample_id": self.sample_ids, "subdomain_id": self.labels})
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)


class MetadataSubdomainStrategy:
    @logged_call()
    def build(self, records: list[dict], metadata_key: str) -> SubdomainAssignments:
        return SubdomainAssignments(
            sample_ids=[record["sample_id"] for record in records],
            labels=[record.get("metadata", {}).get(metadata_key, "unknown") for record in records],
        )


class FrozenFeatureClusterStrategy:
    """
    Uses externally cached frozen features to prevent circular evaluation:
    clustering is computed once on the training split before model fitting.
    """

    @logged_call()
    def build(self, records: list[dict], feature_matrix: np.ndarray, num_clusters: int) -> SubdomainAssignments:
        labels = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto").fit_predict(feature_matrix)
        return SubdomainAssignments(
            sample_ids=[record["sample_id"] for record in records],
            labels=[f"cluster_{label}" for label in labels.tolist()],
        )
