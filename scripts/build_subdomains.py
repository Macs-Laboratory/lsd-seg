from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dsm.data.subdomains import FrozenFeatureClusterStrategy, MetadataSubdomainStrategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute sub-domain assignments for reproducible evaluation.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strategy", choices=["metadata", "cluster"], required=True)
    parser.add_argument("--metadata-key")
    parser.add_argument("--features")
    parser.add_argument("--num-clusters", type=int, default=5)
    args = parser.parse_args()

    records = json.loads(Path(args.manifest).read_text())
    train_records = [record for record in records if record["split"] == "train"]
    if args.strategy == "metadata":
        if not args.metadata_key:
            raise ValueError("--metadata-key is required for metadata strategy.")
        assignments = MetadataSubdomainStrategy().build(train_records, args.metadata_key)
    else:
        if not args.features:
            raise ValueError("--features is required for cluster strategy.")
        assignments = FrozenFeatureClusterStrategy().build(
            train_records,
            feature_matrix=np.load(args.features),
            num_clusters=args.num_clusters,
        )
    assignments.save(args.output)


if __name__ == "__main__":
    main()
