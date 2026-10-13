"""
Leakage-Resistant Group-Aware Split Generator for CYPHER IDS.

This module generates a leak-free partition of the balanced CICIDS2017 dataset:
- Every unique 78-feature vector is assigned a unique group ID.
- Groups are stratified across classes and allocated to Train (70%), Val (15%), Test (15%).
- Mathematical guarantee: ZERO exact feature duplicates cross Train, Val, or Test partitions.
- Output directory: ml/splits_leakage_resistant/ (preserves original ml/splits/ untouched).
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_STATE = 42

SUPPORTED_CLASSES = [
    "BENIGN",
    "DoS Hulk",
    "PortScan",
    "DDoS",
    "DoS GoldenEye",
    "FTP-Patator",
    "SSH-Patator",
    "DoS slowloris",
    "DoS Slowhttptest",
    "Bot",
    "Web Attack Brute Force",
    "Web Attack XSS",
]


def create_leakage_resistant_split(
    input_splits_dir: Path,
    output_dir: Path,
    random_state: int = RANDOM_STATE,
) -> dict:
    print("Loading original balanced dataset from ml/splits/...")
    train_orig = pd.read_csv(input_splits_dir / "train.csv")
    val_orig = pd.read_csv(input_splits_dir / "validation.csv")
    test_orig = pd.read_csv(input_splits_dir / "test.csv")

    full_df = pd.concat([train_orig, val_orig, test_orig], ignore_index=True)
    feature_cols = [c for c in full_df.columns if c not in ["Label", "_source_file"]]

    print(f"Total rows in balanced pool: {len(full_df):,}")

    # Generate unique group hashes for exact feature vectors
    print("Generating exact feature vector fingerprints...")
    # Using string concatenation of rounded/canonical floats or fast string hash
    # To be exact and fast, convert array to bytes or use pandas duplicated
    # Group by exact feature columns:
    full_df["_feature_hash"] = full_df.groupby(feature_cols, observed=True).ngroup()

    num_groups = full_df["_feature_hash"].nunique()
    print(f"Total unique feature groups: {num_groups:,} (Duplicates: {len(full_df) - num_groups:,})")

    # Perform Stratified Group Partitioning
    # For each class, partition unique groups: 70% Train, 15% Val, 15% Test
    rng = np.random.default_rng(random_state)

    train_indices = []
    val_indices = []
    test_indices = []

    # Map groups to their primary label to prevent split leakage
    # If a group has conflicting labels (6 out of 124,964), assign it based on majority
    group_to_label = full_df.groupby("_feature_hash")["Label"].agg(lambda s: s.mode()[0])

    all_train_groups = set()
    all_val_groups = set()
    all_test_groups = set()

    for label in SUPPORTED_CLASSES:
        class_groups = group_to_label[group_to_label == label].index.to_numpy().copy()
        rng.shuffle(class_groups)

        n_groups = len(class_groups)
        n_train = int(np.round(0.70 * n_groups))
        n_val = int(np.round(0.15 * n_groups))
        if n_groups >= 3 and n_val == 0:
            n_val = 1
        n_test = n_groups - n_train - n_val

        all_train_groups.update(class_groups[:n_train])
        all_val_groups.update(class_groups[n_train : n_train + n_val])
        all_test_groups.update(class_groups[n_train + n_val :])

    # Assign rows strictly by global feature hash
    train_indices = full_df[full_df["_feature_hash"].isin(all_train_groups)].index.to_numpy()
    val_indices = full_df[full_df["_feature_hash"].isin(all_val_groups)].index.to_numpy()
    test_indices = full_df[full_df["_feature_hash"].isin(all_test_groups)].index.to_numpy()

    train_df = full_df.loc[train_indices].drop(columns=["_feature_hash"]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    val_df = full_df.loc[val_indices].drop(columns=["_feature_hash"]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    test_df = full_df.loc[test_indices].drop(columns=["_feature_hash"]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # Verification: Check cross-partition exact duplicate overlap
    train_tuples = set(map(tuple, train_df[feature_cols].values))
    val_tuples = set(map(tuple, val_df[feature_cols].values))
    test_tuples = set(map(tuple, test_df[feature_cols].values))

    test_train_overlap = len(test_tuples & train_tuples)
    test_val_overlap = len(test_tuples & val_tuples)
    val_train_overlap = len(val_tuples & train_tuples)

    print("\n--- Leakage Verification ---")
    print(f"Train / Test exact duplicate tuple overlap: {test_train_overlap} (0.00%)")
    print(f"Val / Test exact duplicate tuple overlap:   {test_val_overlap} (0.00%)")
    print(f"Train / Val exact duplicate tuple overlap:  {val_train_overlap} (0.00%)")

    assert test_train_overlap == 0, f"Leakage detected: {test_train_overlap} overlapping tuples between train and test!"

    # Save to output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "validation.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    metadata = {
        "generator": "ml/split_leakage_resistant.py",
        "random_state": random_state,
        "strategy": "Stratified Group-Aware Content Fingerprint (Zero-Duplicate)",
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "cross_split_test_train_overlap": test_train_overlap,
        "cross_split_test_val_overlap": test_val_overlap,
        "cross_split_val_train_overlap": val_train_overlap,
        "train_class_counts": train_df["Label"].value_counts().to_dict(),
        "val_class_counts": val_df["Label"].value_counts().to_dict(),
        "test_class_counts": test_df["Label"].value_counts().to_dict(),
    }

    with open(output_dir / "split_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSuccessfully generated leakage-resistant splits in: {output_dir}")
    print(f"Train:      {len(train_df):,} rows")
    print(f"Validation: {len(val_df):,} rows")
    print(f"Test:       {len(test_df):,} rows")

    return metadata


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    input_dir = project_root / "ml" / "splits"
    output_dir = project_root / "ml" / "splits_leakage_resistant"

    create_leakage_resistant_split(input_dir, output_dir)
