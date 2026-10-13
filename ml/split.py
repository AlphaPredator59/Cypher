from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from ml.data import load_all_data
    from ml.preprocess import preprocess
except ImportError:
    from data import load_all_data
    from preprocess import preprocess


RANDOM_STATE = 42

# Classes with enough data for meaningful initial multiclass training.
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

SAMPLES_PER_CLASS = 25_000


def build_balanced_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Create a capped, approximately balanced multiclass dataset."""

    df = df[df["Label"].isin(SUPPORTED_CLASSES)].copy()

    parts = []

    for label in SUPPORTED_CLASSES:
        class_df = df[df["Label"] == label]

        n = min(SAMPLES_PER_CLASS, len(class_df))

        sampled = class_df.sample(
            n=n,
            random_state=RANDOM_STATE,
        )

        parts.append(sampled)

        print(
            f"{label:<30} "
            f"available={len(class_df):>8,} "
            f"selected={n:>8,}"
        )

    balanced = pd.concat(
        parts,
        ignore_index=True,
    )

    return balanced.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)


def main() -> None:
    print("Loading dataset...")
    df = preprocess(load_all_data())

    print("\nBuilding balanced dataset...")
    df = build_balanced_dataset(df)

    print(f"\nBalanced dataset size: {len(df):,}")

    # First split: train vs remaining.
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["Label"],
        random_state=RANDOM_STATE,
    )

    # Split remaining 30% equally into validation/test.
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["Label"],
        random_state=RANDOM_STATE,
    )

    output_dir = (
        Path(__file__).resolve().parents[1]
        / "ml"
        / "splits"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df.to_csv(
        output_dir / "train.csv",
        index=False,
    )

    val_df.to_csv(
        output_dir / "validation.csv",
        index=False,
    )

    test_df.to_csv(
        output_dir / "test.csv",
        index=False,
    )

    print("\nSplit sizes:")
    print(f"Train:      {len(train_df):,}")
    print(f"Validation: {len(val_df):,}")
    print(f"Test:       {len(test_df):,}")

    print("\nTrain distribution:")
    print(train_df["Label"].value_counts())

    print("\nValidation distribution:")
    print(val_df["Label"].value_counts())

    print("\nTest distribution:")
    print(test_df["Label"].value_counts())


if __name__ == "__main__":
    main()