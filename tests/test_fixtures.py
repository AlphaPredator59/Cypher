import hashlib
import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from ml.features import XGB_FEATURE_NAMES
from ml.pipeline import predict

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
CSV_PATH = FIXTURES_DIR / "manual_crosscheck.csv"
MANIFEST_PATH = FIXTURES_DIR / "manual_crosscheck_manifest.json"

EXPECTED_CLASSES = [
    "BENIGN",
    "Bot",
    "DDoS",
    "DoS GoldenEye",
    "DoS Hulk",
    "DoS Slowhttptest",
    "DoS slowloris",
    "FTP-Patator",
    "PortScan",
    "SSH-Patator",
    "Web Attack Brute Force",
    "Web Attack XSS",
]


def test_fixture_files_exist():
    assert CSV_PATH.exists(), f"Missing fixture file: {CSV_PATH}"
    assert MANIFEST_PATH.exists(), f"Missing manifest file: {MANIFEST_PATH}"


def test_manual_crosscheck_csv_schema_and_features():
    df = pd.read_csv(CSV_PATH)
    assert len(df) == 49, f"Expected 49 rows, got {len(df)}"
    assert len(df.columns) == 80, f"Expected 80 columns (78 features + 2 expected labels), got {len(df.columns)}"

    # Feature columns must match XGB_FEATURE_NAMES in exact order
    feature_cols = list(df.columns[:78])
    assert feature_cols == XGB_FEATURE_NAMES, "Feature columns do not match XGB_FEATURE_NAMES in order"

    # Metadata columns
    assert df.columns[78] == "expected_label"
    assert df.columns[79] == "expected_binary_label"

    # All values in feature columns must be finite numbers
    feature_mat = df[feature_cols].to_numpy()
    assert np.isfinite(feature_mat).all(), "Found non-finite (NaN or Inf) feature values in fixture"


def test_manual_crosscheck_classes_and_binary_labels():
    df = pd.read_csv(CSV_PATH)
    unique_classes = sorted(df["expected_label"].unique())
    assert unique_classes == sorted(EXPECTED_CLASSES), f"Classes mismatch. Got: {unique_classes}"

    # Verify counts: 5 benign, 4 each for 11 attack classes
    counts = df["expected_label"].value_counts().to_dict()
    assert counts["BENIGN"] == 5
    for atk_cls in EXPECTED_CLASSES:
        if atk_cls != "BENIGN":
            assert counts[atk_cls] == 4, f"Class {atk_cls} has {counts[atk_cls]} samples (expected 4)"

    # Verify expected_binary_label mapping
    for _, row in df.iterrows():
        if row["expected_label"] == "BENIGN":
            assert row["expected_binary_label"] == "BENIGN"
        else:
            assert row["expected_binary_label"] == "ATTACK"


def test_manifest_integrity():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(CSV_PATH, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()

    assert manifest["sha256"] == actual_hash, "Manifest SHA-256 does not match actual CSV hash"
    assert manifest["total_rows"] == 49
    assert len(manifest["rows"]) == 49
    assert manifest["feature_count"] == 78
    assert manifest["class_shortages"] == []


def test_fixture_pipeline_evaluation_integrity():
    df = pd.read_csv(CSV_PATH)

    # Strictly separate feature inputs from expected labels
    features_only = df[XGB_FEATURE_NAMES].to_dict(orient="records")

    # Pipeline call must not receive expected label columns
    for row_dict in features_only:
        assert "expected_label" not in row_dict
        assert "expected_binary_label" not in row_dict
        assert "Label" not in row_dict

    out = predict(features_only)
    results = out["results"]
    assert len(results) == 49

    # All results must contain essential decision fields
    for r in results:
        assert "is_anomaly" in r
        assert "score" in r
        assert "threshold" in r
        assert "attack_type" in r
        assert "alert_type" in r
        assert "verdict" in r
        assert r["alert_type"] in ("dual_signal", "known_attack", "anomaly_candidate", "normal_traffic")

    # Offline accuracy check vs manifest expectations
    exact_matches = 0
    binary_matches = 0
    for idx, (res, expected_lbl) in enumerate(zip(results, df["expected_label"])):
        is_attack = res["is_anomaly"]
        expected_binary = "BENIGN" if expected_lbl == "BENIGN" else "ATTACK"
        pred_binary = "ATTACK" if is_attack else "BENIGN"

        exact_match = (res["attack_type"].lower() == expected_lbl.lower()) or (expected_lbl == "BENIGN" and not is_attack)
        if exact_match:
            exact_matches += 1
        if pred_binary == expected_binary:
            binary_matches += 1

    assert exact_matches == 46, f"Expected 46/49 exact matches, got {exact_matches}"
    assert binary_matches == 48, f"Expected 48/49 binary matches, got {binary_matches}"
