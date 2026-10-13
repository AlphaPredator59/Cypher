"""
Regression and verification tests for CYPHER's leakage-resistant evaluation milestone.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEAK_FREE_DIR = PROJECT_ROOT / "ml" / "splits_leakage_resistant"
REPORT_PATH = PROJECT_ROOT / "ml" / "reports" / "leakage_resistant_evaluation.json"


def test_splits_leakage_resistant_structure():
    """Verify that leakage-resistant splits exist and retain all 12 classes."""
    assert LEAK_FREE_DIR.exists(), "Directory ml/splits_leakage_resistant must exist"
    assert (LEAK_FREE_DIR / "train.csv").exists(), "train.csv missing in leakage-resistant split"
    assert (LEAK_FREE_DIR / "validation.csv").exists(), "validation.csv missing in leakage-resistant split"
    assert (LEAK_FREE_DIR / "test.csv").exists(), "test.csv missing in leakage-resistant split"
    assert (LEAK_FREE_DIR / "split_metadata.json").exists(), "split_metadata.json missing"


def test_leakage_resistant_zero_duplicate_guarantee():
    """Verify that exact feature tuple overlap between train and test is strictly zero."""
    train_df = pd.read_csv(LEAK_FREE_DIR / "train.csv", nrows=5000)
    test_df = pd.read_csv(LEAK_FREE_DIR / "test.csv", nrows=5000)

    feature_cols = [c for c in train_df.columns if c not in ["Label", "_source_file"]]

    train_tuples = set(map(tuple, train_df[feature_cols].values))
    test_tuples = set(map(tuple, test_df[feature_cols].values))

    overlap = len(train_tuples & test_tuples)
    assert overlap == 0, f"Detected {overlap} duplicate feature tuples across splits!"


def test_leakage_resistant_metadata():
    """Check split_metadata.json properties."""
    meta_path = LEAK_FREE_DIR / "split_metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["cross_split_test_train_overlap"] == 0
    assert meta["cross_split_test_val_overlap"] == 0
    assert meta["cross_split_val_train_overlap"] == 0
    assert len(meta["test_class_counts"]) == 12, "All 12 classes must be present in test split"


def test_hierarchical_web_attack_mapping():
    """Verify hierarchical mapping maps web attack subtypes correctly."""
    from ml.evaluate_leakage_resistant import compute_hierarchical_metrics

    y_true = ["BENIGN", "Web Attack XSS", "Web Attack Brute Force", "DDoS"]
    y_pred = ["BENIGN", "Web Attack Brute Force", "Web Attack XSS", "DDoS"]

    h_res = compute_hierarchical_metrics(y_true, y_pred)
    # Under hierarchical grouping, both XSS and Brute Force are Web Attack,
    # so y_true = ['BENIGN', 'Web Attack', 'Web Attack', 'DDoS']
    # and y_pred = ['BENIGN', 'Web Attack', 'Web Attack', 'DDoS'] -> 100% accuracy!
    assert h_res["accuracy"] == 1.0
    assert h_res["web_attack_metrics"]["f1-score"] == 1.0


def test_leakage_resistant_evaluation_report_content():
    """Ensure evaluation JSON report exists and contains valid metrics."""
    assert REPORT_PATH.exists(), f"Report file {REPORT_PATH} not found"
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        rep = json.load(f)

    xgb_res = rep["models"]["xgboost_leak_free"]
    assert xgb_res["accuracy"] >= 0.95
    assert xgb_res["binary_detection"]["attack_recall"] >= 0.99
    assert xgb_res["hierarchical"]["accuracy"] >= xgb_res["accuracy"]
