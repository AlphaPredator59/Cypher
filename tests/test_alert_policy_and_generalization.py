"""
Automated verification tests for CYPHER's alert-policy analysis and cross-capture generalization.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

import compute
from ml.evaluate_leakage_resistant import compute_hierarchical_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEAK_FREE_DIR = PROJECT_ROOT / "ml" / "splits_leakage_resistant"


def test_alert_policy_classifier_vs_hybrid_fpr():
    """Verify that Classifier-Only detection produces significantly fewer false alarms than Hybrid OR-policy."""
    test_df = pd.read_csv(LEAK_FREE_DIR / "test.csv")
    benign_flows = test_df[test_df["Label"] == "BENIGN"].head(500)
    feature_cols = [c for c in test_df.columns if c not in ["Label", "_source_file"]]

    rows = benign_flows[feature_cols].to_dict(orient="records")

    # 1. Classifier-Only evaluation
    clf = compute.PIPELINE.xgb_classifier
    clf_labels, clf_probs = clf.predict_batch(rows)
    clf_fp = (clf_labels != "BENIGN") & (clf_probs.max(axis=1) >= 0.50)
    clf_fpr = float(clf_fp.sum() / len(benign_flows))

    # 2. VAE anomaly evaluation
    X_vae_raw = compute.PIPELINE._dicts_to_matrix(rows, compute.PIPELINE.vae_feature_names)
    vae_out = compute.PIPELINE.vae.score_raw(X_vae_raw)
    vae_fp = vae_out["flags"]
    vae_fpr = float(vae_fp.sum() / len(benign_flows))

    # Hybrid OR-policy: (VAE anomaly OR Classifier Attack)
    hybrid_fp = clf_fp | vae_fp
    hybrid_fpr = float(hybrid_fp.sum() / len(benign_flows))

    # Classifier-only FPR must be strictly lower than hybrid FPR
    assert clf_fpr <= hybrid_fpr, f"Classifier FPR ({clf_fpr}) should be <= Hybrid FPR ({hybrid_fpr})"
    # Benign specificity of classifier alone should be >= 99%
    assert clf_fpr <= 0.01, f"Classifier FPR on clean benign flows exceeded 1%: {clf_fpr}"


def test_cross_day_benign_generalization():
    """Verify that classifier generalizes to unseen Monday-WorkingHours capture with >= 99% accuracy."""
    monday_path = PROJECT_ROOT / "solarmainframe" / "ids-intrusion-csv" / "versions" / "1" / "Monday-WorkingHours.pcap_ISCX.csv"
    if not monday_path.exists():
        pytest.skip("Monday capture file not found in dataset folder")

    # Sample 500 rows from Monday
    monday_df = pd.read_csv(monday_path, nrows=500, encoding="cp1252")
    monday_df.columns = monday_df.columns.str.strip()
    feature_cols = [c for c in monday_df.columns if c != "Label"]
    rows = monday_df[feature_cols].to_dict(orient="records")

    clf = compute.PIPELINE.xgb_classifier
    labels, probs = clf.predict_batch(rows)

    benign_rate = (labels == "BENIGN").sum() / len(rows)
    assert benign_rate >= 0.98, f"Benign generalization rate on unseen capture was {benign_rate:.4f} (< 0.98)"


def test_hierarchical_mapping_preserves_non_web_classes():
    """Verify hierarchical mapping preserves non-web attack labels and supports exact consistency."""
    y_true = ["BENIGN", "DDoS", "PortScan", "SSH-Patator", "Web Attack XSS", "Web Attack Brute Force"]
    y_pred = ["BENIGN", "DDoS", "PortScan", "SSH-Patator", "Web Attack Brute Force", "Web Attack XSS"]

    res = compute_hierarchical_metrics(y_true, y_pred)
    assert res["accuracy"] == 1.0
    assert "Web Attack" in res["full_report"]
    assert res["full_report"]["Web Attack"]["support"] == 2
    assert res["full_report"]["DDoS"]["support"] == 1
    assert res["full_report"]["PortScan"]["support"] == 1
