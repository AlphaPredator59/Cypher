"""
Comprehensive test suite for the 12-class XGBAttackClassifier in CYPHER IDS.

Tests:
1. Model loading, artifact integrity, and class alignment.
2. Expected feature dimensions (78) and exact column order.
3. Label encoder consistency between JSON booster and LabelEncoder.
4. Prediction outputs, probability distributions, and confidence bounds.
5. Strict validation of missing columns, non-finite values (NaN, Inf).
6. End-to-end inference on representative attack vectors.
"""

import pytest
import numpy as np
import pandas as pd
from ml.classifier import XGBAttackClassifier
from ml.features import XGB_FEATURE_NAMES, VAE_FEATURE_NAMES, normalize_flow_for_xgb


@pytest.fixture
def classifier():
    return XGBAttackClassifier()


def test_classifier_loading_and_classes(classifier):
    """Verify that the model and label encoder load cleanly with all 12 classes."""
    assert classifier.num_classes == 12
    assert len(classifier.classes) == 12
    expected_classes = [
        "BENIGN", "Bot", "DDoS", "DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest",
        "DoS slowloris", "FTP-Patator", "PortScan", "SSH-Patator",
        "Web Attack Brute Force", "Web Attack XSS"
    ]
    assert classifier.classes == expected_classes


def test_feature_dimensions_and_schema(classifier):
    """Verify that the classifier expects exactly 78 features in canonical order."""
    assert len(classifier.feature_names) == 78
    assert classifier.feature_names == XGB_FEATURE_NAMES
    # Check that the duplicate column Fwd Header Length.1 is accounted for
    assert "Fwd Header Length.1" in classifier.feature_names
    assert "Destination Port" in classifier.feature_names
    assert "Idle Min" in classifier.feature_names


def test_label_encoder_index_alignment(classifier):
    """Verify that booster output indices map 1:1 to label encoder classes."""
    dummy_input = np.zeros((1, 78), dtype=np.float32)
    probs = classifier.predict_proba(dummy_input)
    assert probs.shape == (1, 12)
    assert np.isclose(np.sum(probs), 1.0, atol=1e-4)

    for idx, class_name in enumerate(classifier.classes):
        assert classifier.label_encoder.classes_[idx] == class_name
        assert classifier.label_encoder.inverse_transform([idx])[0] == class_name


def test_prediction_output_shapes_and_confidence(classifier):
    """Verify batch and single prediction return correct types and bounded confidence."""
    rows = [
        {"Destination Port": 80, "Flow Duration": 1000},
        {"Destination Port": 443, "Flow Duration": 5000},
    ]

    # Predict batch
    labels, probs = classifier.predict_batch(rows)
    assert isinstance(labels, np.ndarray)
    assert len(labels) == 2
    assert probs.shape == (2, 12)

    for lbl in labels:
        assert lbl in classifier.classes

    for p in probs:
        assert np.all(p >= 0.0)
        assert np.all(p <= 1.0)
        assert np.isclose(np.sum(p), 1.0, atol=1e-4)

    # Predict single
    res = classifier.predict_single(rows[0])
    assert "predicted_label" in res
    assert "confidence" in res
    assert "is_attack" in res
    assert "probabilities" in res
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["probabilities"]) == 12


def test_feature_mapping_and_duplicate_derivation():
    """Verify normalize_flow_for_xgb derives Fwd Header Length.1 from Fwd Header Length."""
    row = {
        "Dst Port": 80,  # Short name
        "Flow Duration": 100000,
        "Fwd Header Len": 32,  # Short name for Fwd Header Length
    }
    vec = normalize_flow_for_xgb(row, strict=False)
    assert len(vec) == 78
    assert vec[0] == 80.0  # Destination Port
    assert vec[1] == 100000.0  # Flow Duration
    # Fwd Header Length is index 34, Fwd Header Length.1 is index 55
    idx_fwd_len = XGB_FEATURE_NAMES.index("Fwd Header Length")
    idx_fwd_len_1 = XGB_FEATURE_NAMES.index("Fwd Header Length.1")
    assert vec[idx_fwd_len] == 32.0
    assert vec[idx_fwd_len_1] == 32.0


def test_strict_validation_rejects_missing_and_non_finite():
    """Verify that strict=True rejects non-finite values (NaN, Inf) and missing features."""
    # Non-finite value
    bad_row = {"Destination Port": float("nan")}
    with pytest.raises(ValueError, match="non-finite"):
        normalize_flow_for_xgb(bad_row, strict=True)

    bad_inf = {"Destination Port": float("inf")}
    with pytest.raises(ValueError, match="non-finite"):
        normalize_flow_for_xgb(bad_inf, strict=True)

    # Incomplete flow in strict mode
    incomplete_row = {"Destination Port": 80}
    with pytest.raises(ValueError, match="Missing required feature"):
        normalize_flow_for_xgb(incomplete_row, strict=True)


def test_representative_attack_classification(classifier):
    """Verify that known attack flow vectors are classified into their respective classes."""
    # Representative DDoS flow
    ddos_flow = {
        "Destination Port": 80, "Flow Duration": 601248, "Total Fwd Packets": 3,
        "Total Backward Packets": 6, "Total Length of Fwd Packets": 26,
        "Total Length of Bwd Packets": 11601, "Fwd Packet Length Max": 20,
        "Fwd Packet Length Min": 0, "Fwd Packet Length Mean": 8.667,
        "Bwd Packet Length Max": 5840, "Bwd Packet Length Min": 0,
        "Bwd Packet Length Mean": 1933.5, "Flow Bytes/s": 19338.11,
        "Flow Packets/s": 14.97, "Flow IAT Mean": 75156, "Flow IAT Std": 210467.35,
        "Flow IAT Max": 596023, "Bwd IAT Total": 601182, "Bwd IAT Mean": 120236.4,
        "Bwd IAT Std": 265979.5, "Bwd IAT Min": 160, "Fwd PSH Flags": 0,
        "Bwd PSH Flags": 0, "Fwd URG Flags": 0, "Bwd URG Flags": 0,
        "Packet Length Variance": 3681776.01, "FIN Flag Count": 0, "RST Flag Count": 0,
        "PSH Flag Count": 1, "ACK Flag Count": 0, "URG Flag Count": 0,
        "CWE Flag Count": 0, "Down/Up Ratio": 2, "Fwd Avg Bytes/Bulk": 0,
        "Fwd Avg Packets/Bulk": 0, "Fwd Avg Bulk Rate": 0, "Bwd Avg Bytes/Bulk": 0,
        "Bwd Avg Packets/Bulk": 0, "Bwd Avg Bulk Rate": 0,
        "Init_Win_bytes_forward": 8192, "Init_Win_bytes_backward": 229,
        "act_data_pkt_fwd": 2, "min_seg_size_forward": 20, "Fwd Header Length": 60,
    }

    res = classifier.predict_single(ddos_flow)
    assert res["predicted_label"] == "DDoS"
    assert res["confidence"] > 0.80
    assert res["is_attack"] is True
