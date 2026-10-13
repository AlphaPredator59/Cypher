"""
Tests for CYPHER Alert Architecture, Decision Semantics, and Hierarchical Evaluation.

Verifies:
1. Alert type semantics: dual_signal, known_attack, anomaly_candidate, normal_traffic.
2. Anomaly candidate separation: VAE anomalies without classifier match are not promoted to confirmed attacks.
3. Decision rule consistency and metric calculation formulas.
4. Symmetrical hierarchical aggregation for Web Attacks.
5. Database and API endpoint persistence of alert_type.
"""

from __future__ import annotations

import pytest
import numpy as np
from fastapi.testclient import TestClient

from main import app
from ml.pipeline import ThreatPipeline, VAEScorer
from ml.evaluate_leakage_resistant import compute_hierarchical_metrics, compute_metrics, SUPPORTED_CLASSES
import database


@pytest.fixture
def test_client():
    return TestClient(app)


class MockScorer:
    def __init__(self, scores, threshold=38.50959):
        self.scores = np.array(scores, dtype=np.float32)
        self.threshold = float(threshold)

    def score_raw(self, X):
        flags = self.scores > self.threshold
        return {
            "threshold": self.threshold,
            "scores": self.scores,
            "flags": flags,
            "top_idx": np.zeros((len(self.scores), 5), dtype=int),
            "top_sqerr": np.zeros((len(self.scores), 5), dtype=np.float32),
        }


class MockClassifier:
    def __init__(self, labels, probs):
        self.labels = labels
        self.probs = probs
        self.classes = SUPPORTED_CLASSES

    def predict_batch(self, rows):
        prob_matrix = np.zeros((len(rows), len(self.classes)), dtype=np.float32)
        for i, (lbl, p) in enumerate(zip(self.labels, self.probs)):
            if lbl in self.classes:
                idx = self.classes.index(lbl)
                prob_matrix[i, idx] = p
        return np.array(self.labels), prob_matrix

    def predict_single(self, row):
        return self.labels[0], {self.labels[0]: self.probs[0]}


def test_alert_type_semantics_four_states():
    """
    Verifies that the ThreatPipeline outputs exact alert_type values matching the 4-state architecture:
    1. dual_signal (VAE > tau AND Classifier attack)
    2. known_attack (VAE <= tau AND Classifier attack)
    3. anomaly_candidate (VAE > tau AND Classifier BENIGN)
    4. normal_traffic (VAE <= tau AND Classifier BENIGN)
    """
    scores = [100.0, 10.0, 100.0, 10.0]
    labels = ["DDoS", "PortScan", "BENIGN", "BENIGN"]
    probs = [0.99, 0.95, 0.99, 0.99]

    mock_vae = MockScorer(scores)
    mock_clf = MockClassifier(labels, probs)

    pipeline = ThreatPipeline(
        vae_scorer=mock_vae,
        classifier=mock_clf,
        active_classifier="xgb",
        attack_confidence_threshold=0.50,
    )

    dummy_rows = [{"Dst Port": 80} for _ in range(4)]
    results = pipeline.predict_batch(dummy_rows)

    assert len(results) == 4

    # 1. dual_signal
    assert results[0]["alert_type"] == "dual_signal"
    assert results[0]["detection_source"] == "dual_confirmed"
    assert results[0]["is_anomaly"] is True
    assert results[0]["attack_type"] == "DDoS"

    # 2. known_attack
    assert results[1]["alert_type"] == "known_attack"
    assert results[1]["detection_source"] == "classifier_attack"
    assert results[1]["is_anomaly"] is True
    assert results[1]["attack_type"] == "PortScan"

    # 3. anomaly_candidate
    assert results[2]["alert_type"] == "anomaly_candidate"
    assert results[2]["detection_source"] == "vae_anomaly"
    assert results[2]["is_anomaly"] is True
    assert results[2]["attack_type"] == "Attack Type Unclassified"
    assert results[2]["attack_confidence"] is None  # Must not present sub-threshold or benign probability

    # 4. normal_traffic
    assert results[3]["alert_type"] == "normal_traffic"
    assert results[3]["detection_source"] == "clear_normal"
    assert results[3]["is_anomaly"] is False
    assert results[3]["attack_type"] == "Normal Traffic"
    assert results[3]["attack_confidence"] is None


def test_anomaly_candidate_not_promoted_to_confirmed_attack():
    """
    Verifies that when a flow produces high reconstruction error but the supervised
    classifier predicts BENIGN, it is designated as an anomaly_candidate for secondary triage,
    and NOT classified as a confirmed attack family.
    """
    mock_vae = MockScorer([85.0])
    mock_clf = MockClassifier(["BENIGN"], [0.98])

    pipeline = ThreatPipeline(vae_scorer=mock_vae, classifier=mock_clf, active_classifier="xgb")
    results = pipeline.predict_batch([{"Dst Port": 443}])

    res = results[0]
    assert res["alert_type"] == "anomaly_candidate"
    assert res["detection_source"] == "vae_anomaly"
    assert res["attack_type"] == "Attack Type Unclassified"
    assert res["attack_confidence"] is None
    assert res["verdict"] == "Anomalous — Attack Type Unclassified"


def test_metric_formulas_and_denominators():
    """
    Verifies metric formulas: Recall = TP / (TP + FN), FPR = FP / (FP + TN),
    Precision = TP / (TP + FP), Accuracy = (TP + TN) / Total.
    """
    y_true = np.array(["BENIGN"] * 100 + ["DDoS"] * 100)
    # Model predicts 95 DDoS correctly, 5 DDoS as BENIGN (FN=5)
    # Predicts 2 BENIGN as DDoS (FP=2), 98 BENIGN correctly (TN=98)
    y_pred = np.array(["BENIGN"] * 98 + ["DDoS"] * 2 + ["BENIGN"] * 5 + ["DDoS"] * 95)

    metrics = compute_metrics(y_true, y_pred, SUPPORTED_CLASSES)
    b = metrics["binary_detection"]

    assert b["tp"] == 95
    assert b["fn"] == 5
    assert b["fp"] == 2
    assert b["tn"] == 98
    assert b["attack_recall"] == pytest.approx(95 / 100, rel=1e-5)
    assert b["fpr"] == pytest.approx(2 / 100, rel=1e-5)
    assert b["precision"] == pytest.approx(95 / 97, rel=1e-5)
    assert b["binary_accuracy"] == pytest.approx(193 / 200, rel=1e-5)


def test_hierarchical_web_attack_symmetric_aggregation():
    """
    Verifies that hierarchical aggregation applies symmetrically to ground truth and predictions:
    Web Attack XSS and Web Attack Brute Force map to 'Web Attack', while other classes remain unchanged.
    """
    y_true = ["Web Attack XSS", "Web Attack Brute Force", "BENIGN", "DDoS"]
    # Model confused between XSS and Brute Force, but correctly predicted Web Attack family
    y_pred = ["Web Attack Brute Force", "Web Attack XSS", "BENIGN", "DDoS"]

    hierarchical = compute_hierarchical_metrics(y_true, y_pred)

    # In fine-grained 12-class, XSS and Brute Force would both be wrong
    # In hierarchical, both are correct 'Web Attack'
    assert hierarchical["accuracy"] == 1.0
    assert hierarchical["web_attack_metrics"]["recall"] == 1.0
    assert hierarchical["web_attack_metrics"]["precision"] == 1.0
    assert hierarchical["web_attack_metrics"]["f1-score"] == 1.0
    assert hierarchical["web_attack_metrics"]["support"] == 2


def test_database_persistence_and_query_alert_type():
    """
    Verifies that alert_type persists in SQLite and can be filtered in audit log queries.
    """
    database.init_db()

    sample_record = {
        "flow_id": "test-alert-type-001",
        "src_ip": "10.0.0.1",
        "dst_ip": "10.0.0.2",
        "score": 45.0,
        "threshold": 38.51,
        "is_anomaly": True,
        "attack_type": "SSH-Patator",
        "detection_source": "dual_confirmed",
        "alert_type": "dual_signal",
    }

    rec = database.save_flow_record(sample_record)
    assert rec.alert_type == "dual_signal"
    assert rec.detection_source == "dual_confirmed"

    # Query with alert_type filter
    filtered = database.query_recent_flows(limit=10, alert_type="dual_signal")
    assert any(r["flow_id"] == "test-alert-type-001" for r in filtered)

    # Check stats breakdown
    stats = database.get_flow_stats()
    assert "dual_signals" in stats
    assert "known_attacks" in stats
    assert "anomaly_candidates" in stats


def test_api_predict_returns_alert_type(test_client):
    """
    Verifies that /predict and /analyze-batch return the alert_type field in their API schema.
    """
    valid_payload = {
        "row": {
            "Destination Port": 80,
            "Flow Duration": 1000,
            "Total Fwd Packets": 5,
            "Total Backward Packets": 5,
            "Total Length of Fwd Packets": 200,
            "Total Length of Bwd Packets": 400,
            "Fwd Packet Length Max": 50,
            "Fwd Packet Length Min": 20,
            "Fwd Packet Length Mean": 40.0,
            "Fwd Packet Length Std": 5.0,
            "Bwd Packet Length Max": 80,
            "Bwd Packet Length Min": 40,
            "Bwd Packet Length Mean": 60.0,
            "Bwd Packet Length Std": 10.0,
            "Flow Bytes/s": 600.0,
            "Flow Packets/s": 10.0,
            "Flow IAT Mean": 100.0,
            "Flow IAT Std": 20.0,
            "Flow IAT Max": 200.0,
            "Flow IAT Min": 10.0,
        }
    }

    resp = test_client.post("/predict", json=valid_payload)
    assert resp.status_code == 200
    data = resp.json()

    assert "alert_type" in data
    assert data["alert_type"] in ("dual_signal", "known_attack", "anomaly_candidate", "normal_traffic")
    assert "detection_source" in data
    assert "score" in data
    assert "threshold" in data
