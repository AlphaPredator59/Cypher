import os
import pytest
import numpy as np
from fastapi.testclient import TestClient

import compute
from ml.pipeline import ThreatPipeline, VAEScorer
from main import app

client = TestClient(app)


def test_hybrid_vae_only_anomaly_with_classifier_benign(monkeypatch):
    """
    State 3: VAE flags anomaly (score > tau), but classifier predicts BENIGN.
    XGBoost predicting BENIGN must NOT override the VAE anomaly.
    Result must be an unclassified anomaly with detection_source='vae_anomaly'.
    """
    row = {f: 0.0 for f in compute.PIPELINE.vae_feature_names}
    
    orig_score_raw = compute.PIPELINE.vae.score_raw
    def mock_vae_high(X):
        out = orig_score_raw(X)
        out["flags"] = np.array([True])
        out["scores"] = np.array([125.0])  # well above tau 38.51
        return out
        
    def mock_clf_benign(rows):
        return ["BENIGN"], np.array([[0.98] + [0.0]*11])
        
    monkeypatch.setattr(compute.PIPELINE.vae, "score_raw", mock_vae_high)
    monkeypatch.setattr(compute.PIPELINE.xgb_classifier, "predict_batch", mock_clf_benign)
    
    res = compute.predict([row])["results"][0]
    
    assert res["is_anomaly"] is True
    assert res["detection_source"] == "vae_anomaly"
    assert res["verdict"] == "Anomalous — Attack Type Unclassified"
    assert res["attack_type"] == "Attack Type Unclassified"
    assert res["attack_confidence"] is None  # no attack probability presented
    assert res["verdict_confidence"] == res["anomaly_confidence"]
    assert res["score"] == 125.0


def test_hybrid_classifier_detected_attack_below_vae_threshold(monkeypatch):
    """
    State 2: VAE score is below threshold (stealth attack, s <= tau), but
    supervised classifier predicts a known attack (e.g. PortScan) with high confidence.
    Result must be flagged as an anomaly with detection_source='classifier_attack'.
    """
    row = {f: 0.0 for f in compute.PIPELINE.vae_feature_names}
    
    orig_score_raw = compute.PIPELINE.vae.score_raw
    def mock_vae_low(X):
        out = orig_score_raw(X)
        out["flags"] = np.array([False])
        out["scores"] = np.array([15.2])  # below tau 38.51
        return out
        
    def mock_clf_portscan(rows):
        # 12 classes, PortScan has 0.94 probability
        probs = np.zeros((1, 12), dtype=np.float32)
        probs[0, 8] = 0.94  # high confidence
        return ["PortScan"], probs
        
    monkeypatch.setattr(compute.PIPELINE.vae, "score_raw", mock_vae_low)
    monkeypatch.setattr(compute.PIPELINE.xgb_classifier, "predict_batch", mock_clf_portscan)
    
    res = compute.predict([row])["results"][0]
    
    assert res["is_anomaly"] is True
    assert res["detection_source"] == "classifier_attack"
    assert res["verdict"] == "Anomalous — PortScan"
    assert res["attack_type"] == "PortScan"
    assert res["attack_confidence"] == 0.94
    assert res["verdict_confidence"] == 0.94
    assert res["score"] == 15.2
    assert res["score"] < res["threshold"]


def test_hybrid_clean_normal_traffic(monkeypatch):
    """
    State 4: Both VAE score is below threshold AND classifier predicts BENIGN.
    Result must be clean normal traffic with detection_source='clear_normal'.
    """
    row = {f: 0.0 for f in compute.PIPELINE.vae_feature_names}
    
    orig_score_raw = compute.PIPELINE.vae.score_raw
    def mock_vae_low(X):
        out = orig_score_raw(X)
        out["flags"] = np.array([False])
        out["scores"] = np.array([8.4])
        return out
        
    def mock_clf_benign(rows):
        return ["BENIGN"], np.array([[0.99] + [0.0]*11])
        
    monkeypatch.setattr(compute.PIPELINE.vae, "score_raw", mock_vae_low)
    monkeypatch.setattr(compute.PIPELINE.xgb_classifier, "predict_batch", mock_clf_benign)
    
    res = compute.predict([row])["results"][0]
    
    assert res["is_anomaly"] is False
    assert res["detection_source"] == "clear_normal"
    assert res["verdict"] == "Normal Traffic"
    assert res["attack_type"] == "Normal Traffic"
    assert res["attack_confidence"] is None
    assert res["verdict_confidence"] == res["anomaly_confidence"]


def test_hybrid_dual_confirmed_attack(monkeypatch):
    """
    State 1: Both VAE score exceeds threshold AND classifier predicts an attack (e.g. DDoS).
    Result must be flagged with detection_source='dual_confirmed'.
    """
    row = {f: 0.0 for f in compute.PIPELINE.vae_feature_names}
    
    orig_score_raw = compute.PIPELINE.vae.score_raw
    def mock_vae_high(X):
        out = orig_score_raw(X)
        out["flags"] = np.array([True])
        out["scores"] = np.array([320.0])
        return out
        
    def mock_clf_ddos(rows):
        probs = np.zeros((1, 12), dtype=np.float32)
        probs[0, 2] = 0.995
        return ["DDoS"], probs
        
    monkeypatch.setattr(compute.PIPELINE.vae, "score_raw", mock_vae_high)
    monkeypatch.setattr(compute.PIPELINE.xgb_classifier, "predict_batch", mock_clf_ddos)
    
    res = compute.predict([row])["results"][0]
    
    assert res["is_anomaly"] is True
    assert res["detection_source"] == "dual_confirmed"
    assert res["verdict"] == "Anomalous — DDoS"
    assert res["attack_type"] == "DDoS"
    assert res["attack_confidence"] == 0.995
    assert res["verdict_confidence"] == 0.995


def test_hybrid_low_confidence_classifier_policy(monkeypatch):
    """
    Tests policy behavior when classifier predicts an attack but with confidence below operating point (theta=0.50).
    Case A: VAE below threshold and classifier confidence < 0.50 -> Clear normal traffic.
    Case B: VAE above threshold and classifier confidence < 0.50 -> VAE anomaly (unclassified).
    """
    row = {f: 0.0 for f in compute.PIPELINE.vae_feature_names}
    orig_score_raw = compute.PIPELINE.vae.score_raw
    
    def mock_clf_low_conf(rows):
        # PortScan predicted with 0.35 confidence (< 0.50 operating point)
        probs = np.zeros((1, 12), dtype=np.float32)
        probs[0, 8] = 0.35
        return ["PortScan"], probs

    monkeypatch.setattr(compute.PIPELINE.xgb_classifier, "predict_batch", mock_clf_low_conf)
    
    # Case A: VAE below threshold
    def mock_vae_low(X):
        out = orig_score_raw(X)
        out["flags"] = np.array([False])
        out["scores"] = np.array([12.0])
        return out
    monkeypatch.setattr(compute.PIPELINE.vae, "score_raw", mock_vae_low)
    
    res_a = compute.predict([row])["results"][0]
    assert res_a["is_anomaly"] is False
    assert res_a["detection_source"] == "clear_normal"
    assert res_a["verdict"] == "Normal Traffic"
    assert res_a["attack_confidence"] is None

    # Case B: VAE above threshold
    def mock_vae_high(X):
        out = orig_score_raw(X)
        out["flags"] = np.array([True])
        out["scores"] = np.array([85.0])
        return out
    monkeypatch.setattr(compute.PIPELINE.vae, "score_raw", mock_vae_high)
    
    res_b = compute.predict([row])["results"][0]
    assert res_b["is_anomaly"] is True
    assert res_b["detection_source"] == "vae_anomaly"
    assert res_b["verdict"] == "Anomalous — Attack Type Unclassified"
    assert res_b["attack_type"] == "Attack Type Unclassified"
    assert res_b["attack_confidence"] is None


def test_hybrid_non_finite_and_missing_feature_robustness():
    """
    Verifies that pipeline gracefully handles rows with missing keys, non-numeric strings,
    NaNs, and infinities without crashing.
    """
    # 1. Completely empty row
    out_empty = compute.predict([{}])
    assert len(out_empty["results"]) == 1
    assert out_empty["results"][0]["score"] >= 0.0

    # 2. Row with NaN and Inf values
    dirty_row = {f: 0.0 for f in compute.PIPELINE.vae_feature_names}
    dirty_row["Dst Port"] = float("nan")
    dirty_row["Flow Duration"] = float("inf")
    dirty_row["Tot Fwd Pkts"] = "not_a_number"
    
    out_dirty = compute.predict([dirty_row])
    assert len(out_dirty["results"]) == 1
    res = out_dirty["results"][0]
    assert np.isfinite(res["score"])
    assert 0.50 <= res["anomaly_confidence"] <= 0.999
    assert res["detection_source"] in ("clear_normal", "vae_anomaly", "classifier_attack", "dual_confirmed")


def test_api_endpoints_consistency():
    """
    Verifies that all API endpoints (/predict, /analyze, /attack-type, /anomaly-flag)
    maintain backward-compatible schemas and return consistent verdict and detection_source fields.
    """
    row = {f: 0.0 for f in compute.PIPELINE.vae_feature_names}
    row["Dst Port"] = 80
    row["Tot Fwd Pkts"] = 1
    
    # 1. /predict
    res_pred = client.post("/predict", json={"row": row})
    assert res_pred.status_code == 200
    d_pred = res_pred.json()
    assert "detection_source" in d_pred
    assert "is_anomaly" in d_pred
    assert "verdict" in d_pred
    assert "verdict_confidence" in d_pred

    # 2. /analyze
    res_ana = client.post("/analyze", json={"row": row})
    assert res_ana.status_code == 200
    d_ana = res_ana.json()
    assert d_ana["model_output"]["detection_source"] == d_pred["detection_source"]
    assert d_ana["llm_payload"]["detection_source"] == d_pred["detection_source"]

    # 3. /anomaly-flag
    res_flag = client.post("/anomaly-flag", json={"row": row})
    assert res_flag.status_code == 200
    d_flag = res_flag.json()
    assert d_flag["is_anomaly"] == d_pred["is_anomaly"]
    assert d_flag["detection_source"] == d_pred["detection_source"]

    # 4. /attack-type
    res_atk = client.post("/attack-type", json={"row": row})
    assert res_atk.status_code == 200
    d_atk = res_atk.json()
    assert d_atk["attack_type"] == d_pred["attack_type"]
    assert d_atk["verdict"] == d_pred["verdict"]
    assert d_atk["detection_source"] == d_pred["detection_source"]
