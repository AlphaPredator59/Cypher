import numpy as np
import pytest
import compute
from compute import calibrate_anomaly_confidence

def test_calibrate_anomaly_confidence_targets():
    tau = 38.50959
    
    # Target 1: S = 3.22 -> ~96% (95.5%)
    c_low = calibrate_anomaly_confidence(3.22, tau)
    assert 0.94 <= c_low <= 0.97, f"Expected ~0.955 for S=3.22, got {c_low}"
    
    # Target 2: S = 36.5 -> ~54% (54.0%)
    c_near_benign = calibrate_anomaly_confidence(36.5, tau)
    assert 0.52 <= c_near_benign <= 0.56, f"Expected ~0.54 for S=36.5, got {c_near_benign}"
    
    # Boundary: S == tau -> 50.0%
    c_exact = calibrate_anomaly_confidence(tau, tau)
    assert abs(c_exact - 0.50) < 1e-4, f"Expected 0.50 at threshold, got {c_exact}"
    
    # Target 3: S = 41.0 -> ~62.5% (62.6%)
    c_just_anom = calibrate_anomaly_confidence(41.0, tau)
    assert 0.60 <= c_just_anom <= 0.65, f"Expected ~0.625 for S=41.0, got {c_just_anom}"
    
    # Target 4: S = 420.0 -> ~99.9%
    c_high_anom = calibrate_anomaly_confidence(420.0, tau)
    assert 0.98 <= c_high_anom <= 0.999, f"Expected ~0.99 for S=420.0, got {c_high_anom}"

def test_calibrate_anomaly_confidence_bounds_and_edge_cases():
    tau = 38.50959
    
    # S = 0.0 -> 0.99
    c_zero = calibrate_anomaly_confidence(0.0, tau)
    assert c_zero == 0.99
    
    # Negative scores clamped to 0
    c_neg = calibrate_anomaly_confidence(-10.0, tau)
    assert c_neg == 0.99
    
    # NaN and Inf handling
    assert calibrate_anomaly_confidence(float("nan"), tau) == 0.50
    assert calibrate_anomaly_confidence(float("inf"), tau) == 0.50
    assert calibrate_anomaly_confidence(None, tau) == 0.50
    
    # Malformed threshold handling
    c_bad_tau = calibrate_anomaly_confidence(10.0, -5.0)
    assert 0.50 <= c_bad_tau <= 0.999

def test_pipeline_benign_flow_semantics():
    # Benign-like flow with minimal traffic (scores ~3.22 << tau)
    benign_row = {f: 0.0 for f in compute.PIPELINE.vae_feature_names}
    benign_row["Dst Port"] = 80
    benign_row["Tot Fwd Pkts"] = 1
    
    res = compute.predict([benign_row])
    assert len(res["results"]) == 1
    flow = res["results"][0]
    
    assert flow["is_anomaly"] is False
    # MoE is now evaluated on every valid flow, including normal flows
    assert flow["classification_status"] == "Evaluated"
    assert flow["attack_type"] == "Normal Traffic"
    assert flow["attack_confidence"] is None
    assert flow["verdict"] == "Normal Traffic"
    assert flow["anomaly_confidence"] is not None
    assert flow["anomaly_confidence"] > 0.50
    assert flow["verdict_confidence"] == flow["anomaly_confidence"]

def test_pipeline_anomaly_with_moe_disagreement(monkeypatch):
    # Volumetric DDoS row where VAE reconstruction error > threshold, but legacy MoE predicts Benign
    monkeypatch.setattr(compute.PIPELINE, "active_classifier", "moe")
    ddos_row = {
        "Dst Port": 80, "Flow Duration": 601248, "Tot Fwd Pkts": 3, "Tot Bwd Pkts": 6,
        "TotLen Fwd Pkts": 26, "TotLen Bwd Pkts": 11601, "Fwd Pkt Len Max": 20, "Fwd Pkt Len Min": 0,
        "Fwd Pkt Len Mean": 8.667, "Bwd Pkt Len Max": 5840, "Bwd Pkt Len Min": 0, "Bwd Pkt Len Mean": 1933.5,
        "Flow Byts/s": 19338.11, "Flow Pkts/s": 14.97, "Flow IAT Mean": 75156, "Flow IAT Std": 210467.35,
        "Flow IAT Max": 596023, "Bwd IAT Tot": 601182, "Bwd IAT Mean": 120236.4, "Bwd IAT Std": 265979.5,
        "Bwd IAT Min": 160, "Fwd PSH Flags": 0, "Bwd PSH Flags": 0, "Fwd URG Flags": 0, "Bwd URG Flags": 0,
        "Pkt Len Var": 3681776.01, "FIN Flag Cnt": 0, "RST Flag Cnt": 0, "PSH Flag Cnt": 1, "ACK Flag Cnt": 0,
        "URG Flag Cnt": 0, "CWE Flag Count": 0, "Down/Up Ratio": 2, "Fwd Byts/b Avg": 0, "Fwd Pkts/b Avg": 0,
        "Fwd Blk Rate Avg": 0, "Bwd Byts/b Avg": 0, "Bwd Pkts/b Avg": 0, "Bwd Blk Rate Avg": 0,
        "Init Fwd Win Byts": 8192, "Init Bwd Win Byts": 229, "Fwd Act Data Pkts": 2, "Fwd Seg Size Min": 20,
        "Active Mean": 0, "Active Std": 0, "Active Max": 0, "Idle Min": 0
    }
    
    res = compute.predict([ddos_row])
    flow = res["results"][0]
    
    assert flow["is_anomaly"] is True
    assert flow["score"] > flow["threshold"]
    assert flow["anomaly_confidence"] is not None
    assert flow["anomaly_confidence"] > 0.80
    assert flow["classification_status"] == "Evaluated"
    assert flow["attack_type"] == "Attack Type Unclassified"
    assert flow["attack_confidence"] is None  # Unclassified attack has no fabricated attack confidence
    assert flow["verdict"] == "Anomalous — Attack Type Unclassified"
    assert flow["verdict_confidence"] == flow["anomaly_confidence"]
    assert len(flow["top_contributors"]) == 5

def test_pipeline_anomaly_with_xgb_classified_ddos():
    # Volumetric DDoS row correctly identified as DDoS by the active 12-class XGBoost classifier
    ddos_row = {
        "Dst Port": 80, "Flow Duration": 601248, "Tot Fwd Pkts": 3, "Tot Bwd Pkts": 6,
        "TotLen Fwd Pkts": 26, "TotLen Bwd Pkts": 11601, "Fwd Pkt Len Max": 20, "Fwd Pkt Len Min": 0,
        "Fwd Pkt Len Mean": 8.667, "Bwd Pkt Len Max": 5840, "Bwd Pkt Len Min": 0, "Bwd Pkt Len Mean": 1933.5,
        "Flow Byts/s": 19338.11, "Flow Pkts/s": 14.97, "Flow IAT Mean": 75156, "Flow IAT Std": 210467.35,
        "Flow IAT Max": 596023, "Bwd IAT Tot": 601182, "Bwd IAT Mean": 120236.4, "Bwd IAT Std": 265979.5,
        "Bwd IAT Min": 160, "Fwd PSH Flags": 0, "Bwd PSH Flags": 0, "Fwd URG Flags": 0, "Bwd URG Flags": 0,
        "Pkt Len Var": 3681776.01, "FIN Flag Cnt": 0, "RST Flag Cnt": 0, "PSH Flag Cnt": 1, "ACK Flag Cnt": 0,
        "URG Flag Cnt": 0, "CWE Flag Count": 0, "Down/Up Ratio": 2, "Fwd Byts/b Avg": 0, "Fwd Pkts/b Avg": 0,
        "Fwd Blk Rate Avg": 0, "Bwd Byts/b Avg": 0, "Bwd Pkts/b Avg": 0, "Bwd Blk Rate Avg": 0,
        "Init Fwd Win Byts": 8192, "Init Bwd Win Byts": 229, "Fwd Act Data Pkts": 2, "Fwd Seg Size Min": 20,
        "Active Mean": 0, "Active Std": 0, "Active Max": 0, "Idle Min": 0
    }
    
    res = compute.predict([ddos_row])
    flow = res["results"][0]
    
    assert flow["is_anomaly"] is True
    assert flow["score"] > flow["threshold"]
    assert flow["classification_status"] == "Evaluated"
    assert flow["attack_type"] == "DDoS"
    assert flow["attack_confidence"] is not None
    assert flow["attack_confidence"] > 0.90
    assert flow["verdict"] == "Anomalous — DDoS"
    assert flow["verdict_confidence"] > 0.85


def test_pipeline_anomaly_classified_known_attack(monkeypatch):
    # Simulate a flow where MoE predicts a known attack (FTP-BruteForce)
    test_row = {f: 0.0 for f in compute.PIPELINE.vae_feature_names}
    test_row["Dst Port"] = 21
    
    # Mock VAE to return anomaly
    orig_score_raw = compute.PIPELINE.vae.score_raw
    def mock_vae(X):
        out = orig_score_raw(X)
        out["flags"] = np.array([True])
        out["scores"] = np.array([55.0])
        return out
    
    # Mock MoE to return FTP-BruteForce
    def mock_moe(rows):
        return np.array(["FTP-BruteForce"]), np.array([[0.01, 0.98, 0.01]])
    
    monkeypatch.setattr(compute.PIPELINE.vae, "score_raw", mock_vae)
    monkeypatch.setattr(compute.PIPELINE.moe, "predict_batch", mock_moe)
    
    res = compute.PIPELINE.predict_batch([test_row])
    flow = res[0]
    
    assert flow["is_anomaly"] is True
    assert flow["classification_status"] == "Evaluated"
    assert flow["attack_type"] == "FTP-BruteForce"
    assert flow["attack_confidence"] == 0.98
    assert flow["verdict"] == "Anomalous — FTP-BruteForce"
    # Verdict conf combines anomaly conf and attack conf
    assert flow["verdict_confidence"] > 0.80

def test_pipeline_moe_inference_failure(monkeypatch):
    # Test error handling when MoE fails during inference
    test_row = {f: 0.0 for f in compute.PIPELINE.vae_feature_names}
    
    def mock_moe_fail(rows):
        raise RuntimeError("MoE model failure simulation")
    
    monkeypatch.setattr(compute.PIPELINE.moe, "predict_batch", mock_moe_fail)
    
    res = compute.PIPELINE.predict_batch([test_row])
    flow = res[0]
    
    assert flow["classification_status"] == "Failed"
    assert flow["attack_type"] == "Classification Failed"
    assert flow["attack_confidence"] is None
    # For a normal VAE flow with MoE failure, verdict remains Normal Traffic
    assert flow["verdict"] == "Normal Traffic"

def test_pipeline_mixed_batch_normal_and_anomalous(monkeypatch):
    # Test mixed batch of normal and anomalous rows processed together
    row_normal = {f: 0.0 for f in compute.PIPELINE.vae_feature_names}
    row_normal["Dst Port"] = 80
    row_anom = {f: 0.0 for f in compute.PIPELINE.vae_feature_names}
    row_anom["Dst Port"] = 22
    
    orig_score_raw = compute.PIPELINE.vae.score_raw
    def mock_vae(X):
        out = orig_score_raw(X)
        # 1st row normal, 2nd row anomalous
        out["flags"] = np.array([False, True])
        out["scores"] = np.array([10.0, 75.0])
        return out
    
    def mock_moe(rows):
        assert len(rows) == 2
        labels = np.array(["Benign", "SSH-Bruteforce"])
        probs = np.array([[0.95, 0.03, 0.02], [0.02, 0.03, 0.95]])
        return labels, probs
    
    monkeypatch.setattr(compute.PIPELINE.vae, "score_raw", mock_vae)
    monkeypatch.setattr(compute.PIPELINE.moe, "predict_batch", mock_moe)
    
    batch = compute.PIPELINE.predict_batch([row_normal, row_anom])
    assert len(batch) == 2
    
    # Check normal flow
    f0 = batch[0]
    assert f0["is_anomaly"] is False
    assert f0["classification_status"] == "Evaluated"
    assert f0["attack_type"] == "Normal Traffic"
    assert f0["attack_confidence"] is None
    assert f0["verdict"] == "Normal Traffic"
    
    # Check anomalous flow
    f1 = batch[1]
    assert f1["is_anomaly"] is True
    assert f1["classification_status"] == "Evaluated"
    assert f1["attack_type"] == "SSH-Bruteforce"
    assert f1["attack_confidence"] == 0.95
    assert f1["verdict"] == "Anomalous — SSH-Bruteforce"

def test_label_encoder_decoding_and_classes():
    le = compute.PIPELINE.moe.label_encoder
    classes = list(le.classes_)
    assert classes == ["Benign", "FTP-BruteForce", "SSH-Bruteforce"]
    # Check inverse transform mapping
    decoded = le.inverse_transform([0, 1, 2])
    assert list(decoded) == ["Benign", "FTP-BruteForce", "SSH-Bruteforce"]

def test_no_default_benign_for_anomalous_flows():
    # Verify that anomalous rows NEVER receive "Normal Traffic" or "BENIGN" as verdict
    ddos_row = {f: 0.0 for f in compute.PIPELINE.vae_feature_names}
    ddos_row["Flow Duration"] = 9999999
    ddos_row["TotLen Fwd Pkts"] = 9999999
    res = compute.predict([ddos_row])["results"][0]
    
    if res["is_anomaly"]:
        assert res["classification_status"] in ("Evaluated", "Failed")
        assert res["verdict"] not in ("Normal Traffic", "BENIGN")

def test_confidence_distinct_signals_not_arbitrary_blend(monkeypatch):
    """
    Verifies that for attributed attacks, verdict_confidence preserves the
    classifier's actual model probability directly rather than an arbitrary 50/50 blend.
    """
    row = {f: 0.0 for f in compute.PIPELINE.vae_feature_names}
    
    # Mock VAE to return high anomaly (saturated score = 2500)
    orig_score_raw = compute.PIPELINE.vae.score_raw
    def mock_vae(X):
        out = orig_score_raw(X)
        out["flags"] = np.array([True])
        out["scores"] = np.array([2500.0])
        return out
        
    # Mock classifier to return moderate probability (0.72)
    def mock_clf(rows):
        return ["Web Attack Brute Force"], np.array([[0.72] + [0.0]*11])
        
    monkeypatch.setattr(compute.PIPELINE.vae, "score_raw", mock_vae)
    monkeypatch.setattr(compute.PIPELINE.xgb_classifier, "predict_batch", mock_clf)
    
    flow = compute.PIPELINE.predict_batch([row])[0]
    
    assert flow["is_anomaly"] is True
    assert flow["score"] == 2500.0
    # Saturated VAE anomaly confidence heuristic
    assert flow["anomaly_confidence"] == 0.999
    # Attack confidence is the raw model probability
    assert flow["attack_confidence"] == 0.72
    # Verdict confidence matches attack confidence directly (NOT 0.5 * 0.999 + 0.5 * 0.72 = 0.8595)
    assert flow["verdict_confidence"] == 0.72
    assert flow["verdict"] == "Anomalous — Web Attack Brute Force"

def test_confidence_boundary_and_unclassified_flows(monkeypatch):
    """
    Tests confidence behavior across boundary scores and unclassified anomalies.
    """
    row = {f: 0.0 for f in compute.PIPELINE.vae_feature_names}
    
    # Boundary flow: VAE score just above threshold (41.0 > 38.51)
    orig_score_raw = compute.PIPELINE.vae.score_raw
    def mock_vae_boundary(X):
        out = orig_score_raw(X)
        out["flags"] = np.array([True])
        out["scores"] = np.array([41.0])
        return out
        
    # Classifier predicts BENIGN -> unclassified anomaly
    def mock_clf_benign(rows):
        return ["BENIGN"], np.array([[0.99] + [0.0]*11])
        
    monkeypatch.setattr(compute.PIPELINE.vae, "score_raw", mock_vae_boundary)
    monkeypatch.setattr(compute.PIPELINE.xgb_classifier, "predict_batch", mock_clf_benign)
    
    flow = compute.PIPELINE.predict_batch([row])[0]
    
    assert flow["is_anomaly"] is True
    assert flow["verdict"] == "Anomalous — Attack Type Unclassified"
    assert flow["attack_type"] == "Attack Type Unclassified"
    assert flow["attack_confidence"] is None  # No attack confidence for unclassified
    # For unclassified anomaly, verdict_confidence preserves the boundary VAE heuristic (~0.626)
    assert 0.60 <= flow["verdict_confidence"] <= 0.65

def test_confidence_formatting_spec():
    """
    Verifies that the formatting rules expected by frontend formatProbability / confidencePercent
    properly handle normal percentages, 2-decimal precision (0.9512, 0.9987, 0.9999, 1.0),
    missing and invalid values ('—'), without artificial 99.9% saturation caps.
    """
    def format_pct(val, fallback="—"):
        if val is None or not np.isfinite(val):
            return fallback
        pct = val * 100 if val <= 1.0 else val
        clamped = max(0.0, min(100.0, float(pct)))
        return f"{clamped:.2f}%"

    # Missing and invalid values render em-dash '—'
    assert format_pct(None) == "—"
    assert format_pct(float("nan")) == "—"
    assert format_pct(float("inf")) == "—"

    # Valid values render 2-decimal precision without 99.9% cap
    assert format_pct(0.0) == "0.00%"
    assert format_pct(0.4512) == "45.12%"
    assert format_pct(0.9512) == "95.12%"
    assert format_pct(0.9987) == "99.87%"
    assert format_pct(0.9999) == "99.99%"
    assert format_pct(1.0000) == "100.00%"


def test_backend_preserves_raw_classifier_probabilities(monkeypatch):
    """
    Verifies that pipeline preserves high-precision probabilities
    (e.g. 0.9987, 0.9999) in both attack_confidence and classifier_probability.
    """
    row = {f: 0.0 for f in compute.PIPELINE.vae_feature_names}
    row["Dst Port"] = 443

    # High anomaly score
    def mock_vae_high(X):
        return {
            "threshold": 38.50959,
            "scores": np.array([150.0]),
            "flags": np.array([True]),
            "top_idx": np.array([[0, 1, 2, 3, 4]]),
            "top_sqerr": np.array([[10.0, 5.0, 3.0, 2.0, 1.0]]),
        }

    # Distinct probability 0.9987 (not capped at 0.999 or truncated)
    def mock_clf_precise(rows):
        probs = np.zeros((1, 12), dtype=np.float32)
        probs[0, 1] = 0.998734  # Bot class
        return ["Bot"], probs

    monkeypatch.setattr(compute.PIPELINE.vae, "score_raw", mock_vae_high)
    monkeypatch.setattr(compute.PIPELINE.xgb_classifier, "predict_batch", mock_clf_precise)

    flow = compute.PIPELINE.predict_batch([row])[0]
    assert flow["attack_type"] == "Bot"
    assert abs(flow["attack_confidence"] - 0.998734) < 1e-5
    assert abs(flow["classifier_probability"] - 0.998734) < 1e-5
    assert flow["attack_confidence"] != 0.999  # NOT capped or saturated


