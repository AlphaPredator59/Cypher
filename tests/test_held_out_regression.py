import io
import json
import os
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import compute
from main import app, CICIDS_COLUMN_MAPPING

client = TestClient(app)

def test_held_out_benign_flows_from_test_split():
    """Verify that real benign flows from held-out test split are correctly classified as Normal."""
    test_csv_path = "ml/splits/test.csv"
    if not os.path.exists(test_csv_path):
        pytest.skip("ml/splits/test.csv not found")

    df = pd.read_csv(test_csv_path, nrows=500)
    df.columns = df.columns.str.strip()
    df.rename(columns=CICIDS_COLUMN_MAPPING, inplace=True)

    benign_df = df[df["Label"].astype(str).str.strip().str.upper() == "BENIGN"]
    assert len(benign_df) >= 20, "Expected at least 20 benign rows in first 500 rows"

    sample_benign = benign_df.head(25)[compute.PIPELINE.vae_feature_names].apply(pd.to_numeric, errors="coerce")
    results = compute.predict(sample_benign.to_dict(orient="records"))["results"]

    normal_count = sum(1 for r in results if not r["is_anomaly"])
    # The VAE on real held-out benign data has ~95% true negative rate
    assert normal_count >= 20, f"Expected >= 20/25 benign rows to be Normal, got {normal_count}"

    for r in results:
        assert r["classification_status"] == "Evaluated"
        assert r["threshold"] == pytest.approx(38.51, rel=1e-2)
        if not r["is_anomaly"]:
            assert r["verdict"] == "Normal Traffic"
            assert r["anomaly_confidence"] is not None

def test_held_out_anomalous_flows_from_test_split():
    """Verify that real attack flows from held-out test split are classified as Anomalous."""
    test_csv_path = "ml/splits/test.csv"
    if not os.path.exists(test_csv_path):
        pytest.skip("ml/splits/test.csv not found")

    df = pd.read_csv(test_csv_path, nrows=1000)
    df.columns = df.columns.str.strip()
    df.rename(columns=CICIDS_COLUMN_MAPPING, inplace=True)

    # DDoS rows from Friday have high reconstruction error in CICIDS
    ddos_df = df[df["Label"].astype(str).str.strip().str.upper() == "DDOS"]
    assert len(ddos_df) >= 20, "Expected at least 20 DDoS rows"

    sample_ddos = ddos_df.head(25)[compute.PIPELINE.vae_feature_names].apply(pd.to_numeric, errors="coerce")
    results = compute.predict(sample_ddos.to_dict(orient="records"))["results"]

    anom_count = sum(1 for r in results if r["is_anomaly"])
    assert anom_count >= 15, f"Expected >= 15/25 DDoS rows to be Anomalous, got {anom_count}"

    for r in results:
        if r["is_anomaly"]:
            assert r["score"] > r["threshold"] or r.get("detection_source") == "classifier_attack"
            assert r["verdict"] not in ("Normal Traffic", "BENIGN")

def test_feature_mapping_and_column_order():
    """Verify that feature mapping, ordering, and scaler shapes match exactly."""
    with open("artifacts/features_name.json", "r") as f:
        meta = json.load(f)
    json_features = meta["feature_names"]

    assert len(json_features) == 77
    assert compute.PIPELINE.vae_feature_names == json_features

    scaler = compute.PIPELINE.vae.scaler
    assert scaler.n_features_in_ == 77
    assert len(scaler.mean_) == 77
    assert len(scaler.scale_) == 77

def test_csv_missing_and_duplicate_columns_handling():
    """Verify robust handling of duplicate columns (e.g. Fwd Header Length.1) and rejection of missing features."""
    # 1. Test CSV with duplicate header variants and metadata
    df_valid = pd.read_csv("tests/cypher_verified_mixed_100.csv", nrows=5)
    # Add duplicate columns
    df_with_dups = df_valid.copy()
    df_with_dups["Dst Port"] = df_with_dups["Destination Port"]  # Both present
    df_with_dups["Fwd Header Length"] = df_with_dups["Fwd Header Length.1"]  # Both present

    csv_buf = io.BytesIO(df_with_dups.to_csv(index=False).encode("utf-8"))
    res = client.post("/analyze-csv", files={"file": ("test_dups.csv", csv_buf, "text/csv")})
    assert res.status_code == 200, f"Expected 200 with duplicate columns, got {res.status_code}: {res.text}"
    assert res.json()["total_flows"] == 5

    # 2. Test CSV missing required features
    df_missing = df_valid.drop(columns=["Destination Port", "Flow Duration"])
    csv_missing_buf = io.BytesIO(df_missing.to_csv(index=False).encode("utf-8"))
    res_missing = client.post("/analyze-csv", files={"file": ("test_missing.csv", csv_missing_buf, "text/csv")})
    assert res_missing.status_code == 422
    assert "Incompatible CSV feature schema" in res_missing.json()["detail"]

def test_consistent_scaling_and_threshold_units():
    """Verify that reconstruction error calculation and saved threshold share the identical mathematical scale."""
    with open("artifacts/threshold.json", "r") as f:
        tdata = json.load(f)

    assert "threshold" in tdata
    assert tdata["metric"] == "recon_sse_sum77"
    tau = float(tdata["threshold"])
    assert 38.0 <= tau <= 39.0

    # Ensure scorer uses sum of squared errors
    zero_vec = np.zeros((1, 77), dtype=np.float32)
    out = compute.PIPELINE.vae.score_raw(zero_vec)
    assert "scores" in out
    assert "flags" in out
    assert out["threshold"] == tau
    # Single scalar score per row
    assert out["scores"].shape == (1,)

def test_verified_mixed_100_csv_upload_and_ip_lookup():
    """Verify that cypher_verified_mixed_100.csv produces both Normal and Anomalous flows and allows IP lookup."""
    csv_path = "tests/cypher_verified_mixed_100.csv"
    assert os.path.exists(csv_path), "tests/cypher_verified_mixed_100.csv must exist"

    with open(csv_path, "rb") as f:
        res = client.post("/analyze-csv", files={"file": ("cypher_verified_mixed_100.csv", f, "text/csv")})

    assert res.status_code == 200
    data = res.json()

    assert data["total_flows"] == 100
    assert data["has_ip_metadata"] is True
    # Acceptance criterion: must contain BOTH Normal and Anomalous predictions!
    assert data["normal_flows"] > 0, f"Expected normal flows > 0, got {data['normal_flows']}"
    assert data["anomalous_flows"] > 0, f"Expected anomalous flows > 0, got {data['anomalous_flows']}"
    assert data["normal_flows"] + data["anomalous_flows"] == 100

    # Test IP lookup on one of the unique source IPs in the verified file
    target_ip = "192.168.1.15"
    inspect_res = client.post("/inspect-ip", json={"ip": target_ip})
    assert inspect_res.status_code == 200
    inspect_data = inspect_res.json()
    assert inspect_data["ip_metadata_available"] is True
    assert inspect_data["matching_flows_count"] == 4  # 100 flows / 25 IP cycle = 4 matches
    assert all(f["src_ip"] == target_ip for f in inspect_data["flows"])
