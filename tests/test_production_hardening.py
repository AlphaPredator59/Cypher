"""
CYPHER Production Hardening & Reliability Test Suite (Iteration 6).
Validates:
1. Database persistence, indexing, session lifecycle, and query semantics.
2. Resilience against persistence failures (endpoints return inference without crashing).
3. Google Gemini LLM summarizer resilience (missing key, timeout, rate limit, malformed JSON).
4. Strict input validation rejecting NaN, Inf, unparseable strings, and empty payloads (HTTP 422).
5. Cross-session IP flow inspection surviving in-memory cache clear.
6. API response backward compatibility for hybrid detection fields.
"""
import io
import json
import os
import unittest.mock as mock
import numpy as np
import pytest
from fastapi.testclient import TestClient

import database
from database import Base, FlowAuditRecord, get_db_session
import llm
from main import app, LAST_ANALYZED_TRAFFIC

client = TestClient(app)


# ============================================================================
# 1. Database Persistence & ORM Reliability Tests
# ============================================================================

def test_database_initialization_and_schema():
    """Verify table creation, column types, and indexes."""
    database.init_db()
    table = FlowAuditRecord.__table__
    
    # Verify core columns exist
    assert "id" in table.columns
    assert "flow_id" in table.columns
    assert "timestamp" in table.columns
    assert "src_ip" in table.columns
    assert "dst_ip" in table.columns
    assert "src_port" in table.columns
    assert "dst_port" in table.columns
    assert "protocol" in table.columns
    assert "score" in table.columns
    assert "threshold" in table.columns
    assert "is_anomaly" in table.columns
    assert "verdict" in table.columns
    assert "attack_type" in table.columns
    assert "attack_confidence" in table.columns
    assert "detection_source" in table.columns

    # Verify indexes
    index_names = {idx.name for idx in table.indexes}
    assert "ix_flow_src_dst_ip" in index_names


def test_database_save_and_query_single_record():
    """Verify saving a single flow audit record and retrieving its fields."""
    sample_flow = {
        "flow_id": "flow-test-001",
        "src_ip": "10.0.0.99",
        "dst_ip": "172.16.0.1",
        "src_port": 54321,
        "dst_port": 80,
        "protocol": "TCP",
        "score": 142.50,
        "threshold": 38.50959,
        "is_anomaly": True,
        "anomaly_confidence": 0.985,
        "attack_type": "DDoS",
        "attack_confidence": 0.992,
        "classification_status": "Evaluated",
        "verdict": "Anomalous — DDoS",
        "verdict_confidence": 0.992,
        "detection_source": "dual_confirmed",
    }

    record = database.save_flow_record(sample_flow)
    assert record.id is not None
    assert record.flow_id == "flow-test-001"
    assert record.src_ip == "10.0.0.99"
    assert record.is_anomaly is True
    assert record.detection_source == "dual_confirmed"

    # Query back by IP
    results = database.query_flows_by_ip("10.0.0.99", limit=10)
    assert len(results) >= 1
    matched = next(r for r in results if r["flow_id"] == "flow-test-001")
    assert matched["verdict"] == "Anomalous — DDoS"
    assert matched["attack_confidence"] == 0.992


def test_database_bulk_save_and_stats():
    """Verify batch insertion and aggregate statistics calculation."""
    flows = [
        {
            "flow_id": f"batch-{i}",
            "src_ip": "192.168.100.5",
            "dst_ip": "10.10.10.10",
            "score": 10.0 if i % 2 == 0 else 85.0,
            "threshold": 38.50959,
            "is_anomaly": (i % 2 != 0),
            "anomaly_confidence": 0.85,
            "attack_type": "DoS Hulk" if i % 2 != 0 else "Normal Traffic",
            "verdict": "Anomalous — DoS Hulk" if i % 2 != 0 else "Normal Traffic",
            "detection_source": "dual_confirmed" if i % 2 != 0 else "clear_normal",
        }
        for i in range(10)
    ]

    saved_count = database.save_flow_records(flows)
    assert saved_count == 10

    stats = database.get_flow_stats()
    assert stats["total"] >= 10
    assert stats["anomalous"] >= 5
    assert stats["benign"] >= 5


def test_persistence_failure_resilience_in_predict():
    """
    Verify that if the database write encounters an operational error,
    the /predict endpoint logs the failure but returns model inference successfully.
    """
    payload = {
        "row": {
            "Dst Port": 443,
            "Flow Duration": 500,
            "Tot Fwd Pkts": 3,
            "Tot Bwd Pkts": 3,
            "TotLen Fwd Pkts": 150,
            "TotLen Bwd Pkts": 150,
        }
    }

    with mock.patch("database.save_flow_record", side_effect=RuntimeError("Disk write error")):
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "is_anomaly" in data
        assert "score" in data
        assert "verdict" in data
        assert "detection_source" in data


def test_audit_logs_and_stats_endpoints():
    """Verify GET /audit-logs and GET /audit-stats endpoints."""
    res_stats = client.get("/audit-stats")
    assert res_stats.status_code == 200
    stats = res_stats.json()
    assert "total" in stats
    assert "anomalous" in stats
    assert "benign" in stats

    res_logs = client.get("/audit-logs?limit=5")
    assert res_logs.status_code == 200
    logs_data = res_logs.json()
    assert "count" in logs_data
    assert "stats" in logs_data
    assert "flows" in logs_data
    assert isinstance(logs_data["flows"], list)


# ============================================================================
# 2. LLM Alert Summarizer Resilience Tests
# ============================================================================

def test_llm_missing_api_key_graceful_fallback():
    """
    Verify that when GOOGLE_API_KEY is missing or empty:
    1. No network call is made.
    2. A structured fallback conforming to SCHEMA is returned.
    3. provider_status is 'missing_api_key'.
    4. No AI-generated narrative is fabricated.
    """
    payload = {
        "is_anomaly": True,
        "score": 92.45,
        "threshold": 38.50959,
        "verdict": "Anomalous — PortScan",
        "attack_type": "PortScan",
        "verdict_confidence": 0.998,
        "detection_source": "classifier_attack",
        "top_contributors": [{"feature": "Dst Port", "sq_error": 55.2}]
    }

    with mock.patch.dict(os.environ, {"GOOGLE_API_KEY": ""}):
        with mock.patch("llm._CLIENT_CACHE", None):
            res = llm.summarize_alert(payload, payload)
            
            assert res["provider_status"] == "missing_api_key"
            assert "AI incident summarization is unavailable" in res["summary"]
            assert "GOOGLE_API_KEY is not configured" in res["summary"]
            assert res["attack_assessment"]["predicted_type"] == "PortScan"
            assert res["attack_assessment"]["confidence"] == 0.998
            assert "No generative AI explanation was fabricated" in res["attack_assessment"]["interpretation"]
            assert len(res["recommended_actions"]["immediate"]) > 0


def test_llm_timeout_graceful_fallback():
    """Verify that a network timeout in Gemini produces a graceful fallback without crashing."""
    payload = {
        "is_anomaly": True,
        "score": 60.0,
        "threshold": 38.50959,
        "verdict": "Anomalous — DDoS",
        "attack_type": "DDoS",
        "verdict_confidence": 0.95,
        "detection_source": "dual_confirmed",
    }

    mock_client = mock.MagicMock()
    mock_client.invoke.side_effect = TimeoutError("Request timed out after 10.0s")

    with mock.patch("llm.get_llm_client", return_value=mock_client):
        res = llm.summarize_alert(payload, payload)
        assert res["provider_status"] == "timeout"
        assert "timed out" in res["summary"].lower()
        assert res["severity"] == "high"


def test_llm_rate_limit_graceful_fallback():
    """Verify that a 429 ResourceExhausted rate limit produces a graceful fallback."""
    payload = {
        "is_anomaly": False,
        "score": 12.0,
        "threshold": 38.50959,
        "verdict": "Normal Traffic",
        "attack_type": "Normal Traffic",
        "verdict_confidence": 0.88,
        "detection_source": "clear_normal",
    }

    mock_client = mock.MagicMock()
    mock_client.invoke.side_effect = Exception("429 ResourceExhausted: Quota exceeded for gemini-1.5-pro")

    with mock.patch("llm.get_llm_client", return_value=mock_client):
        res = llm.summarize_alert(payload, payload)
        assert res["provider_status"] == "rate_limited"
        assert "rate limit or quota" in res["summary"].lower()
        assert res["severity"] == "low"


def test_llm_malformed_json_graceful_fallback():
    """Verify that non-JSON text from Gemini provider is caught and returns schema fallback."""
    payload = {
        "is_anomaly": True,
        "score": 45.0,
        "threshold": 38.50959,
        "verdict": "Anomalous — Bot",
        "attack_type": "Bot",
        "verdict_confidence": 0.92,
        "detection_source": "classifier_attack",
    }

    mock_resp = mock.MagicMock()
    mock_resp.content = "I cannot analyze this security alert as requested because of content filters."

    mock_client = mock.MagicMock()
    mock_client.invoke.return_value = mock_resp

    with mock.patch("llm.get_llm_client", return_value=mock_client):
        res = llm.summarize_alert(payload, payload)
        assert res["provider_status"] == "malformed_response"
        assert "malformed" in res["summary"].lower() or "non-json" in res["summary"].lower()


def test_llm_model_not_found_graceful_fallback():
    """Verify that a 404 NOT_FOUND error produces a specific model_not_found status."""
    payload = {
        "is_anomaly": True,
        "score": 45.0,
        "threshold": 38.50959,
        "verdict": "Anomalous — Bot",
        "attack_type": "Bot",
        "verdict_confidence": 0.92,
        "detection_source": "classifier_attack",
    }

    mock_client = mock.MagicMock()
    mock_client.invoke.side_effect = Exception("404 NOT_FOUND: models/gemini-2.5-flash-lite is not found")

    with mock.patch("llm.get_llm_client", return_value=mock_client):
        res = llm.summarize_alert(payload, payload)
        assert res["provider_status"] == "model_not_found"
        assert "model not found" in res["summary"].lower() or "retired" in res["summary"].lower()


def test_llm_handles_list_content_response():
    """Verify that langchain-google-genai 4.x list-based content is normalized and parsed successfully."""
    payload = {
        "is_anomaly": True,
        "score": 50.0,
        "verdict": "Anomalous — PortScan",
        "attack_type": "PortScan"
    }

    mock_resp = mock.MagicMock()
    json_text = json.dumps({
        "severity": "high",
        "summary": "Real Gemini AI summary from multi-block response.",
        "what_is_wrong": ["Port scan pattern"],
        "evidence": [],
        "attack_assessment": {"predicted_type": "PortScan", "confidence": 0.95, "interpretation": "Active scan"},
        "recommended_actions": {"immediate": ["Block IP"], "short_term": [], "long_term": []},
        "questions_to_ask": [],
        "limitations": []
    })
    mock_resp.content = [{"type": "text", "text": f"```json\n{json_text}\n```"}]

    mock_client = mock.MagicMock()
    mock_client.invoke.return_value = mock_resp

    with mock.patch("llm.get_llm_client", return_value=mock_client):
        res = llm.summarize_alert(payload, payload)
        assert res["severity"] == "high"
        assert res["summary"] == "Real Gemini AI summary from multi-block response."
        assert "provider_status" not in res  # Successful LLM responses don't have provider_status


def test_summarize_endpoint_survives_provider_failure():
    """Verify that POST /summarize never returns HTTP 500 when Gemini fails."""
    req_body = {
        "llm_payload": {"is_anomaly": True, "score": 75.0, "verdict": "Anomalous — SSH-Patator"},
        "model_output": {"is_anomaly": True, "score": 75.0}
    }

    with mock.patch("main.summarize_alert", side_effect=RuntimeError("Unexpected connection abort")):
        res = client.post("/summarize", json=req_body)
        assert res.status_code == 200
        data = res.json()
        assert "summary" in data
        assert "unavailable" in data["summary"].lower() or "error" in data["summary"].lower()


# ============================================================================
# 3. API & Input Reliability Tests (Strict Validation)
# ============================================================================

def test_predict_rejects_empty_row():
    """Verify that POST /predict clearly rejects an empty row with HTTP 422."""
    res = client.post("/predict", json={"row": {}})
    assert res.status_code == 422
    assert "non-empty" in res.json()["detail"].lower()


def test_predict_rejects_nan_value():
    """Verify that POST /predict clearly rejects NaN without silent zero-imputation."""
    # 1. String "NaN" in JSON
    bad_row = {"Dst Port": "NaN", "Flow Duration": 1000}
    res = client.post("/predict", json={"row": bad_row})
    assert res.status_code == 422
    assert "non-finite" in res.json()["detail"].lower()
    assert "Dst Port" in res.json()["detail"]

    # 2. Raw NaN literal in JSON stream
    raw_payload = b'{"row": {"Dst Port": NaN, "Flow Duration": 1000}}'
    res_raw = client.post("/predict", content=raw_payload, headers={"Content-Type": "application/json"})
    assert res_raw.status_code == 422
    assert "non-finite" in res_raw.json()["detail"].lower()


def test_predict_rejects_inf_value():
    """Verify that POST /predict clearly rejects infinite values with HTTP 422."""
    # 1. String "Infinity" in JSON
    bad_row = {"Dst Port": 80, "Flow Duration": "Infinity"}
    res = client.post("/predict", json={"row": bad_row})
    assert res.status_code == 422
    assert "non-finite" in res.json()["detail"].lower()
    assert "Flow Duration" in res.json()["detail"]

    # 2. Raw Infinity literal in JSON stream
    raw_payload = b'{"row": {"Dst Port": 80, "Flow Duration": Infinity}}'
    res_raw = client.post("/predict", content=raw_payload, headers={"Content-Type": "application/json"})
    assert res_raw.status_code == 422
    assert "non-finite" in res_raw.json()["detail"].lower()


def test_predict_rejects_unparseable_string():
    """Verify that POST /predict clearly rejects non-numeric strings with HTTP 422."""
    bad_row = {"Dst Port": 80, "Tot Fwd Pkts": "malicious_string"}
    res = client.post("/predict", json={"row": bad_row})
    assert res.status_code == 422
    assert "malformed" in res.json()["detail"].lower()
    assert "Tot Fwd Pkts" in res.json()["detail"]


def test_analyze_batch_rejects_empty_rows():
    """Verify that POST /analyze-batch rejects empty row list with HTTP 400."""
    res = client.post("/analyze-batch", json={"rows": []})
    assert res.status_code == 400
    assert "cannot be empty" in res.json()["detail"].lower()


def test_analyze_batch_rejects_row_with_nan():
    """Verify that POST /analyze-batch rejects batch containing non-finite values with HTTP 422."""
    valid_row = {"Dst Port": 80, "Flow Duration": 1000}
    invalid_row = {"Dst Port": "NaN", "Flow Duration": 2000}
    res = client.post("/analyze-batch", json={"rows": [valid_row, invalid_row]})
    assert res.status_code == 422
    assert "Row 2" in res.json()["detail"]
    assert "non-finite" in res.json()["detail"].lower()


# ============================================================================
# 4. Cross-Session Database Inspection & Backward Compatibility
# ============================================================================

def test_inspect_ip_survives_memory_cache_wipe():
    """
    Verify that traffic flows persisted to the database remain discoverable
    via /inspect-ip even when the in-memory cache LAST_ANALYZED_TRAFFIC is wiped
    (simulating a server restart or multi-worker deployment).
    """
    unique_ip = "198.51.100.42"
    persisted_flow = {
        "flow_id": "cross-session-001",
        "src_ip": unique_ip,
        "dst_ip": "10.0.0.1",
        "src_port": 12345,
        "dst_port": 80,
        "score": 55.4,
        "threshold": 38.50959,
        "is_anomaly": True,
        "verdict": "Anomalous — DoS Hulk",
        "attack_type": "DoS Hulk",
        "detection_source": "dual_confirmed",
    }
    database.save_flow_record(persisted_flow)

    # Wipe in-memory session cache to test database fallback
    import main
    main.LAST_ANALYZED_TRAFFIC.clear()

    # Query IP via API endpoint
    res = client.post("/inspect-ip", json={"ip": unique_ip})
    assert res.status_code == 200
    data = res.json()
    assert data["ip_metadata_available"] is True
    assert data["matching_flows_count"] >= 1
    matched = next((f for f in data["flows"] if f.get("flow_id") == "cross-session-001"), None)
    assert matched is not None
    assert matched["src_ip"] == unique_ip
    assert matched["verdict"] == "Anomalous — DoS Hulk"


def test_hybrid_detection_fields_preserved():
    """
    Verify full backward compatibility of API response fields across /predict and /analyze.
    """
    valid_row = {
        "Dst Port": 80,
        "Flow Duration": 1000,
        "Tot Fwd Pkts": 2,
        "Tot Bwd Pkts": 2,
    }

    res = client.post("/predict", json={"row": valid_row})
    assert res.status_code == 200
    d = res.json()

    expected_keys = [
        "is_anomaly", "score", "threshold", "anomaly_confidence",
        "attack_type", "attack_confidence", "classification_status",
        "verdict", "verdict_confidence", "detection_source",
        "top_contributors", "input_issues"
    ]
    for k in expected_keys:
        assert k in d, f"Missing expected backward-compatible key '{k}' in /predict response"
