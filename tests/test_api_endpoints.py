import io
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}

def test_predict_endpoint_calibrated_confidence():
    payload = {
        "row": {
            "Dst Port": 80,
            "Flow Duration": 1000,
            "Tot Fwd Pkts": 2,
            "Tot Bwd Pkts": 2,
            "TotLen Fwd Pkts": 100,
            "TotLen Bwd Pkts": 100
        }
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "is_anomaly" in data
    assert "score" in data
    assert "threshold" in data
    assert "anomaly_confidence" in data
    assert "classification_status" in data
    assert "verdict" in data
    assert "verdict_confidence" in data
    # Ensure confidence is within [0.5, 0.999]
    assert 0.50 <= data["anomaly_confidence"] <= 0.999
    assert data["classification_status"] in ("Evaluated", "Failed")

def test_inspect_ip_validation():
    # Valid IPv4
    res_v4 = client.post("/inspect-ip", json={"ip": "192.168.1.100"})
    assert res_v4.status_code == 200
    data_v4 = res_v4.json()
    assert data_v4["query_ip"] == "192.168.1.100"
    assert "ip_metadata_available" in data_v4

    # Valid IPv6
    res_v6 = client.post("/inspect-ip", json={"ip": "2001:0db8:85a3:0000:0000:8a2e:0370:7334"})
    assert res_v6.status_code == 200
    assert res_v6.json()["query_ip"] == "2001:db8:85a3::8a2e:370:7334"

    # Invalid IP string
    res_bad = client.post("/inspect-ip", json={"ip": "not-an-ip-address"})
    assert res_bad.status_code == 422
    assert "Invalid IP address" in res_bad.json()["detail"]

def test_sample_csv_template():
    res = client.get("/sample-csv-template")
    assert res.status_code == 200
    assert "text/csv" in res.headers["content-type"]
    text = res.text
    assert "Source IP" in text
    assert "Destination IP" in text
    assert "Destination Port" in text

def test_analyze_csv_valid_and_metadata_retention():
    # Load first 2 rows of full 77-feature test fixture with IP metadata
    with open("tests/test_100_flows.csv", "r", encoding="utf-8") as f:
        lines = f.readlines()
    header = lines[0]
    first_two_rows = "".join(lines[:3])

    files = {"file": ("test_flows.csv", io.BytesIO(first_two_rows.encode("utf-8")), "text/csv")}
    res = client.post("/analyze-csv", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["total_flows"] == 2
    assert data["has_ip_metadata"] is True
    assert len(data["results"]) == 2
    assert data["results"][0]["src_ip"] == "192.168.1.10"
    assert data["results"][0]["dst_ip"] == "10.0.0.1"
    assert data["results"][0]["score"] is not None
    assert data["results"][0]["anomaly_confidence"] is not None
    assert "classification_status" in data["results"][0]

    # Verify inspect-ip can query the cached uploaded IP
    inspect_res = client.post("/inspect-ip", json={"ip": "192.168.1.10"})
    assert inspect_res.status_code == 200
    inspect_data = inspect_res.json()
    assert inspect_data["ip_metadata_available"] is True
    assert inspect_data["matching_flows_count"] >= 1
    assert inspect_data["flows"][0]["src_ip"] == "192.168.1.10"
    assert "classification_status" in inspect_data["flows"][0]

def test_attack_type_endpoint():
    res = client.post("/attack-type", json={"row": {"Dst Port": 80, "Tot Fwd Pkts": 1}})
    assert res.status_code == 200
    data = res.json()
    assert "classification_status" in data
    assert "verdict" in data
    assert "verdict_confidence" in data

def test_analyze_csv_errors():
    # Non-CSV extension
    files_txt = {"file": ("bad.txt", io.BytesIO(b"some,data\n1,2"), "text/plain")}
    res_txt = client.post("/analyze-csv", files=files_txt)
    assert res_txt.status_code == 400

    # Empty file
    files_empty = {"file": ("empty.csv", io.BytesIO(b""), "text/csv")}
    res_empty = client.post("/analyze-csv", files=files_empty)
    assert res_empty.status_code == 400

    # Deliberately incomplete CSV: contains only 3 features (rejects with 422 and lists missing features)
    incomplete_csv = (
        "Flow Duration,Tot Fwd Pkts,Dst Port\n"
        "1000,2,80\n"
        "2000,4,443\n"
    )
    files_incomplete = {"file": ("incomplete.csv", io.BytesIO(incomplete_csv.encode("utf-8")), "text/csv")}
    res_incomplete = client.post("/analyze-csv", files=files_incomplete)
    assert res_incomplete.status_code == 422
    err_msg = res_incomplete.json()["detail"]
    assert "Incompatible CSV feature schema" in err_msg
    assert "missing 74 required feature(s)" in err_msg

    # Incompatible schema with non-network columns
    incompatible_csv = "Name,Age,Department,Salary\nAlice,30,HR,60000\nBob,35,IT,75000\n"
    files_bad = {"file": ("employees.csv", io.BytesIO(incompatible_csv.encode("utf-8")), "text/csv")}
    res_bad = client.post("/analyze-csv", files=files_bad)
    assert res_bad.status_code == 422
    assert "Incompatible CSV feature schema" in res_bad.json()["detail"]
    assert "missing 77 required feature(s)" in res_bad.json()["detail"]

def test_analyze_csv_100_rows_end_to_end():
    with open("tests/test_100_flows.csv", "rb") as f:
        file_bytes = f.read()

    files = {"file": ("test_100_flows.csv", io.BytesIO(file_bytes), "text/csv")}
    res = client.post("/analyze-csv", files=files)
    assert res.status_code == 200
    data = res.json()

    assert data["total_flows"] == 100
    assert data["has_ip_metadata"] is True
    assert len(data["results"]) == 100

    # Verify 1-to-1 correspondence and IP retention
    for idx, flow in enumerate(data["results"]):
        assert flow["row_index"] == idx + 1
        expected_src = f"192.168.1.{10 + (idx % 20)}"
        expected_dst = f"10.0.0.{1 + (idx % 5)}"
        assert flow["src_ip"] == expected_src
        assert flow["dst_ip"] == expected_dst
        assert flow["src_port"] == 40000 + idx
        assert flow["protocol"] in ("TCP", "UDP")
        assert flow["score"] is not None
        assert flow["threshold"] is not None
        assert flow["anomaly_confidence"] is not None
        assert flow["classification_status"] in ("Evaluated", "Failed")
        assert flow["verdict"] is not None
        assert flow["verdict_confidence"] is not None

    # Test IP lookup on one of the unique source IPs
    target_ip = "192.168.1.15"
    inspect_res = client.post("/inspect-ip", json={"ip": target_ip})
    assert inspect_res.status_code == 200
    inspect_data = inspect_res.json()
    assert inspect_data["ip_metadata_available"] is True
    assert inspect_data["matching_flows_count"] == 5  # 100 / 20 = 5 matches
    assert all(f["src_ip"] == target_ip for f in inspect_data["flows"])
