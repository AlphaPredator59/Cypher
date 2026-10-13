import ipaddress
import io
import time
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Response
from fastapi.middleware.cors import CORSMiddleware
import compute
import json
from llm import summarize_alert, generate_fallback_summary
import database
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
import random
import os

logger = logging.getLogger("cypher.api")

CSV_FOLDER = os.path.join(
    os.path.dirname(__file__),
    "solarmainframe",
    "ids-intrusion-csv",
    "versions",
    "1"
)
from ml.features import CICIDS_COLUMN_MAPPING

# In-memory session cache for traffic with IP metadata (from uploaded CSVs or ingested telemetry)
LAST_ANALYZED_TRAFFIC: List[Dict[str, Any]] = []

def get_random_row_from_csv():
    import glob

    csv_files = glob.glob(os.path.join(CSV_FOLDER, "*.csv"))
    if not csv_files:
        raise HTTPException(status_code=400, detail="No CSV files found in dataset folder")

    chosen_file = random.choice(csv_files)
    df = pd.read_csv(chosen_file, nrows=500)
    df.columns = df.columns.str.strip()
    df.rename(columns=CICIDS_COLUMN_MAPPING, inplace=True)

    if "Label" in df.columns:
        df = df.drop(columns=["Label"])

    df = df.replace([float("inf"), float("-inf")], 0).fillna(0)
    random_row = df.sample(1).to_dict(orient="records")[0]
    return random_row


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        database.init_db()
        logger.info("CYPHER database initialized successfully.")
    except Exception as exc:
        logger.warning("CYPHER database initialization warning on startup: %s", exc)
    yield


app = FastAPI(title="CYPHER IDS API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8001",
        "http://127.0.0.1:8001",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost",
        "http://127.0.0.1",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RowIn(BaseModel):
    row: Dict[str, Any]


class BatchIn(BaseModel):
    rows: List[Dict[str, Any]]


class InspectIPIn(BaseModel):
    ip: str


def validate_input_row(row: Dict[str, Any], row_index: Optional[int] = None) -> None:
    """
    Validates that a flow dictionary contains valid finite numeric feature values.
    Rejects malformed values (unparseable strings, NaN, Inf) with HTTP 422.
    Non-feature metadata (like src_ip, dst_ip, protocol, flow_id, id) is skipped.
    """
    prefix = f"Row {row_index}: " if row_index is not None else ""
    if not isinstance(row, dict) or not row:
        raise HTTPException(
            status_code=422,
            detail=f"{prefix}Input row must be a non-empty dictionary containing network flow features."
        )

    metadata_keys = {
        "src_ip", "dst_ip", "source_ip", "destination_ip", "Source IP", "Destination IP",
        "protocol", "Protocol", "proto", "Proto",
        "src_port", "Source Port", "Src Port",
        "flow_id", "Flow ID", "FlowId", "id", "ID", "timestamp", "Timestamp",
        "Label", "_source_file", "row_index"
    }

    malformed = []
    non_finite = []

    for k, v in row.items():
        if k in metadata_keys:
            continue
        if v is None:
            continue
        try:
            fval = float(v)
            if not np.isfinite(fval):
                non_finite.append(f"{k}={v}")
        except (ValueError, TypeError):
            malformed.append(f"{k}='{v}'")

    if non_finite:
        raise HTTPException(
            status_code=422,
            detail=f"{prefix}Non-finite feature value(s) detected: {', '.join(non_finite[:5])}. Features must be finite numbers."
        )
    if malformed:
        raise HTTPException(
            status_code=422,
            detail=f"{prefix}Malformed non-numeric feature value(s) detected: {', '.join(malformed[:5])}. Features must be numeric."
        )


def _extract_flow_persistence_dict(r: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "flow_id": str(row.get("flow_id") or row.get("id") or ""),
        "src_ip": row.get("src_ip") or row.get("Source IP") or row.get("Src IP"),
        "dst_ip": row.get("dst_ip") or row.get("Destination IP") or row.get("Dst IP"),
        "src_port": row.get("src_port") or row.get("Source Port") or row.get("Src Port"),
        "dst_port": row.get("dst_port") or row.get("Destination Port") or row.get("Dst Port"),
        "protocol": row.get("protocol") or row.get("Protocol") or "TCP/IP",
        "score": r["score"],
        "threshold": r["threshold"],
        "is_anomaly": r["is_anomaly"],
        "anomaly_confidence": r.get("anomaly_confidence"),
        "attack_type": r.get("attack_type"),
        "attack_confidence": r.get("attack_confidence"),
        "classification_status": r.get("classification_status"),
        "verdict": r.get("verdict"),
        "verdict_confidence": r.get("verdict_confidence"),
        "detection_source": r.get("detection_source"),
        "alert_type": r.get("alert_type"),
    }


def _safe_persist_flow(record: Dict[str, Any]) -> None:
    try:
        database.save_flow_record(record)
    except Exception as exc:
        logger.warning("Database persistence failure (ignored to preserve response): %s", exc)


def _safe_persist_flows(records: List[Dict[str, Any]]) -> None:
    try:
        database.save_flow_records(records)
    except Exception as exc:
        logger.warning("Database batch persistence failure (ignored to preserve response): %s", exc)


@app.post("/analyze-batch",
          summary="Analyze a batch of rows (full model output)",
          description="Runs the model on multiple input rows in one request.")
def analyze_batch(req: BatchIn):
    if not req.rows:
        raise HTTPException(status_code=400, detail="Batch rows list cannot be empty.")
    for idx, r in enumerate(req.rows):
        validate_input_row(r, row_index=idx + 1)

    out = compute.predict(req.rows)
    persist_records = [_extract_flow_persistence_dict(res, row) for res, row in zip(out["results"], req.rows)]
    _safe_persist_flows(persist_records)
    return out


@app.post("/predict",
          summary="Predict anomaly + attack info for a single row",
          description="Runs the model for a single row and returns calibrated confidence and verdict.")
def predict_one(req: RowIn):
    validate_input_row(req.row)
    out = compute.predict([req.row])
    result0 = out["results"][0]

    # Persist prediction
    flow_record = _extract_flow_persistence_dict(result0, req.row)
    _safe_persist_flow(flow_record)

    return {
        "is_anomaly": result0["is_anomaly"],
        "score": result0["score"],
        "threshold": result0["threshold"],
        "anomaly_confidence": result0.get("anomaly_confidence"),
        "attack_type": result0["attack_type"],
        "attack_confidence": result0.get("attack_confidence"),
        "classifier_probability": result0.get("classifier_probability", result0.get("attack_confidence")),
        "classification_status": result0.get("classification_status"),
        "verdict": result0.get("verdict"),
        "verdict_confidence": result0.get("verdict_confidence"),
        "detection_source": result0.get("detection_source"),
        "alert_type": result0.get("alert_type"),
        "top_contributors": result0["top_contributors"],
        "input_issues": out.get("input_issues", {})
    }


@app.post("/attack-type",
          summary="Predict attack type for a single row",
          description="Runs the model for a single row and returns attack classification fields.")
def predict_attack_type(req: RowIn):
    validate_input_row(req.row)
    out = compute.predict([req.row])
    r = out["results"][0]
    return {
        "attack_type": r["attack_type"],
        "attack_confidence": r.get("attack_confidence"),
        "classifier_probability": r.get("classifier_probability", r.get("attack_confidence")),
        "classification_status": r.get("classification_status"),
        "verdict": r.get("verdict"),
        "verdict_confidence": r.get("verdict_confidence"),
        "detection_source": r.get("detection_source"),
        "alert_type": r.get("alert_type"),
    }


@app.post("/anomaly-flag",
          summary="Predict only anomaly flag + score for a single row",
          description="Runs the model for a single row and returns anomaly-related fields.")
def get_anomaly_flag(req: RowIn):
    validate_input_row(req.row)
    out = compute.predict([req.row])
    r = out["results"][0]
    return {
        "is_anomaly": r["is_anomaly"],
        "score": r["score"],
        "threshold": r["threshold"],
        "anomaly_confidence": r.get("anomaly_confidence"),
        "detection_source": r.get("detection_source"),
        "alert_type": r.get("alert_type"),
    }


@app.post("/analyze",
          summary="Analyze single row (full model_output + curated llm_payload)",
          description="Runs the model for a single row and returns full model_output and llm_payload.")
def analyze(req: RowIn):
    validate_input_row(req.row)
    out = compute.predict([req.row])
    r = out["results"][0]

    # Persist analysis
    flow_record = _extract_flow_persistence_dict(r, req.row)
    _safe_persist_flow(flow_record)

    llm_payload = {
        "is_anomaly": r["is_anomaly"],
        "score": r["score"],
        "threshold": r["threshold"],
        "anomaly_confidence": r.get("anomaly_confidence"),
        "attack_type": r["attack_type"],
        "attack_confidence": r.get("attack_confidence"),
        "classification_status": r.get("classification_status"),
        "verdict": r.get("verdict"),
        "verdict_confidence": r.get("verdict_confidence"),
        "detection_source": r.get("detection_source"),
        "alert_type": r.get("alert_type"),
        "top_contributors": r["top_contributors"],
        "input_issues": out.get("input_issues", {}),
    }

    return {
        "model_output": r,
        "llm_payload": llm_payload
    }


@app.post("/summarize")
def summarize(req: dict):
    if not isinstance(req, dict) or "llm_payload" not in req or "model_output" not in req:
        raise HTTPException(status_code=400, detail="Need llm_payload and model_output")

    try:
        return summarize_alert(req["llm_payload"], req["model_output"])
    except Exception as exc:
        logger.warning("Unhandled exception in /summarize: %s", exc)
        return generate_fallback_summary(
            llm_payload=req["llm_payload"],
            model_output=req["model_output"],
            reason=f"Summarization error: {str(exc)[:100]}",
            provider_status="error"
        )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/random-analyze")
def random_analyze():
    row = get_random_row_from_csv()
    out = compute.predict([row])
    result0 = out["results"][0]

    llm_payload = {
        "is_anomaly": result0["is_anomaly"],
        "score": result0["score"],
        "threshold": result0["threshold"],
        "anomaly_confidence": result0.get("anomaly_confidence"),
        "attack_type": result0["attack_type"],
        "attack_confidence": result0.get("attack_confidence"),
        "classification_status": result0.get("classification_status"),
        "verdict": result0.get("verdict"),
        "verdict_confidence": result0.get("verdict_confidence"),
        "detection_source": result0.get("detection_source"),
        "top_contributors": result0["top_contributors"],
        "input_issues": out.get("input_issues", {}),
    }

    return {
        "model_output": result0,
        "llm_payload": llm_payload
    }


def _validate_ip_string(ip_str: str) -> str:
    cleaned = ip_str.strip()
    try:
        obj = ipaddress.ip_address(cleaned)
        return str(obj)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid IP address: '{ip_str}'. Must be a valid IPv4 or IPv6 address."
        )


@app.get("/inspect-ip", summary="Inspect traffic flows associated with an IP address (GET)")
@app.post("/inspect-ip", summary="Inspect traffic flows associated with an IP address (POST)")
def inspect_ip_endpoint(req: Optional[InspectIPIn] = None, ip: Optional[str] = Query(None)):
    query_ip = (req.ip if req and req.ip else ip)
    if not query_ip:
        raise HTTPException(status_code=400, detail="Missing required IP address query or body.")

    clean_ip = _validate_ip_string(query_ip)

    # 1. Search active session report first if available
    matching_flows = []
    if LAST_ANALYZED_TRAFFIC:
        for flow in LAST_ANALYZED_TRAFFIC:
            if flow.get("src_ip") == clean_ip or flow.get("dst_ip") == clean_ip:
                matching_flows.append(flow)

    # 2. Fall back to persisted database audit logs
    if not matching_flows:
        try:
            matching_flows = database.query_flows_by_ip(clean_ip, limit=100)
        except Exception as exc:
            logger.warning("Database query failed during /inspect-ip: %s", exc)

    if matching_flows:
        return {
            "query_ip": clean_ip,
            "ip_metadata_available": True,
            "matching_flows_count": len(matching_flows),
            "message": f"Found {len(matching_flows)} matching traffic flow(s) for IP {clean_ip}.",
            "flows": matching_flows[:100]
        }

    # 3. Check the disk dataset files for IP columns
    # Standard CICIDS2017 feature CSVs do not retain Source/Destination IP headers.
    return {
        "query_ip": clean_ip,
        "ip_metadata_available": False,
        "matching_flows_count": 0,
        "message": (
            "The active server dataset store (CICIDS2017) contains aggregate statistical network flow metrics "
            "without packet IP header attribution. To inspect specific IP addresses, upload a CSV containing "
            "'Source IP' or 'Destination IP' columns using 'Upload CSV Report', or download our sample CSV template."
        ),
        "flows": []
    }


@app.get("/audit-logs", summary="Query persisted flow audit records from database")
def get_audit_logs(
    limit: int = Query(50, ge=1, le=500),
    anomaly_only: bool = Query(False),
    detection_source: Optional[str] = Query(None),
    alert_type: Optional[str] = Query(None),
    ip: Optional[str] = Query(None)
):
    try:
        if ip:
            clean_ip = _validate_ip_string(ip)
            flows = database.query_flows_by_ip(clean_ip, limit=limit)
        else:
            flows = database.query_recent_flows(
                limit=limit,
                anomaly_only=anomaly_only,
                detection_source=detection_source,
                alert_type=alert_type
            )
        stats = database.get_flow_stats()
        return {
            "count": len(flows),
            "stats": stats,
            "flows": flows
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Failed to retrieve audit logs: %s", exc)
        return {"count": 0, "stats": {"total": 0, "anomalous": 0, "benign": 0}, "flows": []}


@app.get("/audit-stats", summary="Get aggregate flow persistence statistics")
def get_audit_stats():
    try:
        return database.get_flow_stats()
    except Exception as exc:
        logger.warning("Failed to retrieve audit stats: %s", exc)
        return {"total": 0, "anomalous": 0, "benign": 0}


@app.post("/analyze-csv", summary="Upload and analyze a network traffic CSV report")
async def analyze_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file type. File must have a .csv extension.")

    # Read content with 50MB size limit
    content_bytes = await file.read()
    if not content_bytes:
        raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")

    max_size = 50 * 1024 * 1024
    if len(content_bytes) > max_size:
        raise HTTPException(status_code=413, detail="File too large. Maximum supported CSV size is 50MB.")

    # Decode
    try:
        content_str = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            content_str = content_bytes.decode("latin-1")
        except Exception:
            raise HTTPException(status_code=400, detail="Unable to decode CSV. Ensure file is UTF-8 or ASCII encoded.")

    try:
        df = pd.read_csv(io.StringIO(content_str))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV contains 0 rows.")

    df.columns = df.columns.str.strip()

    # Extract IP and connection metadata columns
    ip_col_candidates_src = ["Source IP", "Src IP", "Source_IP", "src_ip", "Source_Ip"]
    ip_col_candidates_dst = ["Destination IP", "Dst IP", "Destination_IP", "dst_ip", "Destination_Ip"]
    port_col_candidates_src = ["Source Port", "Src Port", "Source_Port", "src_port", "Sport"]
    port_col_candidates_dst = ["Destination Port", "Dst Port", "Destination_Port", "dst_port", "Dport"]
    proto_col_candidates = ["Protocol", "protocol", "Proto", "proto"]

    id_col_candidates = ["Flow ID", "flow_id", "FlowId", "id", "ID", "Timestamp", "timestamp", "_source_file"]

    src_ip_col = next((c for c in ip_col_candidates_src if c in df.columns), None)
    dst_ip_col = next((c for c in ip_col_candidates_dst if c in df.columns), None)
    src_port_col = next((c for c in port_col_candidates_src if c in df.columns), None)
    dst_port_col = next((c for c in port_col_candidates_dst if c in df.columns), None)
    proto_col = next((c for c in proto_col_candidates if c in df.columns), None)
    id_col = next((c for c in id_col_candidates if c in df.columns), None)

    has_ip_metadata = bool(src_ip_col or dst_ip_col)

    # Extract metadata arrays before renaming/dropping
    src_ips = df[src_ip_col].astype(str).tolist() if src_ip_col else ["N/A"] * len(df)
    dst_ips = df[dst_ip_col].astype(str).tolist() if dst_ip_col else ["N/A"] * len(df)
    src_ports = df[src_port_col].tolist() if src_port_col else [None] * len(df)
    dst_ports = df[dst_port_col].tolist() if dst_port_col else [None] * len(df)
    protocols = df[proto_col].astype(str).tolist() if proto_col else ["TCP/IP"] * len(df)

    # Prepare features for the ML model
    model_df = df.copy()

    # Handle Fwd Header Length fallback if Fwd Header Length is missing but Fwd Header Length.1 is present
    if "Fwd Header Length" not in model_df.columns and "Fwd Header Len" not in model_df.columns and "Fwd Header Length.1" in model_df.columns:
        model_df["Fwd Header Length"] = model_df["Fwd Header Length.1"]

    # Handle Destination Port collision if Dst Port is already in model_df
    if "Dst Port" in model_df.columns and "Destination Port" in model_df.columns:
        model_df = model_df.drop(columns=["Destination Port"])

    model_df.rename(columns=CICIDS_COLUMN_MAPPING, inplace=True)

    # De-duplicate any duplicate columns if present
    if model_df.columns.duplicated().any():
        model_df = model_df.loc[:, ~model_df.columns.duplicated()]

    # Drop non-feature metadata and label columns so they don't contaminate model inputs
    # Note: Dst Port is preserved because it is explicitly part of the 77 trained VAE features
    cols_to_drop = [
        c for c in [src_ip_col, dst_ip_col, src_port_col, proto_col, id_col, "Label", "_source_file", "Fwd Header Length.1"]
        if c and c in model_df.columns and c != "Dst Port"
    ]
    if cols_to_drop:
        model_df = model_df.drop(columns=cols_to_drop)

    # Priority 1: Strict feature validation - reject incomplete inputs with HTTP 422
    required_features = compute.PIPELINE.vae_feature_names
    missing_features = [c for c in required_features if c not in model_df.columns]
    if missing_features:
        sample_missing = missing_features[:10]
        suffix = f" (and {len(missing_features) - 10} more)" if len(missing_features) > 10 else ""
        raise HTTPException(
            status_code=422,
            detail=(
                f"Incompatible CSV feature schema: missing {len(missing_features)} required feature(s): "
                f"{', '.join(sample_missing)}{suffix}. "
                f"The VAE/MoE pipeline requires all 77 CICIDS2017 flow metrics. "
                f"Please verify column headers or download the compliant sample CSV template via /sample-csv-template."
            )
        )

    # Strict numeric conversion: reject missing/NaN/null values without zero-imputation
    for c in required_features:
        model_df[c] = pd.to_numeric(model_df[c], errors="coerce")

    nan_rows_mask = model_df[required_features].isna().any(axis=1)
    if nan_rows_mask.any():
        nan_cols = [c for c in required_features if model_df[c].isna().any()]
        sample_nan = nan_cols[:8]
        suffix = f" (and {len(nan_cols) - 8} more)" if len(nan_cols) > 8 else ""
        raise HTTPException(
            status_code=422,
            detail=(
                f"Incompatible CSV data: {int(nan_rows_mask.sum())} row(s) contain missing, unpopulated, or non-numeric values in required feature(s): "
                f"{', '.join(sample_nan)}{suffix}. "
                f"Silent zero-imputation is disabled for model integrity. All 77 features must contain valid finite numbers."
            )
        )

    # Reject infinite values
    inf_rows_mask = np.isinf(model_df[required_features].to_numpy()).any(axis=1)
    if inf_rows_mask.any():
        raise HTTPException(
            status_code=422,
            detail=(
                f"Incompatible CSV data: {int(inf_rows_mask.sum())} row(s) contain infinite values (inf/-inf). "
                f"Please clean infinite values prior to upload."
            )
        )

    # Extract clean records strictly corresponding to required features
    clean_feature_df = model_df[required_features]
    records = clean_feature_df.to_dict(orient="records")

    # Run batched inference in chunks to keep memory bounded
    t0 = time.perf_counter()
    chunk_size = 500
    all_results = []
    for i in range(0, len(records), chunk_size):
        chunk = records[i:i + chunk_size]
        out = compute.predict(chunk)
        all_results.extend(out["results"])
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    total_flows = len(all_results)
    normal_flows = sum(1 for r in all_results if not r["is_anomaly"])
    anomalous_flows = sum(1 for r in all_results if r["is_anomaly"])

    attack_breakdown: Dict[str, int] = {}
    for r in all_results:
        if r["is_anomaly"]:
            lbl = r.get("verdict") or r.get("attack_type") or "Anomaly"
            attack_breakdown[lbl] = attack_breakdown.get(lbl, 0) + 1

    # Enrich results with metadata and original feature summary
    enriched = []
    for idx, (res, orig) in enumerate(zip(all_results, records)):
        flow_entry = {
            "row_index": idx + 1,
            "src_ip": src_ips[idx],
            "dst_ip": dst_ips[idx],
            "src_port": src_ports[idx],
            "dst_port": dst_ports[idx] if dst_ports[idx] is not None else orig.get("Dst Port"),
            "protocol": protocols[idx],
            "flow_duration": orig.get("Flow Duration", 0),
            "tot_fwd_pkts": orig.get("Tot Fwd Pkts", 0),
            "tot_bwd_pkts": orig.get("Tot Bwd Pkts", 0),
            "flow_byts_s": orig.get("Flow Byts/s", 0),
            "flow_pkts_s": orig.get("Flow Pkts/s", 0),
            "score": res["score"],
            "threshold": res["threshold"],
            "is_anomaly": res["is_anomaly"],
            "anomaly_confidence": res.get("anomaly_confidence"),
            "attack_type": res["attack_type"],
            "attack_confidence": res.get("attack_confidence"),
            "classification_status": res.get("classification_status"),
            "verdict": res.get("verdict"),
            "verdict_confidence": res.get("verdict_confidence"),
            "detection_source": res.get("detection_source"),
            "top_contributors": res.get("top_contributors", []),
            "raw_features": {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in orig.items() if k in compute.PIPELINE.vae_feature_names[:25]}
        }
        enriched.append(flow_entry)

    # Persist flows to SQLite audit log
    _safe_persist_flows(enriched)

    # Cache for subsequent IP lookup queries
    global LAST_ANALYZED_TRAFFIC
    LAST_ANALYZED_TRAFFIC = enriched

    return {
        "filename": file.filename,
        "total_flows": total_flows,
        "normal_flows": normal_flows,
        "anomalous_flows": anomalous_flows,
        "attack_breakdown": attack_breakdown,
        "processing_time_ms": elapsed_ms,
        "has_ip_metadata": has_ip_metadata,
        "results": enriched
    }


@app.get("/sample-csv-template", summary="Download a sample network traffic CSV template")
def sample_csv_template():
    fixture_path = os.path.join(os.path.dirname(__file__), "tests", "test_100_flows.csv")
    if os.path.exists(fixture_path):
        df_sample = pd.read_csv(fixture_path, nrows=10)
        # Drop Label column if present so users get a clean unlabelled production input template
        if "Label" in df_sample.columns:
            df_sample = df_sample.drop(columns=["Label"])
    else:
        # Fallback minimal compliant dataframe
        df_sample = pd.DataFrame(
            columns=["Source IP", "Destination IP", "Source Port", "Protocol"] + compute.PIPELINE.vae_feature_names
        )

    csv_str = df_sample.to_csv(index=False)
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="cypher_traffic_sample.csv"'}
    )