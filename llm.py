"""
CYPHER Intrusion Detection System — LLM Alert Summarizer Module.
Integrates Google Gemini (gemini-1.5-pro) for incident explanation,
with production-grade error resilience, bounded timeouts, and
truthful non-fabricating fallbacks when the provider is unavailable.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("cypher.llm")

SYSTEM_PROMPT = """You are a cybersecurity SOC analyst assistant. You will receive JSON with keys: "llm_payload" 
    and "model_output". Use "llm_payload" as the primary source.
    Use "model_output" only if it adds helpful detail. Do not invent information. describing one
    network flow analyzed by an anomaly detector (VAE) and, if anomalous, an attack classifier (MoE).
    Your job:
    1) Explain clearly what is suspicious and why, using only the fields provided.
    2) If is_anomaly=false, explain why it looks normal and suggest minimal monitoring steps.
    3) If is_anomaly=true, interpret attack_type and top_contributors to describe likely behavior.
    4) Provide practical countermeasures and next investigative steps.
    Rules:
    - Output MUST be valid JSON only (no markdown, no extra text).
    - Do NOT invent IPs, ports, protocols, or events not present.
    - If confidence is low or data is incomplete, say so explicitly.
    - Keep it concise and dashboard-friendly.
    """

USER_TEMPLATE = """Analyze this alert payload and produce the required JSON response.

ALERT_PAYLOAD:
{payload}

Return JSON with this exact schema:
{schema}
"""

SCHEMA = {
    "severity": "low|medium|high|critical",
    "summary": "string",
    "what_is_wrong": ["string"],
    "evidence": [{"feature": "string", "signal": "high_recon_error", "note": "string"}],
    "attack_assessment": {"predicted_type": "string", "confidence": 0.0, "interpretation": "string"},
    "recommended_actions": {"immediate": ["string"], "short_term": ["string"], "long_term": ["string"]},
    "questions_to_ask": ["string"],
    "limitations": ["string"]
}

_CLIENT_CACHE: Optional[Any] = None


def _content_to_text(content: Any) -> str:
    """Converts langchain response content (str or list of dicts/text) to a unified string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif hasattr(item, "text"):
                parts.append(str(getattr(item, "text")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def get_llm_client(timeout_seconds: float = 15.0, model: Optional[str] = None) -> Optional[Any]:
    """
    Lazily creates and caches the Google Gemini client.
    Returns None if GOOGLE_API_KEY is missing, empty, or client initialization fails.
    Avoids crashing at module import time when API credentials are not set.
    """
    global _CLIENT_CACHE
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return None

    if _CLIENT_CACHE is not None:
        return _CLIENT_CACHE

    target_model = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        _CLIENT_CACHE = ChatGoogleGenerativeAI(
            model=target_model,
            api_key=api_key,
            timeout=timeout_seconds,
        )
        return _CLIENT_CACHE
    except Exception as exc:
        logger.warning("Failed to initialize ChatGoogleGenerativeAI: %s", exc)
        return None


def _extract_json(text_or_content: Any) -> dict:
    """Parses JSON response, handling list-based content and markdown code fences if present."""
    text = _content_to_text(text_or_content).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
            if text.startswith("json"):
                text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e != -1 and e > s:
            return json.loads(text[s:e + 1])
        raise


def generate_fallback_summary(
    llm_payload: dict,
    model_output: dict,
    reason: str,
    provider_status: str = "unavailable"
) -> dict:
    """
    Generates a truthful, schema-compliant fallback report when the LLM is unavailable.
    CRITICAL: Never fabricates an AI-generated explanation.
    Clearly discloses provider unavailability while presenting factual deterministic model findings.
    """
    is_anomaly = bool(llm_payload.get("is_anomaly", False))
    verdict = str(llm_payload.get("verdict") or ("Anomalous Flow" if is_anomaly else "Normal Traffic"))
    attack_type = str(llm_payload.get("attack_type") or "Normal Traffic")
    score = llm_payload.get("score")
    threshold = llm_payload.get("threshold", 38.50959)
    detection_source = str(llm_payload.get("detection_source") or "unknown")
    verdict_conf = llm_payload.get("verdict_confidence")
    top_contribs = llm_payload.get("top_contributors") or []

    # Deterministic severity assignment
    if not is_anomaly:
        severity = "low"
    elif detection_source in ("dual_confirmed", "classifier_attack"):
        severity = "high" if any(k in attack_type for k in ("DoS", "DDoS", "Patator", "Bot")) else "medium"
    else:
        severity = "medium"

    evidence = []
    for c in top_contribs[:5]:
        if isinstance(c, dict):
            feat = c.get("feature", "unknown")
            sqe = c.get("sq_error", 0.0)
            sqe_str = f"{sqe:.2f}" if isinstance(sqe, (int, float)) else str(sqe)
            evidence.append({
                "feature": str(feat),
                "signal": "high_recon_error",
                "note": f"Reconstruction squared error: {sqe_str}"
            })

    what_is_wrong = []
    if is_anomaly:
        what_is_wrong.append(f"Deterministic IDS verdict: {verdict} (trigger: {detection_source}).")
        if score is not None:
            score_val = float(score) if isinstance(score, (int, float)) else 0.0
            thresh_val = float(threshold) if isinstance(threshold, (int, float)) else 38.51
            what_is_wrong.append(f"VAE reconstruction score is {score_val:.2f} relative to baseline threshold {thresh_val:.2f}.")
    else:
        what_is_wrong.append("Flow metrics reside within normal baseline boundaries.")

    conf_val = float(verdict_conf) if verdict_conf is not None else 0.0

    return {
        "severity": severity,
        "summary": (
            f"AI incident summarization is unavailable ({reason}). "
            "Deterministic model detection and classification remain fully operational."
        ),
        "what_is_wrong": what_is_wrong,
        "evidence": evidence,
        "attack_assessment": {
            "predicted_type": attack_type,
            "confidence": round(conf_val, 4),
            "interpretation": (
                "Deterministic model output only. No generative AI explanation was fabricated."
            )
        },
        "recommended_actions": {
            "immediate": [
                "Review the deterministic verdict and confidence score in the telemetry console.",
                "Inspect source and destination IP addresses in the connection context."
            ],
            "short_term": [
                "Correlate IP flow history with firewall and authentication logs.",
                "Configure or verify GOOGLE_API_KEY if AI narrative summaries are required."
            ],
            "long_term": [
                "Apply edge firewall rules if anomalous pattern persists across subnets.",
                "Maintain continuous monitoring of statistical flow deviations."
            ]
        },
        "questions_to_ask": [
            "Is the source IP expected to generate traffic to the destination port?",
            "Are related traffic spikes observed in concurrent network flows?"
        ],
        "limitations": [
            f"Generative AI service unavailable: {reason}.",
            "This report contains verified deterministic engine findings only."
        ],
        "provider_status": provider_status
    }


def summarize_alert(llm_payload: dict, model_output: dict, timeout_seconds: float = 15.0) -> dict:
    """
    Summarizes an alert using Google Gemini, or returns a graceful fallback report.
    Never crashes when the API key is missing, network times out, quota is exceeded,
    or the provider returns malformed output.
    """
    client = get_llm_client(timeout_seconds=timeout_seconds)
    if client is None:
        return generate_fallback_summary(
            llm_payload=llm_payload,
            model_output=model_output,
            reason="GOOGLE_API_KEY is not configured in the server environment",
            provider_status="missing_api_key"
        )

    context = {"llm_payload": llm_payload, "model_output": model_output}
    messages = [
        ("system", SYSTEM_PROMPT),
        ("user", USER_TEMPLATE.format(
            payload=json.dumps(context, indent=2),
            schema=json.dumps(SCHEMA, indent=2)
        ))
    ]

    try:
        resp = client.invoke(messages)
        return _extract_json(resp.content)
    except (TimeoutError, Exception) as exc:
        err_msg = str(exc).lower()
        err_type = type(exc).__name__

        # Identify specific failure causes
        if "timeout" in err_msg or "timed out" in err_msg or "timeouterror" in err_type.lower():
            reason = f"Gemini provider request timed out after {timeout_seconds}s"
            status = "timeout"
        elif "not_found" in err_msg or "404" in err_msg or "not found" in err_msg:
            reason = f"Gemini model not found or retired: {str(exc)[:120]}"
            status = "model_not_found"
        elif "resource_exhausted" in err_msg or "429" in err_msg or "quota" in err_msg:
            reason = "Gemini provider rate limit or quota exceeded (HTTP 429)"
            status = "rate_limited"
        elif "permission_denied" in err_msg or "403" in err_msg or "401" in err_msg or "invalid api key" in err_msg:
            reason = "Gemini provider authentication failure (invalid or unauthorized API key)"
            status = "auth_error"
        elif "connect" in err_msg or "socket" in err_msg or "dns" in err_msg or "network" in err_msg:
            reason = f"Gemini provider network connection failure ({str(exc)[:120]})"
            status = "network_error"
        elif isinstance(exc, json.JSONDecodeError) or "jsondecodeerror" in err_type.lower():
            reason = "Gemini provider returned non-JSON or malformed output"
            status = "malformed_response"
        else:
            reason = f"Gemini provider error ({err_type}: {str(exc)[:120]})"
            status = "provider_error"

        logger.warning("Gemini summarization failed gracefully: %s", reason)
        return generate_fallback_summary(
            llm_payload=llm_payload,
            model_output=model_output,
            reason=reason,
            provider_status=status
        )


# Backward-compatible LLM reference (lazy proxy)
class _LazyLLM:
    def invoke(self, *args, **kwargs):
        c = get_llm_client()
        if c is None:
            raise RuntimeError("GOOGLE_API_KEY is not configured")
        return c.invoke(*args, **kwargs)

LLM = _LazyLLM()
