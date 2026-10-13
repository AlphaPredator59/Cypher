"""
CYPHER IDS ML Integration Adapter.
Canonical machine learning implementations are consolidated under root ml/ (ml.pipeline).
This adapter preserves backward-compatible interfaces for FastAPI routes and test harnesses.
"""
from ml.pipeline import (
    SmallMLP,
    DeepMLP,
    GatingNetwork,
    MoEPredictor,
    VAEScorer,
    calibrate_anomaly_confidence,
    ThreatPipeline,
    PIPELINE,
    vae,
    moe,
    vae_feature_names,
    validate_rows,
    predict,
)

__all__ = [
    "SmallMLP",
    "DeepMLP",
    "GatingNetwork",
    "MoEPredictor",
    "VAEScorer",
    "calibrate_anomaly_confidence",
    "ThreatPipeline",
    "PIPELINE",
    "vae",
    "moe",
    "vae_feature_names",
    "validate_rows",
    "predict",
]
