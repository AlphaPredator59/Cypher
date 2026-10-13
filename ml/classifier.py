"""
Supervised Multi-Class Attack Classifier for CYPHER IDS.

Implements the 12-class XGBoost inference interface using pre-trained
weights from ml/models/xgb_baseline.json and label encoding from
ml/models/label_encoder.pkl.

Supported attack taxonomy (12 classes):
- BENIGN
- Bot
- DDoS
- DoS GoldenEye
- DoS Hulk
- DoS Slowhttptest
- DoS slowloris
- FTP-Patator
- PortScan
- SSH-Patator
- Web Attack Brute Force
- Web Attack XSS
"""

import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, List, Any, Tuple, Optional, Union

from ml.features import XGB_FEATURE_NAMES, normalize_flow_for_xgb

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "xgb_baseline.json")
DEFAULT_ENCODER_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "label_encoder.pkl")


class XGBAttackClassifier:
    """
    Production interface for the 12-class XGBoost attack classifier.
    Exposes identical predict_batch() signature to MoEPredictor for seamless integration.
    """

    def __init__(self, model_path: Optional[str] = None, encoder_path: Optional[str] = None):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.encoder_path = encoder_path or DEFAULT_ENCODER_PATH

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"XGBoost model artifact not found at {self.model_path}")
        if not os.path.exists(self.encoder_path):
            raise FileNotFoundError(f"Label encoder artifact not found at {self.encoder_path}")

        # Load booster
        self.booster = xgb.Booster()
        self.booster.load_model(self.model_path)

        # Load label encoder
        self.label_encoder = joblib.load(self.encoder_path)
        self.classes: List[str] = list(self.label_encoder.classes_)
        self.num_classes: int = len(self.classes)
        self.feature_names: List[str] = XGB_FEATURE_NAMES

        # Validate that booster feature names match expected schema
        booster_feats = self.booster.feature_names
        if booster_feats and booster_feats != self.feature_names:
            raise ValueError(
                f"Booster feature mismatch: expected {len(self.feature_names)} features, booster has {len(booster_feats)}"
            )

    def prepare_matrix(self, rows: List[Dict[str, Any]], strict: bool = False) -> np.ndarray:
        """
        Converts input rows into an (N, 78) float32 numpy matrix.
        Validates column ordering, handles both raw and mapped names, and enforces
        strict non-finite / missing value checks if strict=True.
        """
        if not rows:
            return np.empty((0, len(self.feature_names)), dtype=np.float32)

        mat = np.zeros((len(rows), len(self.feature_names)), dtype=np.float32)
        for i, r in enumerate(rows):
            mat[i] = normalize_flow_for_xgb(r, strict=strict)
        return mat

    def predict_proba(self, X: Union[np.ndarray, List[Dict[str, Any]]], strict: bool = False) -> np.ndarray:
        """
        Computes 12-class probability distribution.
        Returns shape (N, 12).
        """
        if isinstance(X, list):
            X_mat = self.prepare_matrix(X, strict=strict)
        else:
            X_mat = X

        if len(X_mat) == 0:
            return np.empty((0, self.num_classes), dtype=np.float32)

        dmat = xgb.DMatrix(X_mat, feature_names=self.feature_names)
        probs = self.booster.predict(dmat)
        return probs.astype(np.float32)

    def predict(self, rows: List[Dict[str, Any]], strict: bool = False) -> Tuple[List[str], np.ndarray]:
        """
        Predicts attack class and confidence for a list of input flow dictionaries.
        Returns:
            predicted_labels: List of class strings (e.g. 'DDoS', 'PortScan', 'BENIGN')
            confidences: 1D array of float confidences (maximum probability)
        """
        if not rows:
            return [], np.array([], dtype=np.float32)

        probs = self.predict_proba(rows, strict=strict)
        pred_indices = np.argmax(probs, axis=1)
        labels = [self.classes[idx] for idx in pred_indices]
        confs = np.max(probs, axis=1)
        return labels, confs

    def predict_batch(self, rows: List[Dict[str, Any]], strict: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Drop-in replacement for MoEPredictor.predict_batch().
        Returns:
            labels: 1D numpy array of predicted label strings
            probs: 2D numpy array of class probabilities, shape (N, 12)
        """
        if not rows:
            return np.array([]), np.empty((0, self.num_classes), dtype=np.float32)

        probs = self.predict_proba(rows, strict=strict)
        pred_indices = np.argmax(probs, axis=1)
        labels = np.array([self.classes[idx] for idx in pred_indices])
        return labels, probs

    def predict_single(self, row: Dict[str, Any], strict: bool = False) -> Dict[str, Any]:
        """
        Detailed single-flow prediction returning winning class, confidence,
        and full per-class probability distribution.
        """
        probs = self.predict_proba([row], strict=strict)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = self.classes[pred_idx]
        confidence = float(probs[pred_idx])

        prob_dist = {lbl: round(float(p), 6) for lbl, p in zip(self.classes, probs)}

        return {
            "predicted_label": pred_label,
            "confidence": round(confidence, 4),
            "is_attack": (pred_label.upper() != "BENIGN"),
            "probabilities": prob_dist,
        }
