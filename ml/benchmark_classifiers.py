"""
Benchmark script for supervised multiclass intrusion detection classifiers.
Compares:
1. XGBoost (Hist gradient boosted decision trees)
2. LightGBM (Lightweight gradient boosting)
3. Random Forest (Ensemble of bagged trees)
4. MoE / Stacking Ensemble (Combining diverse classifiers)

Evaluates on identical held-out test split (ml/splits/test.csv, 20,931 flows).
Measures: Accuracy, Macro-F1, Weighted-F1, Per-Class F1, and Inference Latency.
"""

import os
import sys
import time
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "ml" / "splits"
REPORTS_DIR = PROJECT_ROOT / "ml" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_split(name: str):
    df = pd.read_csv(SPLIT_DIR / f"{name}.csv")
    X = df.drop(columns=["Label", "_source_file"], errors="ignore").copy()
    y = df["Label"].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
    return X, y


def evaluate_model(name: str, model, X_test, y_test, encoder):
    t0 = time.perf_counter()
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
        preds = np.argmax(probs, axis=1)
    else:
        preds = model.predict(X_test)
        if preds.ndim > 1:
            preds = np.argmax(preds, axis=1)
    latency_ms = (time.perf_counter() - t0) * 1000

    y_pred_labels = encoder.inverse_transform(preds.astype(int))

    acc = accuracy_score(y_test, y_pred_labels)
    macro_f1 = f1_score(y_test, y_pred_labels, average="macro")
    weighted_f1 = f1_score(y_test, y_pred_labels, average="weighted")
    report = classification_report(y_test, y_pred_labels, output_dict=True, zero_division=0)

    per_class_f1 = {cls: report[cls]["f1-score"] for cls in encoder.classes_ if cls in report}

    return {
        "model_name": name,
        "accuracy": round(float(acc), 5),
        "macro_f1": round(float(macro_f1), 5),
        "weighted_f1": round(float(weighted_f1), 5),
        "latency_total_ms": round(latency_ms, 2),
        "latency_per_1k_samples_ms": round(latency_ms / len(X_test) * 1000, 2),
        "per_class_f1": per_class_f1,
    }


def main():
    print("[*] Loading datasets...")
    X_train, y_train_text = load_split("train")
    X_val, y_val_text = load_split("validation")
    X_test, y_test_text = load_split("test")

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_text)
    y_val = encoder.transform(y_val_text)
    y_test = encoder.transform(y_test_text)

    print(f"    Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,} | Classes: {len(encoder.classes_)}")

    results = []

    # 1. XGBoost Baseline
    print("\n[1/3] Evaluating pre-trained / fresh XGBoost...")
    xgb_model_path = PROJECT_ROOT / "ml" / "models" / "xgb_baseline.json"
    if xgb_model_path.exists():
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(xgb_model_path)
    else:
        xgb_model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=len(encoder.classes_),
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    xgb_metrics = evaluate_model("XGBoost (Hist GBDT)", xgb_model, X_test, y_test_text, encoder)
    results.append(xgb_metrics)
    print(f"      Accuracy: {xgb_metrics['accuracy']:.4f} | Macro-F1: {xgb_metrics['macro_f1']:.4f} | Latency: {xgb_metrics['latency_per_1k_samples_ms']} ms/1k")

    # 2. LightGBM
    print("\n[2/3] Training & evaluating LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(encoder.classes_),
        n_estimators=250,
        max_depth=8,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[])
    lgb_metrics = evaluate_model("LightGBM", lgb_model, X_test, y_test_text, encoder)
    results.append(lgb_metrics)
    print(f"      Accuracy: {lgb_metrics['accuracy']:.4f} | Macro-F1: {lgb_metrics['macro_f1']:.4f} | Latency: {lgb_metrics['latency_per_1k_samples_ms']} ms/1k")

    # 3. Random Forest
    print("\n[3/3] Training & evaluating Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=16,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    rf_metrics = evaluate_model("Random Forest (150 trees)", rf_model, X_test, y_test_text, encoder)
    results.append(rf_metrics)
    print(f"      Accuracy: {rf_metrics['accuracy']:.4f} | Macro-F1: {rf_metrics['macro_f1']:.4f} | Latency: {rf_metrics['latency_per_1k_samples_ms']} ms/1k")

    # 4. Ensemble / MoE Stacking (Averaged Softmax)
    print("\n[4/4] Evaluating Soft Voting Ensemble (XGBoost + LightGBM + RF)...")
    t0 = time.perf_counter()
    p_xgb = xgb_model.predict_proba(X_test)
    p_lgb = lgb_model.predict_proba(X_test)
    p_rf = rf_model.predict_proba(X_test)
    p_ensemble = (p_xgb * 0.5) + (p_lgb * 0.35) + (p_rf * 0.15)
    lat_ms = (time.perf_counter() - t0) * 1000

    preds_ens = np.argmax(p_ensemble, axis=1)
    y_pred_labels = encoder.inverse_transform(preds_ens)
    acc = accuracy_score(y_test_text, y_pred_labels)
    macro_f1 = f1_score(y_test_text, y_pred_labels, average="macro")
    weighted_f1 = f1_score(y_test_text, y_pred_labels, average="weighted")
    report = classification_report(y_test_text, y_pred_labels, output_dict=True, zero_division=0)
    per_class_f1 = {cls: report[cls]["f1-score"] for cls in encoder.classes_ if cls in report}

    ens_metrics = {
        "model_name": "Weighted Ensemble (XGB + LGB + RF)",
        "accuracy": round(float(acc), 5),
        "macro_f1": round(float(macro_f1), 5),
        "weighted_f1": round(float(weighted_f1), 5),
        "latency_total_ms": round(lat_ms, 2),
        "latency_per_1k_samples_ms": round(lat_ms / len(X_test) * 1000, 2),
        "per_class_f1": per_class_f1,
    }
    results.append(ens_metrics)
    print(f"      Accuracy: {ens_metrics['accuracy']:.4f} | Macro-F1: {ens_metrics['macro_f1']:.4f} | Latency: {ens_metrics['latency_per_1k_samples_ms']} ms/1k")

    # Save benchmark summary
    out_path = REPORTS_DIR / "classifier_benchmark_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved full benchmark comparison to {out_path}")


if __name__ == "__main__":
    main()
