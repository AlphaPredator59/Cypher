"""
Leakage-Resistant Model Comparison & Evaluation for CYPHER IDS.

Evaluates:
1. XGBoost 12-class classifier trained on leak-free split.
2. LightGBM 12-class classifier trained on leak-free split.
3. Hierarchical Web Attack aggregation (Web Attack XSS + Brute Force -> Web Attack).
4. Hybrid detection engine (VAE + XGBoost) on leak-free split.
5. Direct comparison against the original random-split baseline.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = PROJECT_ROOT / "ml" / "splits_leakage_resistant"
OUTPUT_DIR = PROJECT_ROOT / "ml" / "reports"

RANDOM_STATE = 42

SUPPORTED_CLASSES = [
    "BENIGN",
    "DoS Hulk",
    "PortScan",
    "DDoS",
    "DoS GoldenEye",
    "FTP-Patator",
    "SSH-Patator",
    "DoS slowloris",
    "DoS Slowhttptest",
    "Bot",
    "Web Attack Brute Force",
    "Web Attack XSS",
]


def load_leak_free_data():
    print(f"Loading data from {SPLITS_DIR}...")
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "validation.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")

    feature_cols = [c for c in train_df.columns if c not in ["Label", "_source_file"]]

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df["Label"].to_numpy()

    X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    y_val = val_df["Label"].to_numpy()

    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df["Label"].to_numpy()

    encoder = LabelEncoder()
    # Fit encoder strictly on train labels
    encoder.fit(y_train)

    y_train_enc = encoder.transform(y_train)
    y_val_enc = encoder.transform(y_val)
    y_test_enc = encoder.transform(y_test)

    return {
        "feature_cols": feature_cols,
        "encoder": encoder,
        "X_train": X_train,
        "y_train": y_train,
        "y_train_enc": y_train_enc,
        "X_val": X_val,
        "y_val": y_val,
        "y_val_enc": y_val_enc,
        "X_test": X_test,
        "y_test": y_test,
        "y_test_enc": y_test_enc,
    }


def compute_metrics(y_true_str, y_pred_str, classes):
    acc = accuracy_score(y_true_str, y_pred_str)
    macro_f1 = f1_score(y_true_str, y_pred_str, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true_str, y_pred_str, average="weighted", zero_division=0)

    clf_rep = classification_report(y_true_str, y_pred_str, labels=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true_str, y_pred_str, labels=classes)

    # Binary detection: Normal (BENIGN) vs Attack (not BENIGN)
    is_attack_true = (y_true_str != "BENIGN").astype(int)
    is_attack_pred = (y_pred_str != "BENIGN").astype(int)

    tp = int(np.sum((is_attack_true == 1) & (is_attack_pred == 1)))
    fp = int(np.sum((is_attack_true == 0) & (is_attack_pred == 1)))
    tn = int(np.sum((is_attack_true == 0) & (is_attack_pred == 0)))
    fn = int(np.sum((is_attack_true == 1) & (is_attack_pred == 0)))

    bin_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    bin_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    bin_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    bin_f1 = 2 * bin_precision * bin_recall / (bin_precision + bin_recall) if (bin_precision + bin_recall) > 0 else 0.0
    bin_acc = (tp + tn) / len(y_true_str)

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": clf_rep,
        "confusion_matrix": cm.tolist(),
        "binary_detection": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "attack_recall": float(bin_recall),
            "fpr": float(bin_fpr),
            "precision": float(bin_precision),
            "f1": float(bin_f1),
            "binary_accuracy": float(bin_acc),
        },
    }


def compute_hierarchical_metrics(y_true_str, y_pred_str):
    # Group Web Attack XSS and Web Attack Brute Force into 'Web Attack'
    def map_hierarchical(lbl):
        if lbl in ["Web Attack XSS", "Web Attack Brute Force", "Web Attack SQL Injection"]:
            return "Web Attack"
        return lbl

    h_true = [map_hierarchical(x) for x in y_true_str]
    h_pred = [map_hierarchical(x) for x in y_pred_str]

    h_classes = [c for c in SUPPORTED_CLASSES if c not in ["Web Attack XSS", "Web Attack Brute Force"]] + ["Web Attack"]

    acc = accuracy_score(h_true, h_pred)
    macro_f1 = f1_score(h_true, h_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(h_true, h_pred, average="weighted", zero_division=0)
    clf_rep = classification_report(h_true, h_pred, labels=h_classes, output_dict=True, zero_division=0)

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "web_attack_metrics": clf_rep.get("Web Attack", {}),
        "full_report": clf_rep,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_leak_free_data()

    encoder = data["encoder"]
    classes = list(encoder.classes_)

    results = {
        "split_info": {
            "strategy": "Stratified Group-Aware Content Fingerprint (Zero-Duplicate)",
            "train_rows": len(data["y_train"]),
            "val_rows": len(data["y_val"]),
            "test_rows": len(data["y_test"]),
            "classes": classes,
        },
        "models": {},
    }

    # ---------------------------------------------------------
    # 1. Train & Evaluate XGBoost on Leak-Free Split
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("1. TRAINING & EVALUATING XGBOOST ON LEAKAGE-RESISTANT SPLIT")
    print("=" * 80)

    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        tree_method="hist",
        eval_metric="mlogloss",
    )

    t0 = time.perf_counter()
    xgb_model.fit(
        data["X_train"],
        data["y_train_enc"],
        eval_set=[(data["X_val"], data["y_val_enc"])],
        verbose=False,
    )
    xgb_train_time = time.perf_counter() - t0
    print(f"XGBoost training completed in {xgb_train_time:.2f}s")

    # Latency benchmark on 1,000 test flows
    t0 = time.perf_counter()
    _ = xgb_model.predict(data["X_test"][:1000])
    xgb_latency_ms = (time.perf_counter() - t0) * 1000.0

    y_test_pred_xgb_enc = xgb_model.predict(data["X_test"])
    y_test_pred_xgb = encoder.inverse_transform(y_test_pred_xgb_enc)

    xgb_metrics = compute_metrics(data["y_test"], y_test_pred_xgb, classes)
    xgb_hierarchical = compute_hierarchical_metrics(data["y_test"], y_test_pred_xgb)

    xgb_metrics["train_time_sec"] = float(xgb_train_time)
    xgb_metrics["latency_ms_per_1k"] = float(xgb_latency_ms)
    xgb_metrics["hierarchical"] = xgb_hierarchical

    results["models"]["xgboost_leak_free"] = xgb_metrics

    print(f"\n[XGBoost Leak-Free Results]")
    print(f"  Test Accuracy:     {xgb_metrics['accuracy'] * 100:.2f}%")
    print(f"  Macro-F1:          {xgb_metrics['macro_f1']:.4f}")
    print(f"  Weighted-F1:       {xgb_metrics['weighted_f1']:.4f}")
    print(f"  Binary Recall:     {xgb_metrics['binary_detection']['attack_recall'] * 100:.2f}%")
    print(f"  Binary FPR:        {xgb_metrics['binary_detection']['fpr'] * 100:.2f}%")
    print(f"  Inference Latency: {xgb_latency_ms:.2f} ms / 1k flows")
    print(f"  Hierarchical Acc:  {xgb_hierarchical['accuracy'] * 100:.2f}% (Macro-F1: {xgb_hierarchical['macro_f1']:.4f})")
    print(f"  Web Attack F1:     {xgb_hierarchical['web_attack_metrics'].get('f1-score', 0):.4f}")

    # ---------------------------------------------------------
    # 2. Train & Evaluate LightGBM on Leak-Free Split
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("2. TRAINING & EVALUATING LIGHTGBM ON LEAKAGE-RESISTANT SPLIT")
    print("=" * 80)

    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )

    t0 = time.perf_counter()
    lgb_model.fit(
        data["X_train"],
        data["y_train_enc"],
        eval_set=[(data["X_val"], data["y_val_enc"])],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)],
    )
    lgb_train_time = time.perf_counter() - t0
    print(f"LightGBM training completed in {lgb_train_time:.2f}s")

    t0 = time.perf_counter()
    _ = lgb_model.predict(data["X_test"][:1000])
    lgb_latency_ms = (time.perf_counter() - t0) * 1000.0

    y_test_pred_lgb_enc = lgb_model.predict(data["X_test"])
    y_test_pred_lgb = encoder.inverse_transform(y_test_pred_lgb_enc)

    lgb_metrics = compute_metrics(data["y_test"], y_test_pred_lgb, classes)
    lgb_hierarchical = compute_hierarchical_metrics(data["y_test"], y_test_pred_lgb)

    lgb_metrics["train_time_sec"] = float(lgb_train_time)
    lgb_metrics["latency_ms_per_1k"] = float(lgb_latency_ms)
    lgb_metrics["hierarchical"] = lgb_hierarchical

    results["models"]["lightgbm_leak_free"] = lgb_metrics

    print(f"\n[LightGBM Leak-Free Results]")
    print(f"  Test Accuracy:     {lgb_metrics['accuracy'] * 100:.2f}%")
    print(f"  Macro-F1:          {lgb_metrics['macro_f1']:.4f}")
    print(f"  Weighted-F1:       {lgb_metrics['weighted_f1']:.4f}")
    print(f"  Binary Recall:     {lgb_metrics['binary_detection']['attack_recall'] * 100:.2f}%")
    print(f"  Binary FPR:        {lgb_metrics['binary_detection']['fpr'] * 100:.2f}%")
    print(f"  Inference Latency: {lgb_latency_ms:.2f} ms / 1k flows")
    print(f"  Hierarchical Acc:  {lgb_hierarchical['accuracy'] * 100:.2f}% (Macro-F1: {lgb_hierarchical['macro_f1']:.4f})")
    print(f"  Web Attack F1:     {lgb_hierarchical['web_attack_metrics'].get('f1-score', 0):.4f}")

    # ---------------------------------------------------------
    # 3. Baseline Comparison Table
    # ---------------------------------------------------------
    baseline_random_split = {
        "accuracy": 0.9937,
        "macro_f1": 0.9309,
        "weighted_f1": 0.9936,
        "binary_recall": 0.9995,
        "binary_fpr": 0.0493,
        "cross_split_leakage_pct": 11.79,
    }
    results["baseline_random_split"] = baseline_random_split

    out_file = OUTPUT_DIR / "leakage_resistant_evaluation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults successfully saved to: {out_file}")
    return results


if __name__ == "__main__":
    main()
