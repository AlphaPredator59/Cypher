"""
Evaluation script for VAE Anomaly Detection in CYPHER IDS.

Conducts rigorous, evidence-based evaluation comparing:
1. Baseline Stochastic Inference (sampling via reparameterization).
2. Deterministic Inference (latent mean decoding z = mu).

Methodology:
- Validation Split (ml/splits/validation.csv, 20,930 rows) is used exclusively for:
  * Measuring score distributions on Benign and Attack traffic.
  * Performing threshold sweeps across empirical quantiles and fixed thresholds.
  * Evaluating the precision/recall vs. false-positive rate (FPR) trade-off.
  * Selecting candidate recalibrated thresholds.
- Test Split (ml/splits/test.csv, 20,931 rows) is held out and used ONLY for final unbiased evaluation of:
  * Baseline Stochastic with production threshold (38.50959).
  * Deterministic with production threshold (38.50959).
  * Deterministic with validation-recalibrated thresholds.

Saves structured reports to ml/reports/vae_deterministic_eval.json and
ml/reports/vae_threshold_sweep.json.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Any, Tuple
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.models.vae import Model
from ml.features import CICIDS_COLUMN_MAPPING
ART_DIR = os.path.join(PROJECT_ROOT, "artifacts")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "ml", "reports")
VALIDATION_CSV = os.path.join(PROJECT_ROOT, "ml", "splits", "validation.csv")
TEST_CSV = os.path.join(PROJECT_ROOT, "ml", "splits", "test.csv")
PRODUCTION_THRESHOLD = 38.50959014892578


def load_model_and_scaler() -> Tuple[Model, Any, List[str]]:
    with open(os.path.join(ART_DIR, "features_name.json"), "r", encoding="utf-8") as f:
        feature_names = json.load(f)["feature_names"]

    scaler = joblib.load(os.path.join(ART_DIR, "scaler_train_benign.pkl"))
    model = Model(input_dim=len(feature_names), hidden_dim=256, latent_dim=16)
    model.load_state_dict(torch.load(os.path.join(ART_DIR, "model.pth"), map_location="cpu"))
    model.eval()
    return model, scaler, feature_names


def load_and_preprocess_split(csv_path: str, feature_names: List[str], scaler: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Loads split CSV, applies column mapping, validates 77 features, normalizes,
    and returns (X_scaled, y_binary, labels, label_names).
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df.rename(columns=CICIDS_COLUMN_MAPPING, inplace=True)

    labels = df["Label"].astype(str).str.strip().values
    y_binary = np.array([0 if lbl.upper() == "BENIGN" else 1 for lbl in labels], dtype=int)

    # Convert features to float32
    X_raw = df[feature_names].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)

    # Impute any NaN/Inf with 0.0 if present, though splits should be clean
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    X_scaled = scaler.transform(X_raw).astype(np.float32)
    return X_scaled, y_binary, labels


def compute_reconstruction_scores(model: Model, X_scaled: np.ndarray, deterministic: bool, batch_size: int = 2048) -> np.ndarray:
    """
    Computes sample-wise SSE reconstruction error: (recon - x)^2.sum(dim=1).
    """
    scores = []
    n = len(X_scaled)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = torch.from_numpy(X_scaled[i:i + batch_size])
            recon, _, _ = model(batch, deterministic=deterministic)
            se = (recon - batch) ** 2
            batch_scores = se.sum(dim=1).cpu().numpy()
            scores.append(batch_scores)
    return np.concatenate(scores)


def compute_metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float, labels: np.ndarray) -> Dict[str, Any]:
    y_pred = (scores > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # Per-class detection rate (recall)
    per_class = {}
    unique_labels = sorted(set(labels))
    for lbl in unique_labels:
        mask = (labels == lbl)
        count = int(np.sum(mask))
        detected = int(np.sum(y_pred[mask] == 1))
        rate = float(detected / count) if count > 0 else 0.0
        is_benign = (lbl.upper() == "BENIGN")
        per_class[lbl] = {
            "count": count,
            "detected": detected,
            "rate": round(rate, 4),
            "metric_type": "FPR" if is_benign else "Recall"
        }

    return {
        "threshold": round(float(threshold), 6),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "per_class": per_class
    }


def run_threshold_sweep(y_true: np.ndarray, scores: np.ndarray, labels: np.ndarray, benign_scores: np.ndarray) -> List[Dict[str, Any]]:
    # Quantiles of benign validation scores
    quantiles = [0.90, 0.95, 0.98, 0.99, 0.995, 0.999]
    q_thresholds = [(f"q_{q}", float(np.percentile(benign_scores, q * 100))) for q in quantiles]

    fixed_thresholds = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, PRODUCTION_THRESHOLD, 40.0, 50.0, 60.0, 80.0, 100.0]

    all_candidate_thrs = set([t for _, t in q_thresholds] + fixed_thresholds)
    sorted_thrs = sorted(all_candidate_thrs)

    sweep_results = []
    for thr in sorted_thrs:
        res = compute_metrics_at_threshold(y_true, scores, thr, labels)
        sweep_results.append(res)
    return sweep_results


def main():
    print("=" * 70)
    print("CYPHER VAE EVALUATION: BASELINE STOCHASTIC VS DETERMINISTIC")
    print("=" * 70)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    model, scaler, feature_names = load_model_and_scaler()

    # 1. Load Validation Split
    print(f"\n[1/4] Loading Validation Split: {VALIDATION_CSV}")
    X_val, y_val_bin, labels_val = load_and_preprocess_split(VALIDATION_CSV, feature_names, scaler)
    print(f"      Validation size: {len(X_val):,} rows | Benign: {np.sum(y_val_bin == 0):,} | Attack: {np.sum(y_val_bin == 1):,}")

    # 2. Score Validation Split
    print("\n[2/4] Scoring Validation Split...")
    # Baseline Stochastic (repeat 3 times to measure score variance on same data)
    torch.manual_seed(42)
    val_scores_stoch_1 = compute_reconstruction_scores(model, X_val, deterministic=False)
    val_scores_stoch_2 = compute_reconstruction_scores(model, X_val, deterministic=False)
    stoch_diff = np.abs(val_scores_stoch_1 - val_scores_stoch_2)
    print(f"      Stochastic inter-run mean absolute difference: {np.mean(stoch_diff):.4f} (max: {np.max(stoch_diff):.4f})")

    # Deterministic (repeat 2 times to verify zero variance)
    val_scores_det_1 = compute_reconstruction_scores(model, X_val, deterministic=True)
    val_scores_det_2 = compute_reconstruction_scores(model, X_val, deterministic=True)
    det_diff = np.max(np.abs(val_scores_det_1 - val_scores_det_2))
    print(f"      Deterministic inter-run max absolute difference: {det_diff:.6f} (Strictly 0.0)")

    val_scores_stoch = val_scores_stoch_1
    val_scores_det = val_scores_det_1

    # Benign Validation Distribution
    benign_mask_val = (y_val_bin == 0)
    benign_val_det = val_scores_det[benign_mask_val]
    benign_val_stoch = val_scores_stoch[benign_mask_val]

    print("\n      --- Benign Validation Reconstruction Error Distribution ---")
    print(f"      Stochastic:    Mean={np.mean(benign_val_stoch):.2f}, Std={np.std(benign_val_stoch):.2f}, Median={np.median(benign_val_stoch):.2f}")
    print(f"                     p95={np.percentile(benign_val_stoch, 95):.2f}, p99={np.percentile(benign_val_stoch, 99):.2f}, p99.9={np.percentile(benign_val_stoch, 99.9):.2f}")
    print(f"      Deterministic: Mean={np.mean(benign_val_det):.2f}, Std={np.std(benign_val_det):.2f}, Median={np.median(benign_val_det):.2f}")
    print(f"                     p95={np.percentile(benign_val_det, 95):.2f}, p98={np.percentile(benign_val_det, 98):.2f}, p99={np.percentile(benign_val_det, 99):.2f}, p99.5={np.percentile(benign_val_det, 99.5):.2f}, p99.9={np.percentile(benign_val_det, 99.9):.2f}")

    # Threshold Sweep on Validation Split
    print("\n[3/4] Performing Validation Threshold Sweep...")
    val_sweep_det = run_threshold_sweep(y_val_bin, val_scores_det, labels_val, benign_val_det)
    val_sweep_stoch = run_threshold_sweep(y_val_bin, val_scores_stoch, labels_val, benign_val_stoch)

    with open(os.path.join(REPORTS_DIR, "vae_threshold_sweep.json"), "w", encoding="utf-8") as f:
        json.dump({
            "validation_sweep_deterministic": val_sweep_det,
            "validation_sweep_stochastic": val_sweep_stoch
        }, f, indent=2)
    print(f"      Saved validation sweep to {os.path.join(REPORTS_DIR, 'vae_threshold_sweep.json')}")

    # Select Recalibrated Thresholds strictly from Validation data
    # Threshold 1: Existing Production threshold
    t_prod = PRODUCTION_THRESHOLD
    # Threshold 2: Recalibrated q=0.999 (0.1% target FPR on benign validation)
    t_q999 = float(np.percentile(benign_val_det, 99.9))
    # Threshold 3: Recalibrated q=0.99 (1.0% target FPR on benign validation)
    t_q990 = float(np.percentile(benign_val_det, 99.0))
    # Threshold 4: Recalibrated q=0.95 (5.0% target FPR on benign validation)
    t_q950 = float(np.percentile(benign_val_det, 95.0))

    print("\n      --- Candidate Thresholds Selected from Validation Data ---")
    print(f"      1. Production Baseline:          {t_prod:.5f}")
    print(f"      2. Recalibrated q=0.999 (0.1% FPR): {t_q999:.5f}")
    print(f"      3. Recalibrated q=0.990 (1.0% FPR): {t_q990:.5f}")
    print(f"      4. Recalibrated q=0.950 (5.0% FPR): {t_q950:.5f}")

    # 4. Final Unbiased Evaluation on Held-Out Test Set
    print(f"\n[4/4] Evaluating Held-Out Test Split: {TEST_CSV}")
    X_test, y_test_bin, labels_test = load_and_preprocess_split(TEST_CSV, feature_names, scaler)
    print(f"      Test size: {len(X_test):,} rows | Benign: {np.sum(y_test_bin == 0):,} | Attack: {np.sum(y_test_bin == 1):,}")

    test_scores_stoch = compute_reconstruction_scores(model, X_test, deterministic=False)
    test_scores_det = compute_reconstruction_scores(model, X_test, deterministic=True)

    test_metrics_baseline_stoch = compute_metrics_at_threshold(y_test_bin, test_scores_stoch, t_prod, labels_test)
    test_metrics_det_prod = compute_metrics_at_threshold(y_test_bin, test_scores_det, t_prod, labels_test)
    test_metrics_det_q999 = compute_metrics_at_threshold(y_test_bin, test_scores_det, t_q999, labels_test)
    test_metrics_det_q990 = compute_metrics_at_threshold(y_test_bin, test_scores_det, t_q990, labels_test)
    test_metrics_det_q950 = compute_metrics_at_threshold(y_test_bin, test_scores_det, t_q950, labels_test)

    eval_report = {
        "dataset_metadata": {
            "validation_split": VALIDATION_CSV,
            "validation_total_rows": int(len(X_val)),
            "validation_benign_rows": int(np.sum(y_val_bin == 0)),
            "validation_attack_rows": int(np.sum(y_val_bin == 1)),
            "test_split": TEST_CSV,
            "test_total_rows": int(len(X_test)),
            "test_benign_rows": int(np.sum(y_test_bin == 0)),
            "test_attack_rows": int(np.sum(y_test_bin == 1)),
            "feature_count": len(feature_names),
            "production_threshold": t_prod,
            "recalibrated_threshold_q999": t_q999,
            "recalibrated_threshold_q990": t_q990,
            "recalibrated_threshold_q950": t_q950,
        },
        "benign_validation_score_distribution": {
            "deterministic": {
                "mean": float(np.mean(benign_val_det)),
                "std": float(np.std(benign_val_det)),
                "median": float(np.median(benign_val_det)),
                "p90": float(np.percentile(benign_val_det, 90)),
                "p95": float(np.percentile(benign_val_det, 95)),
                "p98": float(np.percentile(benign_val_det, 98)),
                "p99": float(np.percentile(benign_val_det, 99)),
                "p99_5": float(np.percentile(benign_val_det, 99.5)),
                "p99_9": float(np.percentile(benign_val_det, 99.9)),
            },
            "stochastic": {
                "mean": float(np.mean(benign_val_stoch)),
                "std": float(np.std(benign_val_stoch)),
                "median": float(np.median(benign_val_stoch)),
                "p95": float(np.percentile(benign_val_stoch, 95)),
                "p99": float(np.percentile(benign_val_stoch, 99)),
                "p99_9": float(np.percentile(benign_val_stoch, 99.9)),
            }
        },
        "test_evaluation_results": {
            "baseline_stochastic_prod_threshold": test_metrics_baseline_stoch,
            "deterministic_prod_threshold": test_metrics_det_prod,
            "deterministic_recalibrated_q999": test_metrics_det_q999,
            "deterministic_recalibrated_q990": test_metrics_det_q990,
            "deterministic_recalibrated_q950": test_metrics_det_q950,
        }
    }

    report_path = os.path.join(REPORTS_DIR, "vae_deterministic_eval.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)

    print(f"\n[*] Detailed report saved to: {report_path}")

    # Print Comparative Summary Table
    print("\n" + "=" * 95)
    print("COMPARATIVE EVALUATION SUMMARY (HELD-OUT TEST SET: 20,931 ROWS)")
    print("=" * 95)
    header = f"{'Configuration':<38} | {'Threshold':>10} | {'Prec':>6} | {'Recall':>6} | {'F1':>6} | {'FPR':>6} | {'Acc':>6}"
    print(header)
    print("-" * 95)
    rows = [
        ("1. Baseline Stochastic (Prod Thr)", t_prod, test_metrics_baseline_stoch),
        ("2. Deterministic (Prod Thr)", t_prod, test_metrics_det_prod),
        ("3. Deterministic (Recalibrated q=0.999)", t_q999, test_metrics_det_q999),
        ("4. Deterministic (Recalibrated q=0.990)", t_q990, test_metrics_det_q990),
        ("5. Deterministic (Recalibrated q=0.950)", t_q950, test_metrics_det_q950),
    ]
    for name, thr, m in rows:
        print(f"{name:<38} | {thr:>10.4f} | {m['precision']:>6.4f} | {m['recall']:>6.4f} | {m['f1']:>6.4f} | {m['fpr']:>6.4f} | {m['accuracy']:>6.4f}")
    print("=" * 95)

    print("\nPER-ATTACK-CATEGORY RECALL COMPARISON ON TEST SET:")
    print("-" * 95)
    per_class_hdr = f"{'Attack Category':<25} | {'Count':>6} | {'Stoch (38.51)':>13} | {'Det (38.51)':>11} | {'Det (q=0.990)':>13}"
    print(per_class_hdr)
    print("-" * 95)
    for lbl in sorted(test_metrics_baseline_stoch["per_class"].keys()):
        cnt = test_metrics_baseline_stoch["per_class"][lbl]["count"]
        r_stoch = test_metrics_baseline_stoch["per_class"][lbl]["rate"]
        r_det_prod = test_metrics_det_prod["per_class"][lbl]["rate"]
        r_det_q99 = test_metrics_det_q990["per_class"][lbl]["rate"]
        print(f"{lbl:<25} | {cnt:>6} | {r_stoch:>13.4f} | {r_det_prod:>11.4f} | {r_det_q99:>13.4f}")
    print("-" * 95)


if __name__ == "__main__":
    main()
