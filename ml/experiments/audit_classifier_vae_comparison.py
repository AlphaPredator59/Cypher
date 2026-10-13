import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.vae import Model
from ml.features import CICIDS_COLUMN_MAPPING

SPLITS_DIR = PROJECT_ROOT / "ml" / "splits_leakage_resistant"
ART_DIR = PROJECT_ROOT / "artifacts"
THRESHOLD_PROD = 38.50959014892578

print("=== 1. Loading Leakage-Resistant Splits ===")
train_df = pd.read_csv(SPLITS_DIR / "train.csv")
val_df = pd.read_csv(SPLITS_DIR / "validation.csv")
test_df = pd.read_csv(SPLITS_DIR / "test.csv")

feat_cols = [c for c in train_df.columns if c not in ["Label", "_source_file"]]

encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(train_df["Label"])
y_val_enc = encoder.transform(val_df["Label"])
y_test_enc = encoder.transform(test_df["Label"])
classes = list(encoder.classes_)
benign_idx = classes.index("BENIGN")

print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
print(f"Classes ({len(classes)}): {classes}")

# Train XGBoost
print("\n=== 2. Training XGBoost Classifier ===")
xgb_model = xgb.XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1,
    random_state=42, tree_method="hist", eval_metric="mlogloss"
)
xgb_model.fit(train_df[feat_cols], y_train_enc, eval_set=[(val_df[feat_cols], y_val_enc)], verbose=False)

# Load VAE and Scaler
print("\n=== 3. Loading VAE Model and Scaler ===")
with open(ART_DIR / "features_name.json", "r", encoding="utf-8") as f:
    vae_feature_names = json.load(f)["feature_names"]

scaler = joblib.load(ART_DIR / "scaler_train_benign.pkl")
vae_model = Model(input_dim=len(vae_feature_names), hidden_dim=256, latent_dim=16)
vae_model.load_state_dict(torch.load(ART_DIR / "model.pth", map_location="cpu"))
vae_model.eval()

def compute_vae_scores(df: pd.DataFrame) -> np.ndarray:
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.strip()
    df_clean.rename(columns=CICIDS_COLUMN_MAPPING, inplace=True)
    X_raw = df_clean[vae_feature_names].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = scaler.transform(X_raw).astype(np.float32)
    
    scores = []
    with torch.no_grad():
        for i in range(0, len(X_scaled), 2048):
            batch = torch.tensor(X_scaled[i:i+2048], dtype=torch.float32)
            recon, mu, _ = vae_model(batch, deterministic=True)
            batch_scores = torch.mean((batch - recon) ** 2, dim=1).cpu().numpy()
            scores.extend(batch_scores)
    return np.array(scores, dtype=np.float32)

print("Computing VAE scores for Validation...")
val_vae_scores = compute_vae_scores(val_df)
print("Computing VAE scores for Test...")
test_vae_scores = compute_vae_scores(test_df)

# Predictions and probabilities
val_probs = xgb_model.predict_proba(val_df[feat_cols])
val_preds_enc = np.argmax(val_probs, axis=1)
val_preds = encoder.inverse_transform(val_preds_enc)
val_max_probs = np.max(val_probs, axis=1)

test_probs = xgb_model.predict_proba(test_df[feat_cols])
test_preds_enc = np.argmax(test_probs, axis=1)
test_preds = encoder.inverse_transform(test_preds_enc)
test_max_probs = np.max(test_probs, axis=1)

# Function to compute binary metrics
def get_binary_metrics(y_true_labels, is_attack_pred):
    y_true_atk = (y_true_labels != "BENIGN").astype(int)
    is_atk = is_attack_pred.astype(int)
    tp = int(np.sum((y_true_atk == 1) & (is_atk == 1)))
    fp = int(np.sum((y_true_atk == 0) & (is_atk == 1)))
    tn = int(np.sum((y_true_atk == 0) & (is_atk == 0)))
    fn = int(np.sum((y_true_atk == 1) & (is_atk == 0)))
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / len(y_true_labels)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "attack_recall": rec, "fpr": fpr, "precision": prec, "f1": f1, "accuracy": acc,
        "support_attack": tp + fn, "support_benign": fp + tn
    }

print("\n" + "="*80)
print("=== 4. VALIDATION SET COMPARISON (Classifier vs VAE vs Hybrid) ===")
print("="*80)

# 1. Classifier-only (Argmax)
clf_val_atk = (val_preds != "BENIGN")
val_m_clf = get_binary_metrics(val_df["Label"], clf_val_atk)
print(f"Validation Classifier-Only: Recall={val_m_clf['attack_recall']*100:.2f}%, FPR={val_m_clf['fpr']*100:.2f}%, F1={val_m_clf['f1']:.4f}, TP={val_m_clf['tp']}, FP={val_m_clf['fp']}")

# 2. VAE-only (Production Threshold 38.51)
vae_val_atk = (val_vae_scores >= THRESHOLD_PROD)
val_m_vae = get_binary_metrics(val_df["Label"], vae_val_atk)
print(f"Validation VAE-Only (tau={THRESHOLD_PROD:.2f}): Recall={val_m_vae['attack_recall']*100:.2f}%, FPR={val_m_vae['fpr']*100:.2f}%, F1={val_m_vae['f1']:.4f}, TP={val_m_vae['tp']}, FP={val_m_vae['fp']}")

# 3. Hybrid Policy (OR: clf_val_atk | vae_val_atk)
hyb_val_atk = clf_val_atk | vae_val_atk
val_m_hyb = get_binary_metrics(val_df["Label"], hyb_val_atk)
print(f"Validation Hybrid (OR): Recall={val_m_hyb['attack_recall']*100:.2f}%, FPR={val_m_hyb['fpr']*100:.2f}%, F1={val_m_hyb['f1']:.4f}, TP={val_m_hyb['tp']}, FP={val_m_hyb['fp']}")

# Check VAE threshold sweep on validation benign to find 1% FPR or optimal threshold
benign_val_mask = (val_df["Label"] == "BENIGN")
benign_val_scores = val_vae_scores[benign_val_mask]
print(f"\nValidation Benign VAE Scores: min={benign_val_scores.min():.2f}, 50%={np.median(benign_val_scores):.2f}, 95%={np.percentile(benign_val_scores, 95):.2f}, 99%={np.percentile(benign_val_scores, 99):.2f}, max={benign_val_scores.max():.2f}")

print("\n" + "="*80)
print("=== 5. HELD-OUT TEST SET EVALUATION ===")
print("="*80)

# Test Classifier-Only (Argmax)
clf_test_atk_argmax = (test_preds != "BENIGN")
test_m_clf_argmax = get_binary_metrics(test_df["Label"], clf_test_atk_argmax)

# Test Classifier-Only (Threshold 0.50)
clf_test_atk_p50 = (test_preds != "BENIGN") & (test_max_probs >= 0.50)
test_m_clf_p50 = get_binary_metrics(test_df["Label"], clf_test_atk_p50)

# Test VAE-Only (tau = 38.51)
vae_test_atk = (test_vae_scores >= THRESHOLD_PROD)
test_m_vae = get_binary_metrics(test_df["Label"], vae_test_atk)

# Test Hybrid (OR: clf_test_atk_argmax | vae_test_atk)
hyb_test_atk = clf_test_atk_argmax | vae_test_atk
test_m_hyb = get_binary_metrics(test_df["Label"], hyb_test_atk)

# Test Hybrid with P50
hyb_test_atk_p50 = clf_test_atk_p50 | vae_test_atk
test_m_hyb_p50 = get_binary_metrics(test_df["Label"], hyb_test_atk_p50)

print(f"Test Classifier-Only (Argmax): TP={test_m_clf_argmax['tp']}, FP={test_m_clf_argmax['fp']}, TN={test_m_clf_argmax['tn']}, FN={test_m_clf_argmax['fn']}")
print(f"  Recall={test_m_clf_argmax['attack_recall']*100:.4f}%, FPR={test_m_clf_argmax['fpr']*100:.4f}%, Prec={test_m_clf_argmax['precision']*100:.4f}%, F1={test_m_clf_argmax['f1']:.6f}, Acc={test_m_clf_argmax['accuracy']*100:.4f}%")

print(f"\nTest Classifier-Only (P >= 0.50): TP={test_m_clf_p50['tp']}, FP={test_m_clf_p50['fp']}, TN={test_m_clf_p50['tn']}, FN={test_m_clf_p50['fn']}")
print(f"  Recall={test_m_clf_p50['attack_recall']*100:.4f}%, FPR={test_m_clf_p50['fpr']*100:.4f}%, Prec={test_m_clf_p50['precision']*100:.4f}%, F1={test_m_clf_p50['f1']:.6f}, Acc={test_m_clf_p50['accuracy']*100:.4f}%")

print(f"\nTest VAE-Only (tau={THRESHOLD_PROD:.2f}): TP={test_m_vae['tp']}, FP={test_m_vae['fp']}, TN={test_m_vae['tn']}, FN={test_m_vae['fn']}")
print(f"  Recall={test_m_vae['attack_recall']*100:.4f}%, FPR={test_m_vae['fpr']*100:.4f}%, Prec={test_m_vae['precision']*100:.4f}%, F1={test_m_vae['f1']:.6f}, Acc={test_m_vae['accuracy']*100:.4f}%")

print(f"\nTest Hybrid (OR with Argmax): TP={test_m_hyb['tp']}, FP={test_m_hyb['fp']}, TN={test_m_hyb['tn']}, FN={test_m_hyb['fn']}")
print(f"  Recall={test_m_hyb['attack_recall']*100:.4f}%, FPR={test_m_hyb['fpr']*100:.4f}%, Prec={test_m_hyb['precision']*100:.4f}%, F1={test_m_hyb['f1']:.6f}, Acc={test_m_hyb['accuracy']*100:.4f}%")

print(f"\nTest Hybrid (OR with P >= 0.50): TP={test_m_hyb_p50['tp']}, FP={test_m_hyb_p50['fp']}, TN={test_m_hyb_p50['tn']}, FN={test_m_hyb_p50['fn']}")
print(f"  Recall={test_m_hyb_p50['attack_recall']*100:.4f}%, FPR={test_m_hyb_p50['fpr']*100:.4f}%, Prec={test_m_hyb_p50['precision']*100:.4f}%, F1={test_m_hyb_p50['f1']:.6f}, Acc={test_m_hyb_p50['accuracy']*100:.4f}%")

# Check 4 states in Pipeline
dual_signal = vae_test_atk & clf_test_atk_p50
known_attack = (~vae_test_atk) & clf_test_atk_p50
anomaly_candidate = vae_test_atk & (~clf_test_atk_p50)
normal_traffic = (~vae_test_atk) & (~clf_test_atk_p50)

print("\n=== 6. 4-State Alert Breakdown on Test Set ===")
print(f"dual_signal (VAE > tau AND Classifier Attack): {np.sum(dual_signal)}")
print(f"known_attack (VAE normal AND Classifier Attack): {np.sum(known_attack)}")
print(f"anomaly_candidate (VAE > tau AND Classifier Normal): {np.sum(anomaly_candidate)}")
print(f"normal_traffic (VAE normal AND Classifier Normal): {np.sum(normal_traffic)}")
print(f"Total: {np.sum(dual_signal) + np.sum(known_attack) + np.sum(anomaly_candidate) + np.sum(normal_traffic)}")

# What is in anomaly_candidate?
anom_cand_true_labels = test_df["Label"][anomaly_candidate]
print(f"\nTrue labels in anomaly_candidate (total={len(anom_cand_true_labels)}):")
print(anom_cand_true_labels.value_counts().to_dict())

# What is in known_attack?
known_atk_true_labels = test_df["Label"][known_attack]
print(f"\nTrue labels in known_attack (total={len(known_atk_true_labels)}):")
print(known_atk_true_labels.value_counts().to_dict())
