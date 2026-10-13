import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

splits_dir = Path("ml/splits_leakage_resistant")
train_df = pd.read_csv(splits_dir / "train.csv")
val_df = pd.read_csv(splits_dir / "validation.csv")
test_df = pd.read_csv(splits_dir / "test.csv")

feat_cols = [c for c in train_df.columns if c not in ["Label", "_source_file"]]
encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(train_df["Label"])
y_val_enc = encoder.transform(val_df["Label"])
y_test_enc = encoder.transform(test_df["Label"])

xgb_model = xgb.XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1,
    random_state=42, tree_method="hist", eval_metric="mlogloss"
)
xgb_model.fit(train_df[feat_cols], y_train_enc, eval_set=[(val_df[feat_cols], y_val_enc)], verbose=False)

probs = xgb_model.predict_proba(test_df[feat_cols])
preds_argmax_enc = np.argmax(probs, axis=1)
preds_argmax = encoder.inverse_transform(preds_argmax_enc)
max_probs = np.max(probs, axis=1)

benign_idx = list(encoder.classes_).index("BENIGN")

# Rule 1: Argmax != BENIGN
rule_argmax_attack = (preds_argmax != "BENIGN")

# Rule 2: Argmax != BENIGN and max_prob >= 0.50 (used by ThreatPipeline when pred_lbl != BENIGN and atk_c >= 0.50)
rule_thresh_attack = (preds_argmax != "BENIGN") & (max_probs >= 0.50)

# Rule 3: 1.0 - P(BENIGN) >= 0.50
prob_attack_sum = 1.0 - probs[:, benign_idx]
rule_sum_attack = (prob_attack_sum >= 0.50)

diff_argmax_vs_thresh = np.sum(rule_argmax_attack != rule_thresh_attack)
diff_argmax_vs_sum = np.sum(rule_argmax_attack != rule_sum_attack)

print(f"Total test rows: {len(test_df)}")
print(f"Argmax attacks: {np.sum(rule_argmax_attack)}")
print(f"Thresh >= 0.50 attacks: {np.sum(rule_thresh_attack)}")
print(f"Sum non-benign >= 0.50 attacks: {np.sum(rule_sum_attack)}")
print(f"Differences (argmax vs thresh >= 0.50): {diff_argmax_vs_thresh}")
print(f"Differences (argmax vs sum non-benign >= 0.50): {diff_argmax_vs_sum}")

if diff_argmax_vs_thresh > 0:
    diff_mask = (rule_argmax_attack != rule_thresh_attack)
    print("\nDifferences detail:")
    for idx in np.where(diff_mask)[0]:
        print(f"Row {idx}: True={test_df['Label'].iloc[idx]}, Pred={preds_argmax[idx]}, MaxProb={max_probs[idx]:.4f}, BenignProb={probs[idx, benign_idx]:.4f}")
