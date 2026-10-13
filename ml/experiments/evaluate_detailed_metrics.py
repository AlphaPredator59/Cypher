import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

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

y_true_binary = (test_df["Label"] != "BENIGN").to_numpy(dtype=int)

def evaluate_binary(name, y_pred_binary):
    tp = int(np.sum((y_true_binary == 1) & (y_pred_binary == 1)))
    fp = int(np.sum((y_true_binary == 0) & (y_pred_binary == 0))) # wait! fp is true 0 & pred 1!
    fp = int(np.sum((y_true_binary == 0) & (y_pred_binary == 1)))
    tn = int(np.sum((y_true_binary == 0) & (y_pred_binary == 0)))
    fn = int(np.sum((y_true_binary == 1) & (y_pred_binary == 0)))
    
    rec = tp / (tp + fn)
    fpr = fp / (fp + tn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    acc = (tp + tn) / len(y_true_binary)
    print(f"[{name}]")
    print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn} (Total={tp+fp+tn+fn})")
    print(f"  Attack Recall: {rec*100:.4f}% ({tp}/{tp+fn})")
    print(f"  FPR:           {fpr*100:.4f}% ({fp}/{fp+tn})")
    print(f"  Precision:     {prec*100:.4f}%")
    print(f"  F1:            {f1:.6f}")
    print(f"  Accuracy:      {acc*100:.4f}%\n")

# Rule A: Argmax != BENIGN
pred_a = (preds_argmax != "BENIGN").astype(int)
evaluate_binary("Rule A: Argmax != BENIGN", pred_a)

# Rule B: Argmax != BENIGN and MaxProb >= 0.50
pred_b = ((preds_argmax != "BENIGN") & (max_probs >= 0.50)).astype(int)
evaluate_binary("Rule B: (Argmax != BENIGN) & (P_max >= 0.50)", pred_b)

# Rule C: P(Attack) = 1.0 - P(BENIGN) >= 0.50
pred_c = ((1.0 - probs[:, benign_idx]) >= 0.50).astype(int)
evaluate_binary("Rule C: P(Attack) = 1.0 - P(BENIGN) >= 0.50", pred_c)

# Now inspect the confusion matrix of 12 classes under Argmax
classes = list(encoder.classes_)
cm = confusion_matrix(test_df["Label"], preds_argmax, labels=classes)
print("12-Class Confusion Matrix (Row=True, Col=Pred):")
print(f"Classes: {classes}")
print(cm)

print("\nPer-class Support & Classification Report:")
rep = classification_report(test_df["Label"], preds_argmax, labels=classes, digits=4)
print(rep)
