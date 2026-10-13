import pandas as pd
from pathlib import Path

splits_dir = Path("ml/splits")
df = pd.concat([pd.read_csv(splits_dir / f"{s}.csv") for s in ["train", "validation", "test"]], ignore_index=True)
feat_cols = [c for c in df.columns if c not in ["Label", "_source_file"]]

grouped = df.groupby(feat_cols)["Label"].nunique()
conflicting_features = grouped[grouped > 1]
print(f"Total unique feature groups with conflicting labels: {len(conflicting_features)}")

conf_rows = 0
for _, cg in df.groupby(feat_cols):
    if cg["Label"].nunique() > 1:
        conf_rows += len(cg)
        print(f"Conflicting group (rows={len(cg)}): {dict(cg['Label'].value_counts())}")

print(f"Total rows involved in conflicting groups: {conf_rows} out of {len(df)} ({conf_rows/len(df)*100:.4f}%)")
