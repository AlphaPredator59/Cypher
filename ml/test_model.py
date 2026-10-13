"""
Model validation script for TRACE intrusion detection pipeline.

This script tests the trained VAE + MoE ThreatPipeline against a balanced
sample of benign and attack traffic from the CICIDS2017 dataset.

Validation tasks:
1. Load and normalize features using the project's canonical mapping from main.py.
2. Filter out and count malformed/missing/NaN/Inf rows without synthetic imputation.
3. Build a balanced evaluation dataset containing both benign and attack categories.
4. Evaluate VAE anomaly detection against ground-truth labels.
5. Compute Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.
6. For detected anomalies, analyze the MoE predicted attack types and confidence levels.
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from ml.features import CICIDS_COLUMN_MAPPING
from ml.pipeline import PIPELINE

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_FOLDER = os.path.join(
    PROJECT_ROOT,
    "solarmainframe",
    "ids-intrusion-csv",
    "versions",
    "1"
)
REPORTS_DIR = os.path.join(PROJECT_ROOT, "ml", "reports")


def load_dataset_samples(
    csv_folder: str,
    target_per_class: int = 2500,
    random_seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[str], List[int], Dict[str, Any]]:
    """
    Scans the CSV folder and loads a balanced sample of benign and attack rows.
    Strictly validates required features, skipping malformed rows without imputing values.
    Uses vectorized pandas filtering for fast processing.

    Returns:
        rows: List of feature dictionaries ready for ThreatPipeline.predict_batch()
        ground_truth_labels: List of original ground-truth label strings (e.g., 'BENIGN', 'DDoS')
        ground_truth_binary: List of binary ground-truth labels (0 = Benign, 1 = Attack)
        stats: Statistics on loaded rows, skipped rows, and class distributions
    """
    np.random.seed(random_seed)
    vae_required_features = PIPELINE.vae_feature_names
    moe_required_features = PIPELINE.moe.FEATURE_COLUMNS
    all_required_features = list(dict.fromkeys(vae_required_features + moe_required_features))

    csv_files = sorted(glob.glob(os.path.join(csv_folder, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_folder}")

    print(f"[*] Found {len(csv_files)} dataset CSV files in {csv_folder}")
    print(f"[*] Required VAE features: {len(vae_required_features)}, MoE features: {len(moe_required_features)}")

    benign_dfs: List[pd.DataFrame] = []
    attack_dfs_by_type: Dict[str, List[pd.DataFrame]] = {}

    total_rows_scanned = 0
    total_rows_skipped_malformed = 0
    total_rows_skipped_missing_features = 0

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        print(f"[*] Scanning {filename}...", flush=True)

        try:
            chunk_size = 100000
            for chunk in pd.read_csv(filepath, chunksize=chunk_size, encoding="cp1252", low_memory=False):
                total_rows_scanned += len(chunk)
                chunk.columns = chunk.columns.str.strip()
                chunk.rename(columns=CICIDS_COLUMN_MAPPING, inplace=True)

                if "Label" not in chunk.columns:
                    total_rows_skipped_missing_features += len(chunk)
                    continue

                missing_cols = [c for c in all_required_features if c not in chunk.columns]
                if missing_cols:
                    total_rows_skipped_missing_features += len(chunk)
                    continue

                # Clean labels and handle NaN/empty
                chunk["Label"] = chunk["Label"].astype(str).str.strip()
                valid_label_mask = chunk["Label"].notna() & (chunk["Label"] != "") & (chunk["Label"] != "nan")

                # Validate numeric features strictly without imputation
                numeric_df = chunk[all_required_features].apply(pd.to_numeric, errors="coerce")
                valid_numeric_mask = numeric_df.notna().all(axis=1) & np.isfinite(numeric_df).all(axis=1)

                valid_mask = valid_label_mask & valid_numeric_mask
                malformed_count = len(chunk) - valid_mask.sum()
                total_rows_skipped_malformed += malformed_count

                if valid_mask.sum() == 0:
                    continue

                clean_chunk = pd.concat([numeric_df[valid_mask], chunk.loc[valid_mask, ["Label"]]], axis=1)

                # Split into Benign vs Attacks
                is_benign = clean_chunk["Label"].str.upper() == "BENIGN"
                benign_part = clean_chunk[is_benign]
                if not benign_part.empty:
                    benign_dfs.append(benign_part)

                attack_part = clean_chunk[~is_benign]
                if not attack_part.empty:
                    for atk_label, grp in attack_part.groupby("Label"):
                        if atk_label not in attack_dfs_by_type:
                            attack_dfs_by_type[atk_label] = []
                        attack_dfs_by_type[atk_label].append(grp)

        except Exception as e:
            print(f"[!] Error processing {filename}: {e}", file=sys.stderr)

    full_benign_df = pd.concat(benign_dfs, ignore_index=True) if benign_dfs else pd.DataFrame(columns=all_required_features + ["Label"])
    attack_dfs_combined = {
        atk_label: pd.concat(dfs, ignore_index=True)
        for atk_label, dfs in attack_dfs_by_type.items()
    }
    total_valid_attacks = sum(len(df) for df in attack_dfs_combined.values())

    print(f"\n[*] Scan Complete:")
    print(f"    - Total Rows Scanned:           {total_rows_scanned:,}")
    print(f"    - Valid Benign Rows Available:  {len(full_benign_df):,}")
    print(f"    - Valid Attack Rows Available:  {total_valid_attacks:,}")
    print(f"    - Skipped Malformed/NaN/Inf:    {total_rows_skipped_malformed:,}")
    print(f"    - Skipped Missing Features:     {total_rows_skipped_missing_features:,}")

    # Sample benign rows
    target_benign = min(target_per_class, len(full_benign_df))
    selected_benign_df = full_benign_df.sample(n=target_benign, random_state=random_seed, replace=False)

    # Sample attack rows distributed across all available attack types
    target_attacks = min(target_per_class, total_valid_attacks)
    num_attack_types = len(attack_dfs_combined)
    base_per_attack = max(1, target_attacks // num_attack_types)

    sampled_attack_list: List[pd.DataFrame] = []
    sampled_indices_by_type: Dict[str, set] = {}

    for atk_label, atk_df in attack_dfs_combined.items():
        sample_count = min(len(atk_df), base_per_attack)
        sample = atk_df.sample(n=sample_count, random_state=random_seed, replace=False)
        sampled_attack_list.append(sample)
        sampled_indices_by_type[atk_label] = set(sample.index)

    current_attack_count = sum(len(df) for df in sampled_attack_list)
    if current_attack_count < target_attacks:
        needed = target_attacks - current_attack_count
        # Pool all remaining attack samples
        remaining_list = []
        for atk_label, atk_df in attack_dfs_combined.items():
            used_idx = sampled_indices_by_type.get(atk_label, set())
            remaining_mask = ~atk_df.index.isin(used_idx)
            if remaining_mask.any():
                remaining_list.append(atk_df[remaining_mask])

        if remaining_list:
            all_remaining = pd.concat(remaining_list, ignore_index=True)
            extra_sample = all_remaining.sample(n=min(needed, len(all_remaining)), random_state=random_seed, replace=False)
            sampled_attack_list.append(extra_sample)

    selected_attacks_df = pd.concat(sampled_attack_list, ignore_index=True)

    # Combine and shuffle
    combined_df = pd.concat([selected_benign_df, selected_attacks_df], ignore_index=True)
    shuffled_df = combined_df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    ground_truth_labels = shuffled_df["Label"].tolist()
    ground_truth_binary = [0 if label.upper() == "BENIGN" else 1 for label in ground_truth_labels]

    features_df = shuffled_df[all_required_features]
    rows = features_df.to_dict(orient="records")

    attack_breakdown = {str(k): int(v) for k, v in selected_attacks_df["Label"].value_counts().to_dict().items()}

    stats = {
        "total_rows_scanned": int(total_rows_scanned),
        "total_rows_skipped_malformed": int(total_rows_skipped_malformed),
        "total_rows_skipped_missing_features": int(total_rows_skipped_missing_features),
        "total_skipped": int(total_rows_skipped_malformed + total_rows_skipped_missing_features),
        "total_sampled": int(len(shuffled_df)),
        "benign_sampled": int(len(selected_benign_df)),
        "attack_sampled": int(len(selected_attacks_df)),
        "attack_breakdown_sampled": attack_breakdown
    }

    return rows, ground_truth_labels, ground_truth_binary, stats


def evaluate_pipeline(
    rows: List[Dict[str, Any]],
    ground_truth_labels: List[str],
    ground_truth_binary: List[int],
    batch_size: int = 1000
) -> Dict[str, Any]:
    """
    Runs the ThreatPipeline on all sampled rows in batches and computes evaluation metrics.
    """
    print(f"\n[*] Running ThreatPipeline.predict_batch() on {len(rows)} samples in batches of {batch_size}...")
    pipeline_results: List[Dict[str, Any]] = []

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        batch_out = PIPELINE.predict_batch(batch)
        pipeline_results.extend(batch_out)

    y_true = np.array(ground_truth_binary)
    y_pred = np.array([1 if r["is_anomaly"] else 0 for r in pipeline_results])

    # Core metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()

    # Detailed analysis on detected anomalies
    anomaly_indices = [i for i, r in enumerate(pipeline_results) if r["is_anomaly"]]
    anomalies_by_gt: Dict[str, Dict[str, Any]] = {}

    for idx in anomaly_indices:
        gt = ground_truth_labels[idx]
        pred_type = pipeline_results[idx]["attack_type"]
        conf = float(pipeline_results[idx]["attack_confidence"])

        if gt not in anomalies_by_gt:
            anomalies_by_gt[gt] = {
                "detected_count": 0,
                "predicted_types": {},
                "confidences": []
            }

        anomalies_by_gt[gt]["detected_count"] += 1
        anomalies_by_gt[gt]["predicted_types"][pred_type] = (
            anomalies_by_gt[gt]["predicted_types"].get(pred_type, 0) + 1
        )
        anomalies_by_gt[gt]["confidences"].append(conf)

    # Sample anomalous rows for inspection
    sample_anomalies = []
    for idx in anomaly_indices[:15]:
        top_contrib = (
            pipeline_results[idx]["top_contributors"][0]["feature"]
            if pipeline_results[idx]["top_contributors"] else "N/A"
        )
        sample_anomalies.append({
            "ground_truth_label": ground_truth_labels[idx],
            "vae_score": round(float(pipeline_results[idx]["score"]), 4),
            "vae_threshold": round(float(pipeline_results[idx]["threshold"]), 4),
            "predicted_attack_type": pipeline_results[idx]["attack_type"],
            "attack_confidence": round(float(pipeline_results[idx]["attack_confidence"]), 4),
            "top_contributor": top_contrib
        })

    report = {
        "metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "matrix": cm.tolist()
        },
        "total_anomalies_flagged": int(len(anomaly_indices)),
        "anomalies_by_ground_truth": anomalies_by_gt,
        "sample_anomalous_predictions": sample_anomalies
    }

    return report


def print_report(stats: Dict[str, Any], report: Dict[str, Any]) -> None:
    """
    Prints a formatted validation report to stdout.
    """
    m = report["metrics"]
    cm = report["confusion_matrix"]

    print("\n" + "=" * 80)
    print("                      TRACE MODEL VALIDATION REPORT                      ")
    print("=" * 80)

    print("\n1. DATASET QUALITY & INTEGRITY SUMMARY")
    print("-" * 55)
    print(f"  * Total Rows Scanned:           {stats['total_rows_scanned']:,}")
    print(f"  * Malformed / NaN / Inf Rows:   {stats['total_rows_skipped_malformed']:,}")
    print(f"  * Missing Columns Rows:         {stats['total_rows_skipped_missing_features']:,}")
    print(f"  * Total Rows Skipped:           {stats['total_skipped']:,}")
    print(f"  * Total Valid Evaluated Sample: {stats['total_sampled']:,}")
    print(f"    - Benign Samples:             {stats['benign_sampled']:,}")
    print(f"    - Attack Samples:             {stats['attack_sampled']:,}")

    print("\n2. ATTACK SAMPLES BREAKDOWN IN TEST SET")
    print("-" * 55)
    for atk_name, count in sorted(stats["attack_breakdown_sampled"].items(), key=lambda x: -x[1]):
        if count > 0:
            safe_name = atk_name.encode("ascii", errors="replace").decode("ascii")
            print(f"  - {safe_name:<35}: {count:>5} samples")

    print("\n3. VAE ANOMALY DETECTION METRICS (Ground-Truth vs VAE is_anomaly)")
    print("-" * 55)
    print(f"  * Accuracy:                     {m['accuracy'] * 100:.2f}%")
    print(f"  * Precision:                    {m['precision'] * 100:.2f}%")
    print(f"  * Recall:                       {m['recall'] * 100:.2f}%")
    print(f"  * F1 Score:                     {m['f1_score'] * 100:.2f}%")

    print("\n4. CONFUSION MATRIX")
    print("-" * 55)
    print(f"                    Predicted Benign (0)    Predicted Anomaly (1)")
    print(f"  Actual Benign (0)        {cm['true_negatives']:>6} (TN)              {cm['false_positives']:>6} (FP)")
    print(f"  Actual Attack (1)        {cm['false_negatives']:>6} (FN)              {cm['true_positives']:>6} (TP)")
    print(f"\n  Breakdown:")
    print(f"    * True Negatives (TN):  {cm['true_negatives']:,}  (Benign correctly classified)")
    print(f"    * False Positives (FP): {cm['false_positives']:,}  (Benign falsely flagged as anomaly)")
    print(f"    * False Negatives (FN): {cm['false_negatives']:,}  (Attacks missed by VAE)")
    print(f"    * True Positives (TP):  {cm['true_positives']:,}  (Attacks successfully detected)")

    print("\n5. DETECTED ANOMALIES BREAKDOWN BY GROUND-TRUTH LABEL")
    print("-" * 80)
    print(f"{'Ground Truth Label':<30} | {'Detected':<8} | {'Avg Conf':<8} | {'MoE Predictions'}")
    print("-" * 80)

    for gt_label, data in sorted(report["anomalies_by_ground_truth"].items(), key=lambda x: -x[1]["detected_count"]):
        cnt = data["detected_count"]
        confs = data["confidences"]
        avg_conf = np.mean(confs) if confs else 0.0
        pred_types_str = ", ".join([f"{k}: {v}" for k, v in data["predicted_types"].items()])
        safe_gt = gt_label.encode("ascii", errors="replace").decode("ascii")
        print(f"{safe_gt:<30} | {cnt:>8} | {avg_conf:>7.2%} | {pred_types_str}")

    print("\n6. SAMPLE ANOMALOUS DETECTIONS")
    print("-" * 80)
    print(f"{'Ground Truth':<22} | {'Score':<7} | {'Thr':<6} | {'Predicted MoE Type':<16} | {'Conf':<6} | {'Top Contributor'}")
    print("-" * 80)
    for s in report["sample_anomalous_predictions"]:
        safe_gt = s['ground_truth_label'].encode("ascii", errors="replace").decode("ascii")
        print(f"{safe_gt:<22} | {s['vae_score']:>7.2f} | {s['vae_threshold']:>6.2f} | {s['predicted_attack_type']:<16} | {s['attack_confidence']:>6.2%} | {s['top_contributor']}")

    print("=" * 80 + "\n")


def json_serializable(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def main():
    parser = argparse.ArgumentParser(description="TRACE ThreatPipeline Model Validation Test")
    parser.add_argument("--samples-per-class", type=int, default=2500,
                        help="Target number of samples per class (benign vs attack), total = 2x")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Batch size for ThreatPipeline.predict_batch()")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Optional filepath to save JSON validation results")
    args = parser.parse_args()

    rows, gt_labels, gt_binary, stats = load_dataset_samples(
        csv_folder=CSV_FOLDER,
        target_per_class=args.samples_per_class,
        random_seed=args.seed
    )

    report = evaluate_pipeline(
        rows=rows,
        ground_truth_labels=gt_labels,
        ground_truth_binary=gt_binary,
        batch_size=args.batch_size
    )

    print_report(stats, report)

    if args.output_json:
        full_output = {
            "dataset_stats": stats,
            "validation_report": report
        }
        with open(args.output_json, "w") as f:
            json.dump(full_output, f, indent=2, default=json_serializable)
        print(f"[*] Validation report saved to {args.output_json}")


if __name__ == "__main__":
    main()
