import os
import sys
import glob
import json
import numpy as np
import pandas as pd
from ml.features import CICIDS_COLUMN_MAPPING
from ml.pipeline import vae, vae_feature_names

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_FOLDER = os.path.join(
    PROJECT_ROOT,
    "solarmainframe",
    "ids-intrusion-csv",
    "versions",
    "1"
)
REPORTS_DIR = os.path.join(PROJECT_ROOT, "ml", "reports")

THRESHOLD = 38.50959014892578

LABEL_MAPPING = {
    "BENIGN": "BENIGN",
    "DDoS": "DDoS",
    "DoS Hulk": "DoS Hulk",
    "DoS GoldenEye": "DoS GoldenEye",
    "DoS slowloris": "DoS slowloris",
    "DoS Slowhttptest": "DoS Slowhttptest",
    "PortScan": "PortScan",
    "Bot": "Bot",
    "FTP-Patator": "FTP-Patator",
    "SSH-Patator": "SSH-Patator",
    "Infiltration": "Infiltration",
    "Heartbleed": "Heartbleed",
}

def normalize_label(raw_label: str) -> str:
    raw = str(raw_label).strip()
    if raw in LABEL_MAPPING:
        return LABEL_MAPPING[raw]
    raw_lower = raw.lower()
    if "brute force" in raw_lower:
        return "Web Attack (Brute Force)"
    elif "xss" in raw_lower:
        return "Web Attack (XSS)"
    elif "sql injection" in raw_lower:
        return "Web Attack (Sql Injection)"
    elif "infiltration" in raw_lower:
        return "Infiltration"
    elif "heartbleed" in raw_lower:
        return "Heartbleed"
    elif "benign" in raw_lower:
        return "BENIGN"
    return raw

def format_table(headers, rows):
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))
    
    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "|-" + "-|-".join("-" * col_widths[i] for i in range(len(headers))) + "-|"
    data_lines = [
        "| " + " | ".join(str(val).rjust(col_widths[i]) if i > 0 and i < len(row)-1 else str(val).ljust(col_widths[i]) for i, val in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line] + data_lines)

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = vae.device
    model = vae.model
    scaler = vae.scaler
    feature_names = vae_feature_names

    print(f"[*] VAE Device: {device}")
    print(f"[*] Evaluation Threshold: {THRESHOLD}")
    print(f"[*] Feature count: {len(feature_names)}")

    csv_files = sorted(glob.glob(os.path.join(CSV_FOLDER, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {CSV_FOLDER}")

    scores_by_label = {}
    total_scanned = 0
    total_valid = 0
    total_malformed = 0

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        print(f"[*] Scanning {filename}...", flush=True)

        chunk_size = 100000
        for chunk in pd.read_csv(filepath, chunksize=chunk_size, encoding="cp1252", low_memory=False):
            total_scanned += int(len(chunk))
            chunk.columns = chunk.columns.str.strip()
            chunk.rename(columns=CICIDS_COLUMN_MAPPING, inplace=True)

            if "Label" not in chunk.columns:
                continue

            missing_cols = [c for c in feature_names if c not in chunk.columns]
            if missing_cols:
                continue

            chunk["Label"] = chunk["Label"].astype(str).str.strip()
            valid_label_mask = chunk["Label"].notna() & (chunk["Label"] != "") & (chunk["Label"] != "nan")

            numeric_df = chunk[feature_names].apply(pd.to_numeric, errors="coerce")
            valid_numeric_mask = numeric_df.notna().all(axis=1) & np.isfinite(numeric_df).all(axis=1)

            valid_mask = valid_label_mask & valid_numeric_mask
            malformed_count = int(len(chunk) - valid_mask.sum())
            total_malformed += malformed_count

            if valid_mask.sum() == 0:
                continue

            clean_features = numeric_df[valid_mask].values.astype(np.float32)
            clean_labels = chunk.loc[valid_mask, "Label"].apply(normalize_label).values

            total_valid += int(len(clean_features))

            # Transform with scaler
            X_scaled = scaler.transform(clean_features).astype(np.float32)

            # Score in sub-batches
            sub_batch_size = 20000
            for sb_start in range(0, len(X_scaled), sub_batch_size):
                sb_X = X_scaled[sb_start:sb_start + sub_batch_size]
                sb_lbls = clean_labels[sb_start:sb_start + sub_batch_size]

                with torch.no_grad():
                    xb = torch.from_numpy(sb_X).to(device)
                    recon, mu, logvar = model(xb)
                    se = (recon - xb) ** 2
                    sb_scores = se.sum(dim=1).cpu().numpy()

                for lbl, sc in zip(sb_lbls, sb_scores):
                    if lbl not in scores_by_label:
                        scores_by_label[lbl] = []
                    scores_by_label[lbl].append(sc)

    print(f"\n[*] Processing Complete:")
    print(f"    - Total Rows Scanned: {total_scanned:,}")
    print(f"    - Total Valid Rows:   {total_valid:,}")
    print(f"    - Total Malformed:    {total_malformed:,}")

    # Desired order
    desired_order = [
        "BENIGN",
        "DDoS",
        "DoS Hulk",
        "DoS GoldenEye",
        "DoS slowloris",
        "DoS Slowhttptest",
        "PortScan",
        "Bot",
        "FTP-Patator",
        "SSH-Patator",
        "Infiltration",
        "Web Attack (Brute Force)",
        "Web Attack (XSS)",
        "Web Attack (Sql Injection)",
        "Heartbleed"
    ]

    all_labels = [l for l in desired_order if l in scores_by_label]
    for l in sorted(scores_by_label.keys()):
        if l not in all_labels:
            all_labels.append(l)

    results_list = []
    table_rows = []

    for lbl in all_labels:
        sc_arr = np.array(scores_by_label[lbl], dtype=np.float64)
        n_samples = int(len(sc_arr))
        mean_val = float(np.mean(sc_arr))
        median_val = float(np.median(sc_arr))
        p95_val = float(np.percentile(sc_arr, 95))
        min_val = float(np.min(sc_arr))
        max_val = float(np.max(sc_arr))
        
        above_thresh_count = int(np.sum(sc_arr > THRESHOLD))
        pct_above = float((above_thresh_count / n_samples) * 100.0) if n_samples > 0 else 0.0

        is_benign = (lbl.upper() == "BENIGN")
        rate_type = "False-Positive Rate (FPR)" if is_benign else "Detection Rate (Recall/TPR)"

        results_list.append({
            "label": lbl,
            "sample_count": n_samples,
            "mean_score": mean_val,
            "median_score": median_val,
            "p95_score": p95_val,
            "min_score": min_val,
            "max_score": max_val,
            "score_gt_threshold_pct": pct_above,
            "metric_type": rate_type,
            "rate_pct": pct_above
        })

        table_rows.append([
            lbl,
            f"{n_samples:,}",
            f"{mean_val:.2f}",
            f"{median_val:.2f}",
            f"{p95_val:.2f}",
            f"{min_val:.2f}",
            f"{max_val:.2f}",
            f"{pct_above:.2f}%",
            "FPR" if is_benign else "TPR (Detection)"
        ])

    # Save JSON
    os.makedirs(REPORTS_DIR, exist_ok=True)
    json_path = os.path.join(REPORTS_DIR, "vae_analysis_report.json")
    with open(json_path, "w") as f:
        json.dump({
            "threshold": float(THRESHOLD),
            "summary_stats": {
                "total_rows_scanned": int(total_scanned),
                "total_valid_rows": int(total_valid),
                "total_malformed_rows": int(total_malformed)
            },
            "categories": results_list
        }, f, indent=2)
    print(f"\n[+] Saved JSON report to: {json_path}")

    # Save CSV
    csv_path = os.path.join(REPORTS_DIR, "vae_analysis_report.csv")
    pd.DataFrame(results_list).to_csv(csv_path, index=False)
    print(f"[+] Saved CSV report to: {csv_path}")

    # Print Table
    headers = [
        "Label Category", "Sample Count", "Mean Score", "Median",
        "95th Pct", "Min", "Max", "% > Thr (38.51)", "Rate Type"
    ]
    print("\n" + "="*115)
    print(f"VAE ANOMALY DETECTOR EVALUATION REPORT (Threshold = {THRESHOLD:.4f})")
    print("="*115)
    print(format_table(headers, table_rows))
    print("="*115)

if __name__ == "__main__":
    main()
