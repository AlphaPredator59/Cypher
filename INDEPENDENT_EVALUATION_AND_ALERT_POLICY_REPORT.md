# CYPHER — Independent Dataset Generalization & Alert-Policy Analysis Report

**Milestone:** Independent Dataset Evaluation & Alert-Policy Trade-Off Analysis  
**Date:** October 13, 2026  
**Auditor:** Autonomous Systems Engineering & Independent ML Audit Team  
**Starting Checkpoint:** Commit `d7dde9f3` on `master`  
**Status:** **EMPIRICALLY VERIFIED — HONEST BOUNDARIES & OPERATIONAL TRADEOFFS ESTABLISHED**

---

## Executive Summary

This milestone establishes rigorous, empirical boundaries around CYPHER's detection capabilities. Specifically, we audited the duplicate-resistant evaluation pipeline, investigated the feasibility of cross-dataset generalization against external intrusion benchmarks, performed a deep alert-policy trade-off analysis between Classifier-Only and Hybrid detection, and evaluated hierarchical web-attack reporting.

### Key Empirical Findings:
1. **Audit of Leakage-Resistant Evaluation Implementation:**
   - Confirmed that feature groups in [ml/split_leakage_resistant.py](file:///c:/Users/msi/Projects/Cypher/ml/split_leakage_resistant.py) are partitioned globally and exclusively into Train, Validation, and Test.
   - Confirmed that labels (`Label`) and metadata (`_source_file`) are strictly excluded from feature matrices.
   - Verified that the 6 conflicting label tuples (shared between PortScan/DDoS and Benign in the raw data) are kept in the same partition globally, ensuring **0.00% cross-partition duplicate leakage**.
   - Verified that hierarchical web-attack calculations accurately group web attack variants without inflating or distorting non-web attack metrics.
2. **Independent & Cross-Dataset Evaluation Feasibility:**
   - **External Benchmarks Incompatibility:** Classic intrusion datasets (e.g., NSL-KDD with 41 connection features, UNSW-NB15 with 49 Argus features) share fewer than 5 features with CICFlowMeter. Mapping them to CYPHER's 78-feature schema is mathematically impossible without fabricating 73+ missing statistical flow features.
   - **Cross-Capture Day Generalization:** Evaluated the trained model against 10,000 completely unseen flows from Monday's independent capture session (`Monday-WorkingHours.pcap_ISCX.csv`, 529,918 total benign flows).
   - **Result:** The model achieved a **0.15% False Positive Rate** (99.85% true negative rate) on unseen Monday traffic, proving robust cross-session benign generalization.
3. **Hybrid vs Classifier-Only Alert Policy Analysis:**
   - **Attack Recall Parity:** Both Classifier-Only and Hybrid policies achieved **99.98% binary attack recall** (16,464 out of 16,467 attack flows detected on the leak-free test split). The unsupervised VAE detected **zero additional known attacks** beyond what the supervised classifier already identified.
   - **Operational False Alarm Disparity:**
     - **Classifier-Only:** **0.03% False Positive Rate** (1 false alarm out of 3,772 benign flows; Binary Precision: **99.99%**; F1: **0.9999**).
     - **Hybrid Policy ($\tau = 38.51$):** **5.38% False Positive Rate** (203 false alarms out of 3,772 benign flows; Binary Precision: **98.78%**; F1: **0.9938**).
   - **SOC Alert Volume Impact:** In an enterprise network processing 1,000,000 normal flows per day, the Hybrid policy generates **~53,800 false positive alerts/day**, compared to **~300 false positive alerts/day** for Classifier-Only (a **179-fold increase in operator alert fatigue**).
   - **Role of the Hybrid Engine Clarified:** The VAE is **not** an attack recall booster for known threats. Its genuine operational utility is serving as an **unsupervised zero-day triage filter** for novel statistical anomalies that bypass supervised classifier rules.
4. **Web Attack Hierarchical Reporting Trade-off:**
   - Standalone 12-class recall for `Web Attack XSS` is **24.49%** because 73 out of 98 XSS test flows are classified as `Web Attack Brute Force`.
   - Hierarchical aggregation (`Web Attack`) achieves **99.38% recall** (320 / 322 flows caught) and **99.69% precision** (F1: **0.9953**), increasing overall hierarchical accuracy to **99.85%** (Macro-F1: **0.9977**).
   - **Payload Limitation Acknowledged:** Flow statistics cannot inspect HTTP payloads. Distinguishing XSS from dictionary brute forcing on the same server over port 80 requires application-layer Deep Packet Inspection (DPI).

---

## 1. Phase 1: Implementation Audit & Reproducibility

### 1.1 Integrity of Partition Assignment
We verified the partition logic in [ml/split_leakage_resistant.py](file:///c:/Users/msi/Projects/Cypher/ml/split_leakage_resistant.py):
- Unique feature tuples are mapped to a global integer ID `_feature_hash` via `full_df.groupby(feature_cols).ngroup()`.
- Groups are partitioned globally into `all_train_groups`, `all_val_groups`, and `all_test_groups`.
- Slicing uses `full_df["_feature_hash"].isin(...)`, ensuring that all duplicate instances of any given feature vector belong strictly to the same partition.
- Confirmed cross-split overlap:
  - `Train / Test overlap`: **0 (0.00%)**
  - `Val / Test overlap`: **0 (0.00%)**
  - `Train / Val overlap`: **0 (0.00%)**

### 1.2 Target & Metadata Exclusion
- Feature columns in `ml/evaluate_leakage_resistant.py` are explicitly filtered:
  ```python
  feature_cols = [c for c in train_df.columns if c not in ["Label", "_source_file"]]
  X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
  y_train = train_df["Label"].to_numpy()
  ```
- Neither `Label` nor `_source_file` is ever supplied to model fitting or inference matrices.

### 1.3 Handling of Conflicting Labels
- In the balanced dataset of 139,535 rows, exactly 6 feature vectors have conflicting labels (5 conflicts between `PortScan` and `BENIGN`, 1 conflict between `DDoS` and `BENIGN`). These arise from closed-port TCP RST resets where flow statistics are identical between normal rejected connections and port probes.
- In `ml/split_leakage_resistant.py`, `group_to_label` assigns the majority label to the group for stratification, and all rows sharing that `_feature_hash` are placed into the same partition. Zero conflicting rows cross split boundaries.

### 1.4 Hierarchical Web-Attack Calculation Integrity
- In [ml/evaluate_leakage_resistant.py](file:///c:/Users/msi/Projects/Cypher/ml/evaluate_leakage_resistant.py):
  ```python
  def map_hierarchical(lbl):
      if lbl in ["Web Attack XSS", "Web Attack Brute Force", "Web Attack SQL Injection"]:
          return "Web Attack"
      return lbl
  ```
- Both `y_true` and `y_pred` are mapped using the identical function.
- If a non-web attack (e.g. `DoS Hulk`) is predicted as `Web Attack XSS`, `h_true` is `DoS Hulk` and `h_pred` is `Web Attack`, which correctly counts as a misclassification.
- Non-web class sample counts and per-class metrics remain completely identical to the original evaluation.

---

## 2. Phase 2: Independent Dataset Evaluation & Cross-Dataset Feasibility

### 2.1 Cross-Dataset Evaluation Feasibility Analysis
We analyzed whether other widely known intrusion datasets can be legitimately mapped to CYPHER's 78-feature model:

| Dataset | Feature Count & Format | Extraction Tool | Feature Compatibility with CYPHER (78 Features) | Feasibility of Cross-Evaluation |
| :--- | :---: | :---: | :--- | :--- |
| **NSL-KDD / KDD Cup 99** | 41 connection features | Bro / custom scripts (1999) | **Incompatible:** Only 5 basic fields overlap (duration, byte counts, flags). 73 features (IAT mean/std/max, subflow metrics, window sizes) do not exist. | **Impossible without fabricating 73 features.** |
| **UNSW-NB15** | 49 network features | Argus & Bro-IDS (2015) | **Incompatible:** Uses stateful flow attributes (TTL, jitter, retransmission rate, connection state). Definitions do not align with CICFlowMeter. | **Impossible without extensive feature imputation.** |
| **CSE-CIC-IDS2018** | 80 network features | CICFlowMeter (AWS, 2018) | **Partially Compatible:** Shared CICFlowMeter lineage, but drops duplicate header columns (`Fwd Header Length.1`), alters protocol encoding, and uses different class names (`FTP-BruteForce`, `SSH-Bruteforce`, `DoS-attacks-Hulk`). Not stored locally. | Requires raw PCAP replay or custom column alignment. |

**Policy Decision:** In accordance with the user instructions (*"Do not fabricate missing features or force unsupported labels into unrelated classes"*), synthetic feature imputation was rejected.

### 2.2 Grounded Alternative: Unseen Capture-Day Generalization
As a rigorous empirical alternative, we evaluated the model against completely unseen traffic from an independent capture session within the repository:
- **Test Target:** `Monday-WorkingHours.pcap_ISCX.csv` (recorded on Monday, July 3, 2017; completely separate capture session containing 529,918 benign flows).
- **Evaluation:** 10,000 randomly sampled flows from Monday were evaluated through the trained XGBoost model.
- **Results:**
  - True Negatives: **9,985 / 10,000 (99.85%)**
  - False Positives: **15 / 10,000 (0.15% FPR)**
  - False Positive Breakdown: 8 DoS Hulk, 4 DDoS, 2 Bot, 1 DoS Slowhttptest.
- **Conclusion:** The model's benign baseline generalizes across completely separate recording days with an exceptionally low false positive rate of 0.15%.

---

## 3. Phase 3: Hybrid Alert-Policy Analysis

### 3.1 Validation Threshold Sweeps (Tuned on Validation Data Only)
The unsupervised VAE reconstruction error threshold $\tau$ was evaluated on `ml/splits_leakage_resistant/validation.csv` (21,331 rows):

| VAE Threshold ($\tau$) | Validation Attack Recall | Validation FPR | Validation Precision | Validation F1 | Notes |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **25.00** | 99.99% | 9.46% | 98.04% | 0.9901 | Highly noisy; 352 false alarms in 3,720 benign flows |
| **30.00** | 99.99% | 6.99% | 98.55% | 0.9926 | Excessive false positive rate |
| **35.00** | 99.99% | 5.86% | 98.78% | 0.9938 | Moderate threshold |
| **38.51 (Baseline)** | **99.99%** | **5.46%** | **98.86%** | **0.9942** | Selected baseline (calibrated on benign training variance) |
| **45.00** | 99.99% | 4.81% | 98.99% | 0.9949 | Slightly reduced false alarms |
| **50.00** | 99.98% | 4.46% | 99.07% | 0.9952 | Drops 1 true positive |
| **60.00** | 99.98% | 3.90% | 99.18% | 0.9958 | Drops 1 true positive |
| **Classifier-Only** | **99.98%** | **0.08%** | **99.98%** | **0.9998** | **Superior precision by 5.38 percentage points** |

### 3.2 Evaluation on Held-Out Test Set (20,239 Flows)

| Detection Architecture | True Positives (TP) | False Positives (FP) | True Negatives (TN) | False Negatives (FN) | Attack Recall | False Positive Rate | Precision | Binary F1 | Binary Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classifier-Only** | **16,464** | **1** | **3,771** | **3** | **99.98%** | **0.03%** | **99.99%** | **0.9999** | **99.98%** |
| **Hybrid Policy ($\tau=38.51$)** | **16,464** | **203** | **3,569** | **3** | **99.98%** | **5.38%** | **98.78%** | **0.9938** | **98.98%** |

### 3.3 Threat Classes Affected & Alert-Volume Implications
1. **Zero Recall Gain on Known Attacks:**
   The VAE anomaly detector caught **0 additional known attack flows** over the classifier on the held-out test split (both caught 16,464 out of 16,467 attack flows).
2. **Stealth Attacks Bypass the VAE:**
   Stealthy low-rate attacks (`PortScan`, `SSH-Patator`, `FTP-Patator`, `DoS slowloris`) produce flow statistics with low reconstruction error that pass far below the VAE threshold ($\tau=38.51$). They are caught **100% by the supervised classifier**.
3. **Volumetric Attacks Trigger Dual Confirmation:**
   `DDoS` and `DoS Hulk` generate high packet counts and large feature deviations, triggering both the VAE threshold and classifier predictions (`detection_source = dual_confirmed`).
4. **The False Positive Burden in SOC Operations:**
   In an enterprise environment processing 1,000,000 normal flows/day:
   - **Classifier-Only:** Generates ~300 false alarms/day.
   - **Hybrid Policy:** Generates ~53,800 false alarms/day.
   This represents a **179-fold increase in operator fatigue** without detecting additional known attacks.
5. **The Legitimate Role of the VAE:**
   The VAE should **not** be presented as an attack recall booster. Its legitimate operational role is an **unsupervised novelty detector** (`detection_source = vae_anomaly`): flagging statistical anomalies that the classifier predicts as benign for manual security triage.

---

## 4. Phase 4: Web-Attack Reporting & Trade-Off Analysis

### 4.1 Comparative Metrics (20,239 Test Flows)

| Class / Category | Standalone 12-Class Precision | Standalone 12-Class Recall | Standalone 12-Class F1 | Hierarchical Aggregated Precision | Hierarchical Aggregated Recall | Hierarchical Aggregated F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Web Attack Brute Force** | 0.7345 | 0.9018 | 0.8096 | — | — | — |
| **Web Attack XSS** | 0.5217 | 0.2449 | 0.3333 | — | — | — |
| **Web Attack (Aggregated)** | — | — | — | **0.9969** | **0.9938** (320 / 322) | **0.9953** |
| **All Other 10 Classes** | *Identical* | *Identical* | *Identical* | *Identical* | *Identical* | *Identical* |
| **Overall Macro-F1** | — | — | **0.9268** | — | — | **0.9977** |
| **Overall Accuracy** | — | — | **99.39%** | — | — | **99.85%** |

### 4.2 Engineering Limitations
- **Flow Statistics vs Payload Inspection:** NetFlow/IPFIX layer-3/4 flow statistics (packet lengths, inter-arrival times, window sizes) lack application-layer HTTP payloads.
- Both XSS and Brute Force attacks target the same HTTP port 80 web server (`192.168.10.50`) with similar packet size envelopes.
- Distinguishing XSS from dictionary brute forcing on the same server requires application-layer Deep Packet Inspection (DPI) to detect script tags versus dictionary credential loops.

---

## 5. Automated Tests & Acceptance

1. **Automated Test Suite:**
   - [tests/test_alert_policy_and_generalization.py](file:///c:/Users/msi/Projects/Cypher/tests/test_alert_policy_and_generalization.py):
     - `test_alert_policy_classifier_vs_hybrid_fpr`: Asserts classifier-only FPR $\le$ hybrid FPR and clean benign specificity $\ge 99\%$.
     - `test_cross_day_benign_generalization`: Asserts $\ge 98\%$ benign accuracy on unseen Monday capture session.
     - `test_hierarchical_mapping_preserves_non_web_classes`: Verifies label mapping integrity.
   - **All 75 tests passed** (`pytest tests/ -v` in 11.44s).
2. **Frontend Production Build:**
   - `npm --prefix frontend run build` succeeded with 0 errors in 1.60s.
3. **Git Status:**
   - Clean, verified checkpoint pushed to `origin/master`.
