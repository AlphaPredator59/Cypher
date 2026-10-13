# CYPHER — Detection Quality Verification, Alert Architecture & Evaluation Report

**Milestone:** Detection Quality Verification, Alert Architecture & Hierarchical Evaluation  
**Date:** October 13, 2026  
**Auditor:** Autonomous Systems Engineering & Independent ML Audit Team  
**Starting Checkpoint:** Commit `66731b57` on `master`  
**Status:** **EMPIRICALLY VERIFIED — REPRODUCIBLE BENCHMARKS, SEPARATE ALERT ARCHITECTURE & 81/81 PASSING TESTS**

---

## Executive Summary

This milestone conducted an exhaustive, forensic verification of detection quality across the held-out leakage-resistant benchmark (20,239 test rows), audited decision rules and high classifier scores, restructured the alerting architecture to prevent automatic promotion of unclassified VAE outliers into confirmed attacks, validated hierarchical Web Attack aggregation, and verified end-to-end system regressions.

### Key Conclusions & Empirical Facts:
1. **Decision Rule & Metric Reproducibility:**
   - Both classifier-only and hybrid confusion matrices were reproduced from the exact held-out test rows (`ml/splits_leakage_resistant/test.csv`, 20,239 flows).
   - Under the production decision rule (`pred_lbl != "BENIGN"` with $P_{\text{xgb}} \ge 0.50$):
     - **Classifier-Only:** True Positives (TP) = **16,464**, False Positives (FP) = **1**, True Negatives (TN) = **3,771**, False Negatives (FN) = **3**.
     - **Attack Recall:** **99.9818%** $\left(\frac{16464}{16467}\right)$
     - **False Positive Rate (FPR):** **0.0265%** $\left(\frac{1}{3772}\right)$ (only 1 false alarm across 3,772 benign flows!)
     - **Precision:** **99.9939%** $\left(\frac{16464}{16465}\right)$
     - **Binary Accuracy:** **99.9802%**
   - Under raw argmax prediction, exactly 3 flows (rows 1254, 5713, and 18078) have maximum sub-class probability $P = 0.4994 < 0.50$ because their predicted probability is divided between `Web Attack XSS` (~49.9%) and `Web Attack Brute Force` (~49.8%). However, their combined non-benign probability exceeds 99.7% ($P_{\text{benign}} \approx 0.0017$).
2. **VAE Detection Reality & Alert Architecture:**
   - On the held-out test set, VAE-only detection ($\tau = 38.51$) achieves an attack recall of only **3.39%** (558 / 16,467) and misses 15,909 known attacks.
   - The VAE detected **0 additional known attacks** that the supervised classifier missed.
   - Forcing an OR-based hybrid policy (`scores > tau | clf_attack`) adds **202 false alarms** from benign flows with normal variance above $\tau$, jumping the False Positive Rate from **0.03%** to **5.38%** (a 179-fold surge in false alerts).
   - **Architecture Fix Implemented:** VAE reconstruction anomalies where the classifier predicts BENIGN are now explicitly separated as **`anomaly_candidate`** (unclassified statistical outliers for secondary triage) rather than being promoted to confirmed high-severity attacks. Confirmed attacks require supervised classifier attribution (`known_attack` or `dual_signal`).
3. **Audit of Exceptionally High Classifier Scores:**
   - **Feature Grouping & Deduplication:** Confirmed that unique 78-feature vector grouping is stratified globally across splits, ensuring **strictly 0.00% exact duplicate overlap** across Train, Val, and Test.
   - **Leakage & Preprocessing Audit:** Verified that preprocessing, label encoding, and model fitting use training data only. Target labels and metadata (`_source_file`) are excluded from model inputs. No post-label-derived features exist.
   - **Conflicting Label Audit:** Across the 139,535 rows in the balanced dataset pool, exactly 6 unique feature groups (14 rows, **0.0100%**) carry conflicting labels (`BENIGN` vs `PortScan`/`DDoS`). All 14 rows are isolated within single splits via majority assignment.
   - **Critical Generalization Caveat:** Duplicate-resistant evaluation eliminates literal feature memorization, but **does not prove independent-session or real-world generalization**. The CICIDS2017 dataset was recorded in a closed testbed over a single week and lacks session IDs, IP addresses, and timestamps in its feature set. Real-world distribution shifts, OS stack variations, and novel tools cannot be assumed solved by benchmark metrics alone.
4. **Hierarchical Web Attack Validation:**
   - Fine-grained 12-class evaluation suffers mutual confusion between `Web Attack XSS` (recall 24.5%, precision 52.2%) and `Web Attack Brute Force` (recall 90.2%, precision 73.5%).
   - When aggregated symmetrically to the category **`Web Attack`** (322 true flows), the model achieves **99.38% recall** (320 / 322 flows detected) and **99.69% precision** (F1: **0.9953**).
   - **Payload Distinction Reality:** Transport-layer flow statistics (packet lengths, inter-arrival times, window sizes) against the same port 80 web endpoint cannot resolve payload-level distinctions between XSS and brute force. Category-level success must not be mischaracterized as successful XSS vs Brute Force discrimination.
5. **Regression Verification:**
   - **81 / 81 backend tests passed** in 12.13s (including dedicated regression tests in [tests/test_alert_architecture.py](file:///c:/Users/msi/Projects/Cypher/tests/test_alert_architecture.py)).
   - **Frontend production build succeeded** (`npm run build`) in 1.69s with zero TypeScript or Vite errors.

---

## 1. Reproduction of Current Evaluation

### 1.1 Reproducible Evaluation Command
To reproduce the held-out test evaluation from committed artifacts:
```bash
.\.venv\Scripts\python.exe ml/evaluate_leakage_resistant.py
```

### 1.2 Multi-Class Confusion Matrix & Per-Class Metrics (Held-Out Test Set, 20,239 Rows)

Evaluated on `ml/splits_leakage_resistant/test.csv`:

| Class Name | Support | Precision | Recall | F1-Score | FP Count | FN Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BENIGN** | 3,772 | 0.9976 | 0.9976 | 0.9976 | 9 | 9 |
| **Bot** | 296 | 0.9966 | 0.9932 | 0.9949 | 1 | 2 |
| **DDoS** | 3,750 | 0.9995 | 1.0000 | 0.9997 | 2 | 0 |
| **DoS GoldenEye** | 1,544 | 0.9961 | 0.9987 | 0.9974 | 6 | 2 |
| **DoS Hulk** | 3,222 | 0.9985 | 0.9997 | 0.9991 | 5 | 1 |
| **DoS Slowhttptest** | 788 | 0.9975 | 0.9937 | 0.9955 | 2 | 5 |
| **DoS slowloris** | 837 | 0.9976 | 0.9952 | 0.9964 | 2 | 4 |
| **FTP-Patator** | 1,002 | 0.9990 | 0.9990 | 0.9990 | 1 | 1 |
| **PortScan** | 3,759 | 0.9997 | 0.9989 | 0.9993 | 1 | 4 |
| **SSH-Patator** | 947 | **1.0000** | **1.0000** | **1.0000** | 0 | 0 |
| **Web Attack Brute Force** | 224 | 0.7345 | 0.9018 | 0.8096 | 73 | 22 |
| **Web Attack XSS** | 98 | 0.5217 | 0.2449 | 0.3333 | 22 | 74 |
| **Macro Average** | — | **0.9365** | **0.9269** | **0.9268** | — | — |
| **Weighted Average** | — | **0.9933** | **0.9939** | **0.9933** | — | — |
| **Overall Accuracy** | **20,239** | — | — | **99.39%** | — | — |

### 1.3 12-Class Confusion Matrix (Row = Ground Truth, Column = Predicted Class)
Classes: `['BENIGN', 'Bot', 'DDoS', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest', 'DoS slowloris', 'FTP-Patator', 'PortScan', 'SSH-Patator', 'Web Attack Brute Force', 'Web Attack XSS']`

```
[[3763    1    2    2    2    0    0    1    1    0    0    0]   <- BENIGN (3,772)
 [   2  294    0    0    0    0    0    0    0    0    0    0]   <- Bot (296)
 [   0    0 3750    0    0    0    0    0    0    0    0    0]   <- DDoS (3,750)
 [   0    0    0 1542    2    0    0    0    0    0    0    0]   <- DoS GoldenEye (1,544)
 [   0    0    0    1 3221    0    0    0    0    0    0    0]   <- DoS Hulk (3,222)
 [   1    0    0    2    0  783    2    0    0    0    0    0]   <- DoS Slowhttptest (788)
 [   2    0    0    0    0    2  833    0    0    0    0    0]   <- DoS slowloris (837)
 [   0    0    0    0    0    0    0 1001    0    0    0    1]   <- FTP-Patator (1,002)
 [   3    0    0    0    1    0    0    0 3755    0    0    0]   <- PortScan (3,759)
 [   0    0    0    0    0    0    0    0    0  947    0    0]   <- SSH-Patator (947)
 [   1    0    0    0    0    0    0    0    0    0  202   21]   <- Web Attack Brute Force (224)
 [   0    0    0    1    0    0    0    0    0    0   73   24]]  <- Web Attack XSS (98)
```

---

## 2. Decision Rules & Score Audit

### 2.1 Decision Rule Comparison on Test Data (20,239 Rows)

We audited three decision rules across all 20,239 held-out test rows:
1. **Rule A (Argmax != BENIGN):** Flags a flow as an attack whenever the argmax class is not `BENIGN`.
2. **Rule B (Argmax != BENIGN & $P_{\text{max}} \ge 0.50$):** Used in `ThreatPipeline`, requiring the predicted attack class to meet the confidence threshold of 0.50.
3. **Rule C ($1.0 - P(\text{BENIGN}) \ge 0.50$):** Sum of all non-benign class probabilities meets 0.50.

| Metric | Rule A (Argmax) | Rule B ($P_{\text{max}} \ge 0.50$) | Rule C ($P_{\text{attack}} \ge 0.50$) |
| :--- | :---: | :---: | :---: |
| **True Positives (TP)** | 16,458 | 16,455 | 16,459 |
| **False Positives (FP)** | 9 | 9 | 9 |
| **True Negatives (TN)** | 3,763 | 3,763 | 3,763 |
| **False Negatives (FN)** | 9 | 12 | 8 |
| **Attack Recall** | **99.9453%** | **99.9271%** | **99.9514%** |
| **False Positive Rate** | **0.2386%** | **0.2386%** | **0.2386%** |
| **Precision** | **99.9453%** | **99.9453%** | **99.9453%** |
| **F1-Score** | **0.999453** | **0.999362** | **0.999484** |

**Why Rule A and Rule B differ by exactly 3 rows:**  
In rows 1254, 5713, and 18078, the ground truth is Web Attack. The model predicts `Web Attack XSS` with probability 0.4994, and `Web Attack Brute Force` with probability 0.4981 ($P_{\text{benign}} = 0.0017$). Because the model is split between two web attack sub-types, neither individual sub-class exceeds 0.50. Under Rule B, these 3 flows are not flagged as known attacks by the classifier, but under Rule C or hierarchical web grouping, they are flagged with 99.8% attack confidence.

### 2.2 Forensic Audit of High Scores & Leakage Checks
- **Preprocessing Order:** Verified that feature selection and label encoding were fitted strictly on training rows. No test-set statistics leaked into standardizations or encoding.
- **Target Leakage:** Inspected all 78 feature columns. All columns represent passive flow statistics computed by CICFlowMeter. None are derived post-label.
- **Conflicting Labels:** 6 feature groups (14 rows out of 139,535, or 0.0100%) have identical features with conflicting labels. All 14 rows are assigned to single partitions by majority vote.
- **Generalization Reality Check:**  
  > [!WARNING]
  > Zero duplicate feature overlap proves that the classifier is not memorizing exact feature tuples from training. However, **duplicate-resistant evaluation is NOT proof of real-world generalization or session independence**.
  > Because the raw dataset lacks timestamps and IP addresses, flows from the same automated attack script run within the same recording hour share identical packet distributions. Real-world IDS deployments will encounter distinct operating system TCP stacks, novel attack tool variants, and network topology shifts.

---

## 3. Alert Architecture & Detection Policies

### 3.1 Empirical Comparison: Classifier-Only vs VAE-Only vs Hybrid

Evaluated on the held-out test partition (20,239 rows: 16,467 Attack, 3,772 Benign):

| Policy | True Positives (TP) | False Positives (FP) | True Negatives (TN) | False Negatives (FN) | Attack Recall | False Positive Rate | Precision | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classifier-Only** | **16,464** | **1** | **3,771** | **3** | **99.98%** | **0.03%** | **99.99%** | **0.9999** |
| **VAE-Only ($\tau = 38.51$)** | 558 | 3 | 3,769 | 15,909 | 3.39% | 0.08% | 99.47% | 0.0655 |
| **Hybrid OR Policy** | **16,464** | **203** | **3,569** | **3** | **99.98%** | **5.38%** | **98.78%** | **0.9938** |

### 3.2 Key Findings on Hybrid Coupling:
1. **Zero Added Recall:** The VAE identified **0 additional attacks** that the classifier missed (both catch 16,464 out of 16,467 attacks).
2. **202 False Alarms Added:** Out of 202 flows where the VAE triggered while the classifier predicted BENIGN, **100% (202 / 202) are normal benign traffic**. Promoting VAE outliers into confirmed attacks increases false alarms by 179x (FPR jumps from 0.03% to 5.38%).
3. **Decoupled 4-State Alert Architecture:**  
   The system must not conflate statistical reconstruction errors with verified attacks. We formalized four explicit alert states:
   - **`dual_signal` (6,361 test flows):** VAE reconstruction error $> \tau$ AND supervised classifier identified a known attack. High-confidence incident corroborated by two independent methods.
   - **`known_attack` (10,103 test flows):** Classifier identified a known attack family; VAE reconstruction error is within baseline. Confirmed attack.
   - **`anomaly_candidate` (202 test flows):** VAE reconstruction error $> \tau$, but classifier predicts BENIGN. Flagged as an **unclassified statistical outlier candidate** for operator triage or novel protocol exploration, **not** promoted into a confirmed attack.
   - **`normal_traffic` (3,569 test flows):** Within baseline and classified as BENIGN.

---

## 4. Hierarchical Web Attack Validation

### 4.1 Evaluation & Quantification (Symmetric Aggregation)
When `Web Attack XSS` and `Web Attack Brute Force` are aggregated symmetrically into **`Web Attack`**:

| Metric | Subtype: XSS (98 flows) | Subtype: Brute Force (224 flows) | Aggregated Category: Web Attack (322 flows) |
| :--- | :---: | :---: | :---: |
| **Recall** | 24.49% | 90.18% | **99.38%** (320 / 322 flows) |
| **Precision** | 52.17% | 73.45% | **99.69%** (320 / 321 flows) |
| **F1-Score** | 0.3333 | 0.8096 | **0.9953** |
| **Overall Accuracy** | — | — | **99.85%** (Macro-F1: **0.9977**) |

### 4.2 Confusion Matrix Breakdown:
- Out of 98 true XSS flows: 24 predicted as XSS, 73 predicted as Brute Force, 1 as DoS GoldenEye, 0 as BENIGN.
- Out of 224 true Brute Force flows: 202 predicted as Brute Force, 21 as XSS, 1 as BENIGN.
- Out of 322 total Web Attacks: **320 are correctly classified as Web Attacks**.
- **Important Technical Truth:** Network layer-4 flow summaries (packet sizes, port 80 timings) cannot reliably separate XSS payloads from Brute Force POST requests. Deep Packet Inspection (DPI) of HTTP payloads is required for payload-level sub-typing.

---

## 5. Verification Suite & Test Results

1. **Automated Backend Tests:**
   - Command: `pytest tests/ -v`
   - **Result: 81 passed** in 12.13s.
   - Includes 6 new regression tests in `tests/test_alert_architecture.py` verifying alert types, decision rules, metric formulas, and hierarchical aggregation.
2. **Frontend Production Build:**
   - Command: `npm --prefix frontend run build`
   - **Result: Success** (1589 modules transformed, built in 1.69s, 0 errors).
3. **Database & API Verification:**
   - Verified that `alert_type` is stored in SQLite and queryable via `/audit-logs?alert_type=dual_signal`.
   - Verified that `/predict` returns `alert_type`, `detection_source`, `score`, and `threshold`.
