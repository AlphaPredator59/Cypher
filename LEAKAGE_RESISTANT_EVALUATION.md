# CYPHER — Leakage-Resistant Evaluation & Generalization Report

**Milestone:** Leakage-Resistant Evaluation & Generalization Audit  
**Date:** October 13, 2026  
**Auditor:** Autonomous Systems Engineering & Independent ML Audit Team  
**Starting Checkpoint:** Commit `27ef1460` on `master`  
**Status:** **EMPIRICALLY VERIFIED — 0.00% CROSS-SPLIT DUPLICATION GUARANTEED**

---

## Executive Summary

Following the initial independent validation report ([FINAL_VALIDATION_REPORT.md](file:///c:/Users/msi/Projects/Cypher/FINAL_VALIDATION_REPORT.md)), this milestone investigated whether CYPHER's strong classification metrics generalize beyond repeated flow records, established a strict zero-duplicate group-aware benchmark, evaluated an alternative model architecture, quantified hierarchical web-attack aggregation, and verified system regressions.

### Key Conclusions:
1. **Provenance & Available Grouping Keys:**
   - The CICIDS2017 dataset store consists of 8 daily PCAP captures converted to statistical flow vectors.
   - **Crucial Limitation:** The dataset CSVs do **not** contain packet timestamps, IP addresses (`Source IP`, `Destination IP`), or flow session IDs.
   - **Cross-File Splitting Infeasibility:** 10 out of 11 attack classes appear in only **one** daily capture file (e.g., Tuesday contains all Patator attacks, Thursday morning contains all Web Attacks, Friday afternoon contains all DDoS). Partitioning across files would eliminate attack classes from either train or test, breaking multiclass evaluation entirely.
2. **Origin of Duplicate Flows:**
   - Automated attack scripts (e.g., Hydra for Patator, Hulk for DoS) generate bursts of identical TCP packets with fixed window sizes, identical lengths, and standard timeouts. In the raw dataset, `SSH-Patator` contains **45.4%** duplicate feature vectors, and `PortScan` contains **42.9%**.
   - Under standard random stratified splitting, identical feature tuples were scattered across train and test partitions (causing 11.79% duplicate overlap).
3. **Leakage-Resistant Split Design:**
   - Developed a **Stratified Group-Aware Content Fingerprint Split** ([ml/split_leakage_resistant.py](file:///c:/Users/msi/Projects/Cypher/ml/split_leakage_resistant.py)), assigning every unique 78-feature tuple to a unique group ID and partitioning groups globally into Train (97,965 rows), Validation (21,331 rows), and Test (20,239 rows).
   - **Guaranteed Result:** **Strictly 0 (0.00%) duplicate feature tuples cross any split boundary**.
   - The original baseline splits in `ml/splits/` were preserved 100% untouched.
4. **Generalization Results on Leak-Free Test Split (20,239 Flows):**
   - **XGBoost 12-Class Accuracy:** **99.39%** (Macro-F1: **0.9268**, Weighted-F1: **0.9933**).
   - **Binary Attack Recall:** **99.95%** with a False Positive Rate of only **0.24%** (9 benign flows flagged out of 3,772).
   - **Alternative Model (LightGBM):** **99.30%** accuracy (Macro-F1: **0.9200**, Latency: **14.30 ms / 1k flows**). XGBoost remains the superior performer in accuracy (99.39% vs 99.30%) and latency (9.70 ms vs 14.30 ms).
   - **Generalization Validated:** Even with 0% duplicate leakage, `SSH-Patator` achieved **100.0%** recall/precision, `FTP-Patator` achieved **99.90%**, and `PortScan` achieved **99.89%**. The model learned genuine structural statistical patterns rather than memorizing duplicate records.
5. **Hierarchical Web Attack Handling:**
   - Grouping `Web Attack XSS` and `Web Attack Brute Force` into `Web Attack` achieves **99.38% recall** and **99.69% precision** (F1: **0.9953**), raising overall hierarchical accuracy to **99.85%** (Macro-F1: **0.9977**).
   - The mutual confusion between XSS and Brute Force is purely intra-category ambiguity due to identical HTTP layer-4 flow statistics. Resolving sub-types requires HTTP payload inspection.
6. **Regression Suite:**
   - 72 / 72 tests passed (including 5 new automated leak-free split verification tests).
   - Frontend production build succeeded with 0 errors.

---

## 1. Audit of the Existing Split

### 1.1 Dataset Provenance & File Inventory
The dataset resides at `solarmainframe/ids-intrusion-csv/versions/1/` (CICIDS2017 generated via CICFlowMeter). Each file corresponds to a specific recording day/window:

| File Name | Size (MB) | Total Rows | Attack Classes Present |
| :--- | :---: | :---: | :--- |
| `Monday-WorkingHours.pcap_ISCX.csv` | 176.9 MB | 529,918 | `BENIGN` only (no attacks) |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | 135.1 MB | 445,909 | `FTP-Patator` (7,938), `SSH-Patator` (5,897), `BENIGN` |
| `Wednesday-workingHours.pcap_ISCX.csv` | 225.2 MB | 692,703 | `DoS Hulk` (231,073), `DoS GoldenEye` (10,293), `DoS slowloris` (5,796), `DoS Slowhttptest` (5,499), `Heartbleed` (11), `BENIGN` |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | 52.0 MB | 170,366 | `Web Attack Brute Force` (1,507), `Web Attack XSS` (652), `Web Attack SQL Injection` (21), `BENIGN` |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | 83.1 MB | 288,602 | `Infiltration` (36), `BENIGN` |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | 58.3 MB | 191,033 | `Bot` (1,966), `BENIGN` |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | 76.9 MB | 286,467 | `PortScan` (158,930), `BENIGN` |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | 77.1 MB | 225,745 | `DDoS` (128,027), `BENIGN` |

### 1.2 Available Grouping Keys & Limitations
- **Headers Inspected:** All CSVs contain exactly 79 columns: 78 statistical network flow features and 1 label column (`Label`).
- **Missing Attribution Data:** The dataset stripped all packet IP headers (`Source IP`, `Destination IP`), packet arrival `Timestamp`, and `Flow ID`.
- **Infeasibility of File-Level Group Splitting:** Because attacks were executed on dedicated scheduled days, each attack class appears in exactly **one** PCAP file. If `Tuesday` is placed in the training partition, the test partition contains 0 samples of SSH-Patator and FTP-Patator. A pure file-grouped split cannot evaluate multiclass attack detection.

### 1.3 How Duplicate Features Arise
Automated network attack tools fire standardized packet bursts:
- `SSH-Patator` / `FTP-Patator`: Password guessing tools open identical TCP sessions with identical handshake packets, fixed TCP window sizes (8192 / 229), zero initial data length, and constant timeouts.
- `PortScan`: Probing closed or filtered ports produces identical TCP SYN/RST sequences.
- `DoS Hulk`: Generates repetitive HTTP GET requests with randomized user-agents but identical packet length envelopes.

**Duplicate Rate in Raw Dataset:**
- `SSH-Patator`: 5,897 rows $\rightarrow$ 3,219 unique feature tuples (**45.4% duplicates**)
- `PortScan`: 158,930 rows $\rightarrow$ 90,819 unique feature tuples (**42.9% duplicates**)
- `FTP-Patator`: 7,938 rows $\rightarrow$ 5,933 unique feature tuples (**25.3% duplicates**)
- `DoS Hulk`: 231,073 rows $\rightarrow$ 172,849 unique feature tuples (**25.2% duplicates**)
- `BENIGN`: 2,273,097 rows $\rightarrow$ 2,096,484 unique feature tuples (**7.8% duplicates**)
- `Web Attack XSS`: 652 rows $\rightarrow$ 652 unique feature tuples (**0.0% duplicates**)

In `ml/split.py`, random row-level sampling scattered identical duplicate rows across Train (70%), Validation (15%), and Test (15%), creating the 2,467 cross-split duplicates identified in Task 1.

---

## 2. Design of the Leakage-Resistant Experiment

### 2.1 The Stratified Group-Aware Content Fingerprint Split
To construct the strongest defensible leak-free partition without sacrificing class coverage:
1. **Content Fingerprinting:** Every unique 78-feature tuple is assigned a unique `_feature_hash`. Across the balanced pool of 139,535 rows, there are exactly **124,964 unique feature groups**.
2. **Global Group Partitioning:** Groups are stratified across all 12 classes and allocated to `Train` (70%), `Validation` (15%), and `Test` (15%). All rows sharing the same feature hash are placed together into the same partition.
3. **Preservation of Baselines:** Output saved in [ml/splits_leakage_resistant/](file:///c:/Users/msi/Projects/Cypher/ml/splits_leakage_resistant). The original `ml/splits/` directory remains untouched.

### 2.2 Leakage Verification & Class Coverage
| Partition | Row Count | Group Count | Overlap with Train | Overlap with Val | Overlap with Test |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 97,965 | 87,477 | — | 0 (0.00%) | **0 (0.00%)** |
| **Validation** | 21,331 | 18,744 | 0 (0.00%) | — | **0 (0.00%)** |
| **Test** | 20,239 | 18,743 | **0 (0.00%)** | 0 (0.00%) | — |

**All 12 classes are well-represented in the held-out test split:**
- `BENIGN`: 3,772
- `PortScan`: 3,759
- `DDoS`: 3,750
- `DoS Hulk`: 3,222
- `DoS GoldenEye`: 1,544
- `FTP-Patator`: 1,002
- `SSH-Patator`: 947
- `DoS slowloris`: 837
- `DoS Slowhttptest`: 788
- `Bot`: 296
- `Web Attack Brute Force`: 224
- `Web Attack XSS`: 98

---

## 3. Fair Model Comparison

Both models were trained on `ml/splits_leakage_resistant/train.csv` (97,965 rows) with preprocessors fitted solely on training data, evaluated against `validation.csv` for tuning, and evaluated once on the final held-out `test.csv` (20,239 rows).

### 3.1 Primary Benchmark Results

| Model / Architecture | Test Accuracy | Macro-F1 | Weighted-F1 | Binary Attack Recall | Binary FPR | Inference Latency (ms/1k) | Cross-Split Duplicate Overlap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Leak-Free Split)** | **99.39%** | **0.9268** | **0.9933** | **99.95%** | **0.24%** | **9.70 ms** | **0.00% (0 rows)** |
| **LightGBM (Leak-Free Split)** | **99.30%** | **0.9200** | **0.9926** | **99.94%** | **0.32%** | **14.30 ms** | **0.00% (0 rows)** |
| *Original Baseline (Random Split)* | *99.37%* | *0.9309* | *0.9936* | *99.95%* | *4.93%* | *10.44 ms* | *11.79% (2,467 rows)* |

### 3.2 Per-Class Breakdown (XGBoost on Leak-Free Split)
| Class Name | Support | Precision | Recall | F1-Score | Random-Split F1 | Delta vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BENIGN** | 3,772 | 0.9976 | 0.9976 | 0.9976 | 0.9973 | +0.0003 |
| **Bot** | 296 | 0.9966 | 0.9932 | 0.9949 | 0.9912 | +0.0037 |
| **DDoS** | 3,750 | 0.9995 | 1.0000 | 0.9997 | 0.9998 | -0.0001 |
| **DoS GoldenEye** | 1,544 | 0.9961 | 0.9987 | 0.9974 | 0.9993 | -0.0019 |
| **DoS Hulk** | 3,222 | 0.9985 | 0.9997 | 0.9991 | 0.9992 | -0.0001 |
| **DoS Slowhttptest** | 788 | 0.9975 | 0.9937 | 0.9955 | 0.9923 | +0.0032 |
| **DoS slowloris** | 837 | 0.9976 | 0.9952 | 0.9964 | 0.9861 | +0.0103 |
| **FTP-Patator** | 1,002 | 0.9990 | 0.9990 | 0.9990 | 0.9994 | -0.0004 |
| **PortScan** | 3,759 | 0.9997 | 0.9989 | 0.9993 | 0.9980 | +0.0013 |
| **SSH-Patator** | 947 | **1.0000** | **1.0000** | **1.0000** | 0.9987 | +0.0013 |
| **Web Attack Brute Force** | 224 | 0.7345 | 0.9018 | 0.8096 | 0.7590 | +0.0506 |
| **Web Attack XSS** | 98 | 0.5217 | 0.2449 | 0.3333 | 0.4153 | -0.0820 |

**Critical Scientific Finding:**
Even when cross-split duplicate feature overlap is reduced to **strictly 0.00%**, XGBoost maintains **99.39% accuracy** and near-perfect classification across volumetric and brute-force attacks (`SSH-Patator` 100.0%, `FTP-Patator` 99.9%, `PortScan` 99.9%, `DoS Hulk` 99.9%). The model was **not** merely memorizing duplicate records; decision boundaries generalized to previously unseen feature tuples.

### 3.3 Hybrid Detector Evaluation on Leak-Free Split
Evaluating the VAE anomaly scorer + classifier pipeline on the 20,239 leak-free test flows:
- **True Positives (TP):** 16,464 attack flows detected
- **False Negatives (FN):** 3 attack flows missed
- **False Positives (FP):** 203 benign flows flagged
- **True Negatives (TN):** 3,569 benign flows correctly cleared
- **Attack Detection Recall:** **99.98%** $\left(\frac{16464}{16467}\right)$
- **False Positive Rate:** **5.38%** $\left(\frac{203}{3772}\right)$
- **Precision:** **98.78%**
- **Binary F1:** **0.9938**
- **Multiclass Threat Attribution:** **98.76%** overall (19,988 / 20,239) and **99.71%** on detected attacks.
- **Telemetry Breakdown:**
  - `classifier_attack`: 10,104 flows (stealth attacks rescued below VAE threshold)
  - `dual_confirmed`: 6,361 flows (simultaneous VAE anomaly + classifier confirmation)
  - `clear_normal`: 3,572 flows (cleared normal traffic)
  - `vae_anomaly`: 202 flows (high reconstruction error without classifier attack match)

---

## 4. Web Attack Hierarchical Reporting

### 4.1 Evaluation & Quantification
When `Web Attack XSS` and `Web Attack Brute Force` are aggregated into a unified **`Web Attack`** reporting category while retaining subtype predictions for operator drill-down:

| Metric | Fine-Grained 12-Class | Hierarchical Aggregated | Trade-off / Impact |
| :--- | :---: | :---: | :--- |
| **Overall Accuracy** | 99.39% | **99.85%** | **+0.46%** overall accuracy gain |
| **Macro-Averaged F1** | 0.9268 | **0.9977** | **+0.0709** macro-F1 gain |
| **Web Attack Recall** | XSS: 24.5%, BruteForce: 90.2% | **99.38%** (320 / 322 flows) | Catches virtually all web attacks |
| **Web Attack Precision** | XSS: 52.2%, BruteForce: 73.5% | **99.69%** | Extremely low false alarm rate |
| **Web Attack F1-Score** | XSS: 0.3333, BruteForce: 0.8096 | **0.9953** | Eliminates intra-category penalty |

### 4.2 Limitations & Payload-Level Requirements
- Hierarchical grouping **does not** solve payload-level disambiguation. It accurately reflects what network layer-4 statistical summaries can discern.
- Because both XSS and brute-force attacks target the same HTTP port 80 web server with similar packet sizes and TCP handshake patterns, flow statistics alone cannot distinguish between malicious `<script>` tags and credential dictionary loops.
- **Future Requirement:** True payload differentiation requires application-layer Deep Packet Inspection (DPI) extracting HTTP query strings, POST bodies, and URI character entropy.

---

## 5. Acceptance Deliverables & Regression Status

1. **Reproducible Scripts:**
   - [ml/split_leakage_resistant.py](file:///c:/Users/msi/Projects/Cypher/ml/split_leakage_resistant.py): Generates leak-free group-aware split with automated zero-duplicate assertions.
   - [ml/evaluate_leakage_resistant.py](file:///c:/Users/msi/Projects/Cypher/ml/evaluate_leakage_resistant.py): Trains XGBoost and LightGBM, benchmarks latency, computes hierarchical web attack metrics, and exports results.
2. **Automated Test Suite:**
   - [tests/test_leakage_resistant_evaluation.py](file:///c:/Users/msi/Projects/Cypher/tests/test_leakage_resistant_evaluation.py): 5 dedicated automated tests verifying leak-free split structure, zero cross-split duplicate overlap, metadata consistency, and hierarchical mapping.
   - **All 72 tests passed** (`pytest tests/ -v` in 11.93s).
3. **Frontend Production Build:**
   - `npm --prefix frontend run build` succeeded in 1.60s with 0 errors.
