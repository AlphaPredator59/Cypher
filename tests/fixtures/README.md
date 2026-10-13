# CYPHER Manual Crosscheck Fixture

This directory contains a small, manually testable verification dataset sampled directly from the duplicate-resistant held-out CICIDS2017 test split (`ml/splits_leakage_resistant/test.csv`).

It allows human operators, QA engineers, and automated harnesses to validate the running CYPHER IDS application (both backend API and frontend dashboard) without fabricating data or labels.

---

## 1. Fixture Overview

- **Dataset File:** [`manual_crosscheck.csv`](file:///c:/Users/msi/Projects/Cypher/tests/fixtures/manual_crosscheck.csv)
- **Manifest File:** [`manual_crosscheck_manifest.json`](file:///c:/Users/msi/Projects/Cypher/tests/fixtures/manual_crosscheck_manifest.json)
- **Total Flows:** 49 rows
- **Columns (80 total):**
  - **Columns 0–77 (78 features):** Canonical XGBoost input features in the exact order expected by the active classifier (`ml/features.py:XGB_FEATURE_NAMES`). `Fwd Header Length.1` is preserved in its canonical position.
  - **Column 78:** `expected_label` (ground-truth multiclass label from CICIDS2017).
  - **Column 79:** `expected_binary_label` (`BENIGN` or `ATTACK`).

### Class Distribution (All 12 Supported Classes Covered)

| Class Label | Expected Binary | Sample Count | Source Split Available | Shortage? |
| :--- | :--- | :---: | :---: | :---: |
| `BENIGN` | `BENIGN` | 5 | 3,772 | None |
| `Bot` | `ATTACK` | 4 | 296 | None |
| `DDoS` | `ATTACK` | 4 | 3,750 | None |
| `DoS GoldenEye` | `ATTACK` | 4 | 1,544 | None |
| `DoS Hulk` | `ATTACK` | 4 | 3,222 | None |
| `DoS Slowhttptest` | `ATTACK` | 4 | 788 | None |
| `DoS slowloris` | `ATTACK` | 4 | 837 | None |
| `FTP-Patator` | `ATTACK` | 4 | 1,002 | None |
| `PortScan` | `ATTACK` | 4 | 3,759 | None |
| `SSH-Patator` | `ATTACK` | 4 | 947 | None |
| `Web Attack Brute Force` | `ATTACK` | 4 | 224 | None |
| `Web Attack XSS` | `ATTACK` | 4 | 98 | None |
| **Total** | | **49** | **20,239** | **0 shortages** |

---

## 2. Testing via the Web Dashboard (UI)

1. **Start the CYPHER Backend and Frontend:**
   ```bash
   # Terminal 1: Backend
   python -m uvicorn main:app --reload --port 8000

   # Terminal 2: Frontend
   npm --prefix frontend run dev
   ```
2. Open the browser to `http://localhost:5173`.
3. In the sidebar or navigation bar, open **"Upload CSV Report"**.
4. Drag and drop or browse to select [`tests/fixtures/manual_crosscheck.csv`](file:///c:/Users/msi/Projects/Cypher/tests/fixtures/manual_crosscheck.csv).
5. Click **"Analyze CSV"**.
6. The dashboard will display:
   - Total Flows Processed: `49`.
   - Flow Audit Table showing each flow's:
     - **Verdict** (e.g., `Anomalous — DDoS`, `Normal Traffic`, `Anomalous — Attack Type Unclassified`).
     - **Alert Type Badge** (`dual_signal`, `known_attack`, `anomaly_candidate`, `normal_traffic`).
     - **Anomaly Confidence & Classifier Probability**.
7. Cross-check each row in the UI table against the row in [`manual_crosscheck.csv`](file:///c:/Users/msi/Projects/Cypher/tests/fixtures/manual_crosscheck.csv) using `expected_label` and `expected_binary_label`.

---

## 3. Testing via the API Endpoints

### A. Bulk Upload via `/analyze-csv` (Recommended)
You can post the fixture directly to `/analyze-csv`:
```bash
curl -F "file=@tests/fixtures/manual_crosscheck.csv" http://localhost:8000/analyze-csv
```
> **Note:** The backend automatically isolates the 78 canonical input features and drops `expected_label` and `expected_binary_label` so ground-truth labels never reach the model.

### B. Single-Flow Testing via `/predict` or `/analyze`
To test an individual row via `/predict`, extract the 78 numeric features into JSON. **Never send `expected_label` or `expected_binary_label` to the model endpoint.**

Example (PowerShell):
```powershell
$row = Get-Content tests/fixtures/manual_crosscheck.csv | Select-Object -Skip 1 -First 1
# Parse header and values into a JSON dictionary excluding expected_* columns, then send:
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType "application/json" -Body '{"row": {"Destination Port": 80, "Flow Duration": 1234, ...}}'
```

Example (Python test script):
```python
import pandas as pd
import requests

df = pd.read_csv("tests/fixtures/manual_crosscheck.csv")
features = [c for c in df.columns if not c.startswith("expected_")]

for idx, row in df.iterrows():
    payload = {"row": row[features].to_dict()}
    response = requests.post("http://localhost:8000/predict", json=payload).json()
    print(f"Row {idx:2d} | Expected: {row['expected_label']:22s} | Verdict: {response['verdict']}")
```

---

## 4. Ground-Truth Comparison & Baseline Expectations

The table below summarizes the expected behavior when running this fixture through the active pipeline (`compute.predict`):

| Row Index Range | Expected Label | Expected Binary | Expected Verdict / Alert Type | Baseline Agreement Notes |
| :--- | :--- | :---: | :--- | :--- |
| `0` | `BENIGN` | `BENIGN` | `Anomalous — Attack Type Unclassified` (`anomaly_candidate`) | **VAE Outlier:** VAE reconstruction score exceeds $\tau=38.51$, but classifier detects no known attack. Highlighted as anomaly candidate. |
| `1` – `4` | `BENIGN` | `BENIGN` | `Normal Traffic` (`normal_traffic`) | Clean traffic within baseline. |
| `5` – `8` | `Bot` | `ATTACK` | `Anomalous — Bot` (`known_attack` / `dual_signal`) | 100% exact multiclass recognition. |
| `9` – `12` | `DDoS` | `ATTACK` | `Anomalous — DDoS` (`known_attack` / `dual_signal`) | 100% exact multiclass recognition. |
| `13` – `16` | `DoS GoldenEye` | `ATTACK` | `Anomalous — DoS GoldenEye` (`dual_signal`) | 100% exact multiclass recognition. |
| `17` – `20` | `DoS Hulk` | `ATTACK` | `Anomalous — DoS Hulk` (`dual_signal`) | 100% exact multiclass recognition. |
| `21` – `24` | `DoS Slowhttptest`| `ATTACK` | `Anomalous — DoS Slowhttptest` (`known_attack`) | 100% exact multiclass recognition. |
| `25` – `28` | `DoS slowloris` | `ATTACK` | `Anomalous — DoS slowloris` (`known_attack`) | 100% exact multiclass recognition. |
| `29` – `32` | `FTP-Patator` | `ATTACK` | `Anomalous — FTP-Patator` (`known_attack`) | 100% exact multiclass recognition. |
| `33` – `36` | `PortScan` | `ATTACK` | `Anomalous — PortScan` (`known_attack`) | 100% exact multiclass recognition. |
| `37` – `40` | `SSH-Patator` | `ATTACK` | `Anomalous — SSH-Patator` (`known_attack`) | 100% exact multiclass recognition. |
| `41` – `44` | `Web Attack Brute Force` | `ATTACK` | `Anomalous — Web Attack Brute Force` (`known_attack`) | 100% exact multiclass recognition. |
| `45` – `46` | `Web Attack XSS` | `ATTACK` | `Anomalous — Web Attack Brute Force` (`known_attack`) | **Sibling Web Attack:** Classified under sibling web attack family due to payload similarity in CICIDS2017. Binary detection remains 100% attack. |
| `47` – `48` | `Web Attack XSS` | `ATTACK` | `Anomalous — Web Attack XSS` (`known_attack`) | Exact multiclass recognition. |

### Baseline Summary Metrics on Fixture
- **Exact Multiclass Agreement:** 46 / 49 (93.88%)
- **Binary Intrusion Detection Agreement:** 48 / 49 (97.96%)
- **Attack Recall on Fixture:** 44 / 44 (100.0%)

---

## 5. Provenance and Integrity

- Generated deterministically from `ml/splits_leakage_resistant/test.csv`.
- Full traceability for every sampled flow (including original row indices and source `.pcap_ISCX.csv` capture files) is recorded in [`manual_crosscheck_manifest.json`](file:///c:/Users/msi/Projects/Cypher/tests/fixtures/manual_crosscheck_manifest.json).
- SHA-256 Checksum: `d697a9441cc4eae37ce6d4596b65ee54808400150fc9125a73ebade507e14bad`.
