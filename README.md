# CYPHER — Autonomous Dual-Stage Network Intrusion Detection System

[![Backend](https://img.shields.io/badge/Backend-FastAPI-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Frontend](https://img.shields.io/badge/Frontend-React%2018%20%2B%20Vite-61DAFB.svg?style=flat&logo=react)](https://react.dev/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![Visualizer](https://img.shields.io/badge/Visualizer-Three.js%20CRT-black.svg?style=flat&logo=three.js)](https://threejs.org/)
[![LLM](https://img.shields.io/badge/LLM-Gemini%201.5%20Pro-4285F4.svg?style=flat&logo=google)](https://deepmind.google/technologies/gemini/)
[![Tests](https://img.shields.io/badge/Tests-23%20Passed-brightgreen.svg?style=flat)](https://docs.pytest.org/)

**CYPHER** is a dual-stage **Network Intrusion Detection System (IDS)** designed for flow-based threat detection and attribution. It couples an unsupervised **Variational Autoencoder (VAE)** for anomaly detection with a supervised **Mixture of Experts (MoE)** ensemble of 8 specialized classifiers coordinated by a learned neural gating network.

Detected threats are presented in an interactive **Retro-Futuristic CRT Command Console** built with React, Vite, and Three.js WebGL shaders. In addition, an optional **Google Gemini 1.5 Pro** integration provides structured Security Operations Center (SOC) incident briefs for individual flows.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
  - [Two-Stage Detection & Attribution Pipeline](#two-stage-detection--attribution-pipeline)
  - [Decision Semantics & Four Operational States](#decision-semantics--four-operational-states)
- [Core Components](#core-components)
  - [1. VAE Anomaly Detector (`VAEScorer`)](#1-vae-anomaly-detector-vaescorer)
  - [2. Mixture of Experts Ensemble (`MoEPredictor`)](#2-mixture-of-experts-ensemble-moepredictor)
  - [3. Threat Pipeline Orchestrator (`ThreatPipeline`)](#3-threat-pipeline-orchestrator-threatpipeline)
  - [4. Calibrated Anomaly Confidence](#4-calibrated-anomaly-confidence)
  - [5. SOC Alert Summarizer (Google Gemini LLM)](#5-soc-alert-summarizer-google-gemini-llm)
- [Interactive Frontend Console (`frontend/`)](#interactive-frontend-console-frontend)
  - [Visual Design & CRT Shaders](#visual-design--crt-shaders)
  - [Inspection Queue Modes](#inspection-queue-modes)
  - [Structured Results Dashboard](#structured-results-dashboard)
  - [Telemetry Deep-Dive Drawer](#telemetry-deep-dive-drawer)
- [CSV Input Specifications & Strict Validation](#csv-input-specifications--strict-validation)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Setup & Execution](#setup--execution)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
  - [Docker Deployment](#docker-deployment)
- [Testing & Quality Assurance](#testing--quality-assurance)
- [Known Limitations](#known-limitations)

---

## Overview

Modern networks generate high volumes of telemetry records per minute. Traditional signature-based detection systems struggle with novel zero-day attacks, while purely supervised models fail when faced with unbalanced network traffic or unseen attack vectors.

**CYPHER addresses these challenges with a dual-stage architecture:**

1. **Unsupervised Anomaly Scoring (Stage 1)**: A 77-dimensional Variational Autoencoder (VAE) trained on normal traffic scores reconstruction error against an empirically derived threshold ($\tau = 38.50959$). Flows with reconstruction squared errors exceeding this threshold are flagged as anomalous.
2. **Supervised Mixture of Experts (Stage 2)**: An ensemble of 8 diverse classifiers (gradient-boosted trees, linear models, kernel SVM, and deep PyTorch networks) evaluates flows and weights expert predictions using a learned neural gating router.
3. **Transparent State Semantics**: Novel or unsupported attack categories (such as DDoS or PortScan) that are detected as anomalies by the VAE are clearly reported as `Anomalous — Attack Type Unclassified`, rather than being misclassified as benign or assigned a fabricated label.
4. **Analyst Command Console**: A hardware-accelerated WebGL CRT interface gives security analysts live metric telemetry, multi-flow CSV batch analysis, IP-based lookup, and deep feature-level attribution.

---

## System Architecture

### Two-Stage Detection & Attribution Pipeline

```text
Raw Network Flow (from CSV Report, Single JSON, or IP Query)
                       │
                       ▼
        ┌──────────────────────────────┐
        │  CSV / Payload Normalization │ ── Validates required 77 features
        └──────────────┬───────────────┘    Rejects NaN / inf / missing columns
                       │
                       ▼
        ┌──────────────────────────────┐
        │     VAE Anomaly Detector     │ ── 77 Features scaled via StandardScaler
        │    (Unsupervised Autoenc)    │ ── Computes Reconstruction Squared Error (SE)
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │    Threshold Decision        │
        │      (Score vs τ = 38.51)    │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │ Mixture of Experts Ensemble  │ ── Evaluates 46 discriminant features
        │     (8 Machine Learning      │ ── Gating Network computes softmax weights
        │     & Neural Experts)        │ ── Predicts class & uncalibrated probability
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────────────────────────────┐
        │                   Decision Reconciliation                    │
        │                                                              │
        │  • Score <= 38.51         ──▶ "Normal Traffic"               │
        │  • Score > 38.51 + FTP/SSH ──▶ "Anomalous — [Attack Name]"    │
        │  • Score > 38.51 + Benign  ──▶ "Anomalous — Attack Type       │
        │                                Unclassified"                 │
        │  • MoE Inference Failure  ──▶ "Classification Failed"        │
        └──────────────────────────────┬───────────────────────────────┘
                                       │
                       (Optional Analyst Action)
                                       ▼
        ┌──────────────────────────────────────────────────────────────┐
        │                 Google Gemini 1.5 Pro SOC Brief              │
        │          Translates telemetry into structured JSON brief     │
        └──────────────────────────────────────────────────────────────┘
```

### Decision Semantics & Four Operational States

CYPHER enforces strict, honest decision semantics in [`compute.py`](file:///c:/Users/msi/Projects/Cypher/compute.py):

| Condition | Verdict | Predicted Attack | Classification Status | Attack Confidence | Anomaly Confidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Normal Flow** ($S \le \tau$) | `Normal Traffic` | `Normal Traffic` | `Evaluated` | `--` (Not applicable) | Calibrated normalcy metric [0.50, 0.99] |
| **Recognized Attack** ($S > \tau$, MoE predicts supported attack) | `Anomalous — [Attack Name]` (e.g. `Anomalous — FTP-BruteForce`) | Actual attack name (`FTP-BruteForce` or `SSH-Bruteforce`) | `Evaluated` | Raw MoE ensemble probability | Calibrated anomaly metric [0.50, 0.999] |
| **Unclassified Anomaly** ($S > \tau$, MoE predicts Benign / novel class) | `Anomalous — Attack Type Unclassified` | `Attack Type Unclassified` | `Evaluated` | `--` (Unclassified) | Calibrated anomaly metric [0.50, 0.999] |
| **Classifier Failure** (MoE throws runtime exception) | `Classification Failed` | `Classification Failed` | `Failed` | `Failed` | Calibrated anomaly metric [0.50, 0.999] |

> **Security Rule:** A classification failure never converts an anomalous flow into benign traffic. Unclassified anomalies preserve the full VAE anomaly score and confidence while honestly stating that specific attack attribution is unavailable with current classifier artifacts.

---

## Core Components

### 1. VAE Anomaly Detector (`VAEScorer`)

- **Implementation:** [`compute.py`](file:///c:/Users/msi/Projects/Cypher/compute.py) & [`app/model/model.py`](file:///c:/Users/msi/Projects/Cypher/app/model/model.py)
- **Artifacts:** `artifacts/model.pth`, `artifacts/threshold.json`, `artifacts/scaler_train_benign.pkl`, `artifacts/features_name.json`

The VAE acts as an unsupervised filter trained exclusively on benign traffic. When anomalous traffic occurs, the decoder cannot reconstruct the abnormal feature patterns, causing a spike in reconstruction error:

$$\text{Score} = \sum_{j=1}^{77} (x_{\text{scaled}, j} - \hat{x}_{\text{recon}, j})^2$$

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Input Dimension** | `77` | Continuous network flow features |
| **Encoder Layers** | `77 → 256 → 256` | Linear layers with LeakyReLU activations and BatchNorm |
| **Latent Space** | `16` | Gaussian latent vector $(\mu, \sigma)$ with clamp $[-10, 10]$ |
| **Decoder Layers** | `16 → 256 → 256 → 77` | Linear layers with LeakyReLU activations reconstructing scaled input |
| **Anomaly Threshold ($\tau$)** | `38.50959` | Empirically derived cutoff configured in `artifacts/threshold.json` |
| **Top-K Attribution** | `5` | Identifies the top 5 individual features driving reconstruction error |

### 2. Mixture of Experts Ensemble (`MoEPredictor`)

- **Implementation:** [`compute.py`](file:///c:/Users/msi/Projects/Cypher/compute.py)
- **Artifacts Directory:** `artifacts/moeModels/`

To attribute detected anomalies to specific threat types, CYPHER uses an ensemble of 8 diverse models weighted by a learned neural gating router:

| Expert Classifier | Type | Implementation | Artifact File |
| :--- | :--- | :--- | :--- |
| **XGBoost** | Gradient-Boosted Trees | `xgboost.XGBClassifier` | `xgb.json` |
| **LightGBM** | Gradient-Boosted Trees | `lightgbm.LGBMClassifier` | `lgb.pkl` |
| **Random Forest** | Bagged Decision Trees | `sklearn.ensemble.RandomForestClassifier` | `rf.pkl` |
| **CatBoost** | Gradient-Boosted Trees | `catboost.CatBoostClassifier` | `cat.cbm` |
| **Logistic Regression** | Linear Model | `sklearn.linear_model.LogisticRegression` | `lr.pkl` |
| **Support Vector Machine** | Kernel Method | `sklearn.svm.SVC` (probability enabled) | `svm.pkl` |
| **Small MLP** | PyTorch Neural Net | `Linear(46→64) → ReLU → Linear(64→32) → ReLU → Linear(32→3)` | `small_mlp.pt` |
| **Deep MLP** | PyTorch Neural Net | `Linear(46→128) → ReLU → Linear(128→64) → ReLU → Linear(64→32) → Linear(32→3)` | `deep_mlp.pt` |
| **Gating Network** | PyTorch Router | `Linear(46→64) → ReLU → Linear(64→32) → ReLU → Linear(32→8) → Softmax` | `gating.pt` |

**Ensemble Probability Aggregation:**
$$\vec{P}_{\text{final}} = \sum_{i=1}^{8} g_i(\vec{x}) \cdot \vec{P}_i(\vec{x})$$
where $g_i(\vec{x})$ is the softmax routing weight from the gating network, and $\vec{P}_i$ is the probability distribution output by expert $i$.

### 3. Threat Pipeline Orchestrator (`ThreatPipeline`)

- **Implementation:** [`compute.py`](file:///c:/Users/msi/Projects/Cypher/compute.py)

The `ThreatPipeline` orchestrates the end-to-end execution:
1. Converts raw dictionaries into numpy feature arrays for VAE (77 features) and MoE (46 features).
2. Calculates reconstruction squared error and compares against $\tau = 38.50959$.
3. Runs the MoE classifier in a single batched operation across all flows to evaluate classification status for every record.
4. Implements the 4-state decision logic, cleanly segregating normal traffic, supported attack attributions, unclassified anomalies, and classification failures.
5. Computes top feature contributors for each flow based on individual squared errors.

### 4. Calibrated Anomaly Confidence

- **Implementation:** `calibrate_anomaly_confidence()` in [`compute.py`](file:///c:/Users/msi/Projects/Cypher/compute.py)

Rather than outputting raw threshold distances or binary flags, CYPHER calculates a mathematically bounded heuristic confidence metric in $[0.50, 0.999]$:

- **For Normal Traffic ($S \le \tau$):**
  $$c_{\text{benign}} = 0.50 + 0.49 \cdot \left(1.0 - \min\left(1.0, \frac{S}{\tau}\right)\right)^{0.85}$$
  Approaches $99.0\%$ as score approaches $0.0$; equals $50.0\%$ at the decision boundary $S = \tau$.
- **For Anomalous Traffic ($S > \tau$):**
  $$c_{\text{anom}} = 0.50 + 0.499 \cdot \left(1.0 - \exp\left(-4.5 \cdot \frac{S - \tau}{\tau}\right)\right)$$
  Equals $50.0\%$ at the decision boundary $S = \tau$, scaling exponentially up to $99.9\%$ for extreme reconstruction errors.

### 5. SOC Alert Summarizer (Google Gemini LLM)

- **Implementation:** [`llm.py`](file:///c:/Users/msi/Projects/Cypher/llm.py)
- **Model:** `gemini-1.5-pro` (`temperature=0.2`) via `langchain_google_genai`

Analysts can generate an incident brief for any selected flow. The LLM translates the flow's reconstruction error, top feature drivers, and model verdict into a structured JSON response containing:
- Severity level (`low`, `medium`, `high`, `critical`)
- Summary explanation
- Specific evidence points with feature attribution
- Recommended immediate, short-term, and long-term actions
- Investigative questions and known limitations

---

## Interactive Frontend Console (`frontend/`)

CYPHER includes a web application in [`frontend/`](file:///c:/Users/msi/Projects/Cypher/frontend) combining a Cyberpunk / Retro-Futuristic CRT aesthetic with practical network analysis tools:

### Visual Design & CRT Shaders
- **Hardware-Accelerated CRT Shaders**: Three.js WebGL canvas running custom GLSL shaders (scanlines, barrel curvature distortion, bloom, and phosphor persistence).
- **Dual-View Workflow**: Landing Hero view with launch CTA, transitioning into the Cybersecurity Operations Console.

### Inspection Queue Modes
1. **Mode A: Inspect IP**:
   - Validates IPv4 or IPv6 inputs.
   - Queries recent traffic analysis cache (`LAST_ANALYZED_TRAFFIC`) to display all flows originating from or targeting the queried IP address.
2. **Mode B: Upload CSV Report**:
   - Drag-and-drop or file selector supporting `.csv` files up to 50MB.
   - Sends file to `/analyze-csv` and renders batch summary metrics, attack distribution, and sortable traffic table.
3. **Mode C: Quick Test & Dataset Sample**:
   - **Quick Test**: Sends a pre-configured flow payload to `/predict`.
   - **Random Dataset Sample**: Calls `/random-analyze` to retrieve and score a random flow from the local benchmark dataset.

### Structured Results Dashboard
- **Telemetry Summary Cards**: Live counts for Total Flows, Normal Traffic, Anomalies Detected, and Backend Processing Time in milliseconds.
- **Attack Classification Chips**: Breakdown tags summarizing detected attack categories (e.g. `Anomalous — Attack Type Unclassified: 48`).
- **Interactive Controls**:
  - Live search filter by IP, port, protocol, verdict, or attack type.
  - Verdict filter (`All Verdicts`, `Anomalies Only`, `Normal Only`).
  - Column sorting on Row Index, Anomaly Score, and Flow Duration.
  - Full pagination controls.
- **Traffic Table Columns**:
  - `#` (Row index)
  - `Source IP` / `Dest IP`
  - `Port` / `Protocol`
  - `Score` (colored according to threshold)
  - `Status` (`ANOMALY` / `CLEAR` pill)
  - `Verdict` (`Normal Traffic`, `Anomalous — Attack Type Unclassified`, etc.)
  - `Predicted Attack` (dedicated attribution column)
  - `Verdict Conf` (calibrated percentage badge)
  - `Inspect` (opens flow deep-dive drawer)
- **CSV Export**: `Export CSV` button exports enriched records with complete metadata, anomaly scores, predicted attacks, and confidences via `exportFlowsToCsv`.

### Telemetry Deep-Dive Drawer
- Slide-out drawer displaying flow connection context, duration, and packet counts.
- Multi-model decision breakdown (Anomaly Status, Classification Status, Predicted Attack, Anomaly Confidence, Attack Class Confidence, Verdict Confidence).
- VAE reconstruction error contributor bars showing exact square error per feature.
- One-click Google Gemini SOC Alert generator (`POST /summarize`).

---

## CSV Input Specifications & Strict Validation

The `/analyze-csv` endpoint enforces strict validation in [`main.py`](file:///c:/Users/msi/Projects/Cypher/main.py):

### File Constraints
- **File Type:** Must have a `.csv` extension.
- **Encoding:** UTF-8 or Latin-1.
- **Size Limit:** Up to 50MB per upload.

### Metadata Retention
The following metadata headers are extracted and preserved for display, IP lookup, and CSV re-export (they are excluded from model feature arrays):
- **IP Addresses:** `Source IP`, `Destination IP` (or `Src IP`, `Dst IP`)
- **Port Numbers:** `Source Port` (`Dst Port` is retained as a VAE/MoE feature)
- **Protocol:** `Protocol` / `Proto`
- **Identifiers:** `Flow ID`, `Timestamp`, `_source_file`

### Feature Requirements & Canonical Mapping
The pipeline requires all **77 canonical flow features**. The backend automatically standardizes common aliases via `CICIDS_COLUMN_MAPPING`:
- `"Destination Port"` $\rightarrow$ `"Dst Port"`
- `"Total Fwd Packets"` $\rightarrow$ `"Tot Fwd Pkts"`
- `"Total Length of Fwd Packets"` $\rightarrow$ `"TotLen Fwd Pkts"`
- `"Init_Win_bytes_forward"` $\rightarrow$ `"Init Fwd Win Byts"`

### Strict Integrity Validation (No Silent Zero-Imputation)
- **Missing Columns:** If any of the 77 required features are missing from the CSV, the upload is rejected with **HTTP 422 Unprocessable Entity**, explicitly listing the missing column names.
- **Missing / NaN / Non-Numeric Values:** Rows containing `NaN`, `null`, empty fields, or non-parseable values are rejected with **HTTP 422**, identifying the affected columns. Silent zero-imputation is disabled to preserve model integrity.
- **Infinite Values:** Rows containing `inf` or `-inf` are rejected with **HTTP 422**.

---

## API Reference

The backend provides the following FastAPI endpoints:

| Method | Endpoint | Description | Request Format | Response Highlights |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/health` | Health check probe | None | `{"status": "ok"}` |
| `POST` | `/predict` | Full prediction for a single row | JSON `{ "row": { ... } }` | `is_anomaly`, `score`, `threshold`, `anomaly_confidence`, `attack_type`, `attack_confidence`, `classification_status`, `verdict`, `verdict_confidence`, `top_contributors` |
| `POST` | `/analyze-batch` | Batch prediction for multiple rows | JSON `{ "rows": [ { ... } ] }` | `results` list + `input_issues` dictionary |
| `POST` | `/attack-type` | Secondary attack classification only | JSON `{ "row": { ... } }` | `attack_type`, `attack_confidence`, `classification_status`, `verdict`, `verdict_confidence` |
| `POST` | `/anomaly-flag` | Fast check returning only anomaly status | JSON `{ "row": { ... } }` | `is_anomaly`, `score`, `threshold`, `anomaly_confidence` |
| `POST` | `/analyze` | Single row model output + curated LLM payload | JSON `{ "row": { ... } }` | `model_output` and `llm_payload` |
| `GET` | `/random-analyze` | Picks a random flow from local dataset | None | `model_output` and `llm_payload` |
| `POST` | `/inspect-ip` | Queries cached flows by IP address (POST) | JSON `{ "ip": "192.168.1.15" }` | `query_ip`, `ip_metadata_available`, `matching_flows_count`, `message`, `flows` |
| `GET` | `/inspect-ip` | Queries cached flows by IP address (GET) | Query param `?ip=192.168.1.15` | Same as POST variant |
| `POST` | `/analyze-csv` | Upload and analyze a traffic report CSV | Multipart form data `file: .csv` | `total_flows`, `normal_flows`, `anomalous_flows`, `attack_breakdown`, `processing_time_ms`, `results` |
| `GET` | `/sample-csv-template` | Download compliant sample CSV template | None | CSV file attachment with IP headers and feature columns |
| `POST` | `/summarize` | Generate structured SOC alert via Gemini LLM | JSON `{ "llm_payload": {...}, "model_output": {...} }` | Structured incident report conforming to Gemini SOC schema |

Interactive Swagger documentation is available at `http://localhost:8000/docs`.

---

## Project Structure

```text
Cypher/
├── main.py                         # FastAPI web application & endpoints
├── compute.py                      # VAEScorer, MoEPredictor, ThreatPipeline, confidence calibration
├── llm.py                          # Gemini 1.5 Pro LangChain SOC incident summarizer
├── requirements.txt                # Python backend dependencies
├── Dockerfile                      # Backend container configuration
│
├── frontend/                       # Interactive React + Three.js CRT Console
│   ├── src/
│   │   ├── App.tsx                 # Console state machine & layout
│   │   ├── Scene.tsx               # Three.js CRT WebGL wrapper
│   │   ├── types.ts                # TypeScript interfaces (TrafficFlowRecord, ModelResult, etc.)
│   │   ├── utils.ts                # Number formatting, IP validation, CSV export utility
│   │   ├── components/
│   │   │   ├── InspectionQueue.tsx # Mode tabs: Inspect IP, Upload CSV, Single Flow Test
│   │   │   ├── ResultsDashboard.tsx# Batch analysis dashboard, table, filters, pagination, export
│   │   │   ├── MetricReadout.tsx   # Score gauge meter, metric tiles, unclassified banners
│   │   │   └── FlowDetailDrawer.tsx# Feature deep-dive drawer & LLM summary trigger
│   │   ├── shaders/                # WebGL CRT shaders (curvature, scanlines, phosphor)
│   │   └── styles.css              # Cyberpunk terminal stylesheet
│   ├── package.json                # Frontend dependencies (React 18, Vite, Three.js, Lucide)
│   └── vite.config.ts              # Vite configuration (runs on port 3000)
│
├── artifacts/                      # Pre-trained ML model artifacts
│   ├── model.pth                   # PyTorch VAE weights (740 KB)
│   ├── threshold.json              # Anomaly threshold (38.50959)
│   ├── scaler_train_benign.pkl     # StandardScaler fitted on benign training flows
│   ├── features_name.json          # 77 canonical VAE feature names
│   └── moeModels/                  # Trained MoE classifier artifacts
│       ├── cat.cbm                 # CatBoost model
│       ├── deep_mlp.pt             # PyTorch Deep MLP weights
│       ├── small_mlp.pt            # PyTorch Small MLP weights
│       ├── gating.pt               # PyTorch Gating Network router weights
│       ├── lgb.pkl                 # LightGBM model
│       ├── lr.pkl                  # Logistic Regression model
│       ├── rf.pkl                  # Random Forest model
│       ├── svm.pkl                 # Support Vector Machine model
│       ├── xgb.json                # XGBoost model
│       ├── label_encoder.pkl       # LabelEncoder (['Benign', 'FTP-BruteForce', 'SSH-Bruteforce'])
│       └── scaler.pkl              # StandardScaler for 46 MoE features
│
├── tests/                          # Verified Pytest regression test suite
│   ├── test_api_endpoints.py       # API route integration tests
│   ├── test_confidence_and_pipeline.py # Confidence calibration & 4-state pipeline tests
│   ├── test_held_out_regression.py # Held-out test split & CSV upload regression tests
│   ├── cypher_verified_mixed_100.csv # Verified test dataset (50 normal, 50 anomalous)
│   └── test_100_flows.csv          # 100-flow test fixture with IP headers
│
└── ml/                             # Offline training & preprocessing utilities
    ├── preprocess.py               # Feature normalization
    └── splits/                     # Benchmark dataset splits
```

---

## Setup & Execution

### Backend Setup

**Prerequisites:** Python 3.10+ (tested on Python 3.12)

```bash
# 1. Clone the repository
git clone https://github.com/AlphaPredator59/Cypher.git
cd Cypher

# 2. Create and activate a virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Optional: Configure Gemini API Key for /summarize
# Create a .env file in the project root:
echo GOOGLE_API_KEY=your_gemini_api_key_here > .env

# 5. Start the FastAPI backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000` (Swagger docs at `http://localhost:8000/docs`).

### Frontend Setup

**Prerequisites:** Node.js 18+, npm

```bash
# 1. Navigate to the frontend directory
cd frontend

# 2. Install dependencies
npm install

# 3. Start the Vite development server
npm run dev
```

Open `http://127.0.0.1:3000` in any modern web browser with WebGL support.

To create a production build:
```bash
npm run build
```

### Docker Deployment

To build and run the backend inside a container:

```bash
# Build the Docker image
docker build -t cypher-ids:latest .

# Run the container
docker run -d \
  --name cypher-ids-container \
  -p 8000:8000 \
  -e GOOGLE_API_KEY="your_api_key_optional" \
  cypher-ids:latest

# Check health
curl http://localhost:8000/health
```

---

## Testing & Quality Assurance

The repository includes a comprehensive regression test suite under [`tests/`](file:///c:/Users/msi/Projects/Cypher/tests) covering API contracts, feature mapping, strict validation, confidence bounds, and pipeline decision semantics.

### Running Tests

Execute the test suite using `pytest`:

```bash
# Using the virtual environment's Python
.\.venv\Scripts\python.exe -m pytest tests -v
```

### Test Suite Coverage

- **`tests/test_api_endpoints.py`** (8 tests):
  - `/health` availability
  - `/predict` calibrated confidence bounds $[0.50, 0.999]$
  - IPv4 and IPv6 format validation on `/inspect-ip`
  - `/sample-csv-template` CSV download and header structure
  - `/analyze-csv` metadata retention and IP lookup caching
  - `/attack-type` response format
  - `/analyze-csv` error handling (invalid extensions, empty files, missing columns, non-numeric values)
  - 100-flow end-to-end processing and 1-to-1 IP correspondence
- **`tests/test_confidence_and_pipeline.py`** (9 tests):
  - Anomaly confidence calibration mathematical targets
  - Confidence boundary handling and edge cases
  - Normal flow semantics (`Normal Traffic`, no false alarms)
  - Anomaly with MoE disagreement (`Anomalous — Attack Type Unclassified`, preserved anomaly score)
  - Known attack classification (`Anomalous — FTP-BruteForce`)
  - MoE inference failure safety (preserves anomaly, sets `Classification Failed`)
  - Mixed batches containing both normal and anomalous flows
  - LabelEncoder inverse transformation and supported class verification
  - Invariant that anomalous flows never receive `Normal Traffic` or `BENIGN`
- **`tests/test_held_out_regression.py`** (6 tests):
  - Verification of real benign flows from held-out test split
  - Verification of real attack flows from held-out test split
  - Exact feature count, ordering, and scaler dimension alignment
  - Missing and duplicate column handling
  - Consistent scaling and threshold units
  - `cypher_verified_mixed_100.csv` upload and IP lookup verification

**Verified Test Result:** All 23 tests pass cleanly.

---

## Known Limitations

1. **Current Classifier Vocabulary**:
   - The pre-trained Mixture of Experts artifacts (`artifacts/moeModels/`) were trained on CSE-CIC-IDS2018 data covering three specific classes: `Benign`, `FTP-BruteForce`, and `SSH-Bruteforce`.
   - The classifier **does not possess trained weights** for distinct attribution of DDoS, DoS Hulk, GoldenEye, PortScan, Slowloris, or Botnet attacks.
2. **Detection vs. Attribution Distinction**:
   - High-volume volumetric threats (such as DDoS and DoS) exhibit substantial reconstruction errors ($S \gg 38.51$) and are reliably flagged as anomalies by the VAE.
   - However, because these attack families are outside the classifier's vocabulary, the system reports them as **`Anomalous — Attack Type Unclassified`**. This design prevents hallucinating incorrect labels while preserving the anomaly alert.
3. **Synthetic IP Metadata**:
   - The underlying statistical flow benchmarks (CICIDS2017 / CSE-CIC-IDS2018) do not include packet-level IP headers.
   - IP addresses present in sample templates and test fixtures are synthetically generated for demonstration of the IP inspection workflow.
4. **LLM Summarization Requirement**:
   - The `/summarize` endpoint requires a valid Google Gemini API key set in `GOOGLE_API_KEY`. Without this key, core detection and classification functions remain fully operational, but automated natural-language brief generation is disabled.
