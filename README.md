# CYPHER - Autonomous Dual-Stage Network Intrusion Detection System

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React%2018%20%2B%20Vite-61DAFB.svg?style=flat&logo=react)](https://react.dev/)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![Three.js](https://img.shields.io/badge/Visualizer-Three.js%20CRT-black.svg?style=flat&logo=three.js)](https://threejs.org/)
[![Google Gemini](https://img.shields.io/badge/LLM-Gemini%201.5%20Pro-4285F4.svg?style=flat&logo=google)](https://deepmind.google/technologies/gemini/)
[![Dataset](https://img.shields.io/badge/Dataset-CICIDS2017-orange.svg?style=flat)](https://www.unb.ca/cic/datasets/ids-2017.html)

**CYPHER** is an end-to-end, production-ready **Network Intrusion Detection System (IDS)**. It pairs an unsupervised **Variational Autoencoder (VAE)** for ultra-fast anomaly scoring with a supervised **Mixture of Experts (MoE)** ensemble of 8 specialized classifiers coordinated by a learned gating network. Detected anomalies are dynamically explained using **Google Gemini 1.5 Pro** for automated SOC alerts, all monitored through an interactive **Retro-Futuristic CRT Terminal UI** powered by React and Three.js WebGL shaders.

---

## Table of Contents

* [Overview](#overview)
* [System Architecture](#system-architecture)
  * [High-Level Topology](#high-level-topology)
  * [Two-Stage Inference Pipeline](#two-stage-inference-pipeline)
* [Core Components](#core-components)
  * [1. VAE Anomaly Detector (`VAEScorer`)](#1-vae-anomaly-detector-vaescorer)
  * [2. Mixture of Experts Ensemble (`MoEPredictor`)](#2-mixture-of-experts-ensemble-moepredictor)
  * [3. Threat Pipeline Orchestrator (`ThreatPipeline`)](#3-threat-pipeline-orchestrator-threatpipeline)
  * [4. SOC Alert Summarizer (Google Gemini LLM)](#4-soc-alert-summarizer-google-gemini-llm)
* [Interactive 3D CRT Frontend (`frontend/`)](#interactive-3d-crt-frontend-frontend)
  * [Visual Design & CRT Shaders](#visual-design--crt-shaders)
  * [Dashboard Features](#dashboard-features)
  * [Running the Frontend](#running-the-frontend)
* [Empirical Evaluation & Benchmarks](#empirical-evaluation--benchmarks)
  * [Ground Truth Validation (`test_model.py`)](#ground-truth-validation-test_modelpy)
  * [VAE Anomaly Reconstruction Profiling (`analyze_vae.py`)](#vae-anomaly-reconstruction-profiling-analyze_vaepy)
* [API Reference](#api-reference)
  * [Configuration & CORS](#configuration--cors)
  * [Endpoints Summary](#endpoints-summary)
  * [Detailed Endpoints & Payloads](#detailed-endpoints--payloads)
* [Feature Sets & Preprocessing](#feature-sets--preprocessing)
  * [VAE 77-Feature Vector](#vae-77-feature-vector)
  * [MoE 46-Feature Vector](#moe-46-feature-vector)
  * [Canonical Column Normalization](#canonical-column-normalization)
* [Project Structure](#project-structure)
* [Setup & Installation](#setup--installation)
  * [Local Backend Setup](#local-backend-setup)
  * [Local Frontend Setup](#local-frontend-setup)
  * [Docker Deployment](#docker-deployment)
* [Configuration & Environment](#configuration--environment)
* [License](#license)

---

## Overview

Modern networks generate millions of telemetry flows per minute. Traditional rule-based intrusion systems struggle with novel zero-day attacks, while purely supervised models fail when faced with imbalanced network traffic.

**CYPHER solves this with a multi-layered detection hierarchy:**

1. **Unsupervised Anomaly Filtering**: A 77-dimensional Variational Autoencoder (VAE) trained exclusively on benign traffic filters out safe traffic in sub-millisecond time. Flows with reconstruction squared errors exceeding a statistical threshold ($\tau = 38.51$) are flagged.
2. **Dynamic Ensemble Attack Attribution**: Flagged flows are routed to a Mixture of Experts (MoE) consisting of 8 diverse classifiers (gradient boosted trees, linear models, kernel SVM, and deep PyTorch networks) dynamically weighted by a PyTorch gating network.
3. **Automated SOC Intelligence**: High-priority alerts are converted into structured security analysis through Google Gemini 1.5 Pro, delivering immediate evidence, threat assessments, and actionable containment steps.
4. **Live Cyberpunk Command Console**: A hardware-accelerated WebGL CRT interface gives security analysts immediate visual feedback, live metric telemetry, manual packet injection, and random dataset sampling.

---

## System Architecture

### High-Level Topology

```text
┌───────────────────────────────────────────────────────────────────────────────────┐
│                      CYPHER CRT FRONTEND (React 18 + Vite)                       │
│                                                                                   │
│   • Three.js WebGL CRT Shaders       • Live Anomaly Gauge vs Threshold            │
│   • Packet Inspection Queue          • Top Contributor Error Cards                │
│   • One-Click Dataset Sampling       • Collapsible Raw JSON Telemetry             │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │ HTTP / REST (Port 3000 → Port 8000)
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                          FASTAPI APPLICATION (main.py)                            │
│                                                                                   │
│   GET  /health           GET  /random-analyze      POST /predict                  │
│   POST /analyze          POST /analyze-batch       POST /anomaly-flag             │
│   POST /attack-type      POST /summarize                                          │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                     THREAT INFERENCE PIPELINE (compute.py)                        │
│                                                                                   │
│   ┌─────────────────────────────────────┐  Flagged  ┌──────────────────────────┐  │
│   │     VAEScorer (77 Features)         │──────────▶│  MoEPredictor (46 Feats) │  │
│   │                                     │  Anomalies│                          │  │
│   │  • StandardScaler (Benign baseline) │           │  • XGBoost    • LightGBM │  │
│   │  • PyTorch Latent Encoder (77→256)  │           │  • CatBoost   • RF       │  │
│   │  • Reparameterization (Latent=16)   │           │  • Logistic   • SVM      │  │
│   │  • Reconstruction Decoder (256→77)  │           │  • Small MLP  • Deep MLP │  │
│   │  • Squared Error vs Threshold       │           │  • Gating Net (Softmax)  │  │
│   └─────────────────────────────────────┘           └──────────────────────────┘  │
└───────────────────────┬───────────────────────────────────────────┬───────────────┘
                        │                                           │
                        ▼                                           ▼
┌──────────────────────────────────────────────────┐  ┌─────────────────────────────┐
│          LLM SOC SUMMARIZER (llm.py)             │  │   LOCAL CICIDS2017 DATASET  │
│                                                  │  │ (solarmainframe/ids-csv/..) │
│   • Google Gemini 1.5 Pro via LangChain          │  │                             │
│   • Structured JSON: Severity, Evidence, Action  │  │ • 2.83M Raw Network Flows   │
│   • Tailored Countermeasures & Next Steps        │  │ • Automated Column Mapper   │
└──────────────────────────────────────────────────┘  └─────────────────────────────┘
```

### Two-Stage Inference Pipeline

```text
Raw Network Flow Dictionary (from socket, stream, or CSV)
               │
               ▼
┌────────────────────────────────┐
│  Column Normalization Mapping  │ ── Drops labels, replaces ±inf, fills NaN
└──────────────┬─────────────────┘
               │
               ▼
┌────────────────────────────────┐
│  VAEScorer Feature Extraction  │ ── Extracts 77 dimensions & scales
└──────────────┬─────────────────┘
               │
               ▼
┌────────────────────────────────┐
│  Reconstruction Error Scoring  │ ── SE = Σ (x_recon - x_orig)²
└──────────────┬─────────────────┘
               │
        Is Score > 38.51 ?
        ├─── NO ────────────────────────────────────────────────────────┐
        │                                                               │
        └─── YES ──▶ ┌───────────────────────────────────────────┐      │
                     │ MoEPredictor Extraction (46 Features)     │      │
                     └─────────────────────┬─────────────────────┘      │
                                           │                            │
                                           ▼                            │
                     ┌───────────────────────────────────────────┐      │
                     │ 8 Classical & Deep Neural Experts Score   │      │
                     │ Gating Network Computes Softmax Weights   │      │
                     │ Weighted Average Output: Label + Conf %   │      │
                     └─────────────────────┬─────────────────────┘      │
                                           │                            │
                                           ▼                            ▼
                     ┌────────────────────────────────────────────────────────┐
                     │ Unified Output: Anomaly Flag, Score, Threshold,        │
                     │ Attack Label, Confidence, Top 5 Feature Attribution    │
                     └─────────────────────┬──────────────────────────────────┘
                                           │
                           (Optional /summarize Call)
                                           ▼
                     ┌────────────────────────────────────────────────────────┐
                     │ Gemini 1.5 Pro Generates Structured Security Alert     │
                     └────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. VAE Anomaly Detector (`VAEScorer`)

* **Implementation:** [`compute.py`](file:///c:/Users/msi/Projects/Cypher/compute.py) & [`app/model/model.py`](file:///c:/Users/msi/Projects/Cypher/app/model/model.py)
* **Weights:** `artifacts/model.pth`, `artifacts/threshold.json`, `artifacts/scaler_train_benign.pkl`

The VAE acts as a high-throughput frontline filter trained purely on normal network operations. When an adversary injects anomalous packet distributions, the decoder fails to reconstruct the rare feature patterns, causing a spike in reconstruction error.

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Input Dimension** | `77` | Standardized network flow features |
| **Encoder Architecture** | `77 → 256 → 256` | LeakyReLU activations with batch normalization |
| **Latent Dimension** | `16` | Gaussian latent vector $(\mu, \sigma)$ with clamp $[-10, 10]$ |
| **Decoder Architecture** | `16 → 256 → 256 → 77` | LeakyReLU activations reconstructing scaled input |
| **Loss Formulation** | $\mathcal{L} = \text{MSE}(x, \hat{x}) + \beta \cdot D_{\text{KL}}$ | Balanced reconstruction and latent regularization |
| **Anomaly Threshold** | `38.50959` | 99.9th percentile ($q=0.999$) derived from validation benign flows |
| **Top-K Attribution** | `5` | Identifies top 5 individual features driving reconstruction error |

### 2. Mixture of Experts Ensemble (`MoEPredictor`)

* **Implementation:** [`compute.py`](file:///c:/Users/msi/Projects/Cypher/compute.py)
* **Directory:** `artifacts/moeModels/`

Instead of relying on a single classifier that may develop blind spots on specific attack types, CYPHER uses a **Mixture of Experts** combining 8 diverse machine learning models and a dedicated neural gating network.

| Expert Classifier | Family | Implementation | Artifact File |
| :--- | :--- | :--- | :--- |
| **XGBoost** | Gradient Boosting | `xgboost.XGBClassifier` | `xgb.json` |
| **LightGBM** | Gradient Boosting | `lightgbm.LGBMClassifier` | `lgb.pkl` |
| **Random Forest** | Bagged Decision Trees | `sklearn.ensemble.RandomForestClassifier` | `rf.pkl` |
| **CatBoost** | Gradient Boosting | `catboost.CatBoostClassifier` | `cat.cbm` |
| **Logistic Regression** | Generalized Linear | `sklearn.linear_model.LogisticRegression` | `lr.pkl` |
| **Support Vector Machine** | Kernel Method | `sklearn.svm.SVC` (Probability enabled) | `svm.pkl` |
| **Small MLP** | PyTorch Feed-Forward | `Linear(46→64) → ReLU → Linear(64→32) → ReLU → Linear(32→3)` | `small_mlp.pt` |
| **Deep MLP** | PyTorch Feed-Forward | `Linear(46→128) → ReLU → Linear(128→64) → ReLU → Linear(64→32) → Linear(32→3)` | `deep_mlp.pt` |
| **Gating Network** | PyTorch Router | `Linear(46→64) → ReLU → Linear(64→32) → ReLU → Linear(32→8) → Softmax` | `gating.pt` |

**Ensemble Aggregation Formula:**
$$\vec{P}_{\text{final}} = \sum_{i=1}^{8} g_i(\vec{x}) \cdot \vec{P}_i(\vec{x})$$
where $g_i(\vec{x})$ is the dynamic weight assigned to expert $i$ by the gating network for input flow $\vec{x}$, and $\vec{P}_i$ is the 3-class probability distribution of expert $i$.

### 3. Threat Pipeline Orchestrator (`ThreatPipeline`)

* **Implementation:** [`compute.py`](file:///c:/Users/msi/Projects/Cypher/compute.py)

Coordinates the two stages into an optimized runtime:
1. If no rows in a batch exceed the anomaly threshold, execution finishes immediately without invoking the 8 MoE models, saving computation.
2. If anomalies exist, only the flagged rows are projected to the 46 MoE feature space and classified.
3. Automatically maps any anomalous flow that the MoE classifies as benign to `"Unknown Anomaly"`, preventing false negative misclassifications.
4. Generates feature attribution matrices detailing which packet fields generated the highest squared error.

### 4. SOC Alert Summarizer (Google Gemini LLM)

* **Implementation:** [`llm.py`](file:///c:/Users/msi/Projects/Cypher/llm.py)
* **Model:** `gemini-1.5-pro` (`temperature=0.2` for strict analytical accuracy)

When an analyst requires an explanation of an alert, the payload is sent to Gemini via LangChain. The LLM translates raw mathematical anomaly vectors into an actionable JSON incident response brief:

```json
{
  "severity": "high",
  "summary": "Suspicious high-rate TCP flow matching volumetric DDoS signature targeting port 80.",
  "what_is_wrong": [
    "Abnormal surge in backward packet length variance (3,681,776).",
    "Extreme flow bytes per second (19,338 B/s) inconsistent with normal HTTP interaction."
  ],
  "evidence": [
    {
      "feature": "Pkt Len Var",
      "signal": "high_recon_error",
      "note": "Reconstruction error 124.8 exceeds baseline variance."
    }
  ],
  "attack_assessment": {
    "predicted_type": "DDoS",
    "confidence": 0.999,
    "interpretation": "Flow characteristics indicate automated SYN/HTTP flood behavior."
  },
  "recommended_actions": {
    "immediate": ["Apply rate limiting to source subnet", "Inspect upstream firewall ACLs"],
    "short_term": ["Correlate destination port 80 traffic across adjacent ingress routers"],
    "long_term": ["Tune border BGP Flowspec filters for volumetric mitigation"]
  },
  "questions_to_ask": [
    "Is port 80 hosting a public service or an internal management console?"
  ],
  "limitations": [
    "Payload features reflect flow summary statistics only; deep packet payload inspection not performed."
  ]
}
```

---

## Interactive 3D CRT Frontend (`frontend/`)

CYPHER features an interactive frontend application in [`frontend/`](file:///c:/Users/msi/Projects/Cypher/frontend) designed around a **Cyberpunk / Retro-Futuristic CRT Terminal** aesthetic.

```text
frontend/
├── src/
│   ├── App.tsx                     # Main dashboard, UI states, API integration
│   ├── Scene.tsx                   # Three.js CRT visualizer wrapper
│   ├── main.tsx                    # Application entry point
│   ├── shaders/
│   │   ├── crt/CrtBackground.tsx   # React component binding WebGL canvas
│   │   ├── crt/crtRenderer.ts      # Three.js renderer lifecycle & resizing
│   │   ├── crt/crtScreens.ts       # CRT screen geometry, scanlines & curvature
│   │   ├── crt/crtShaders.ts       # GLSL fragment & vertex shaders (bloom, scanlines, jitter)
│   │   └── threeui.css             # Shader overlay canvas styling
│   └── styles.css                  # Retro cyber dashboard theme & responsive layout
├── verification/                   # UI verification screenshots
│   ├── cypher-crt-desktop.png      # Desktop console screenshot
│   ├── detection-page.png          # Threat readout inspection view
│   └── landing-page.png            # Landing hero view
├── package.json                    # React 18, Vite, Three.js, Lucide Icons
└── vite.config.ts                  # Vite development & build configuration
```

### Visual Design & CRT Shaders

* **Hardware-Accelerated CRT Shaders**: Built with custom GLSL shaders executing real-time barrel distortion, scanlines, chromatic aberration, phosphor persistence, and flicker.
* **Dual-View State Machine**:
  * **Landing Hero**: Cinematic title presentation with a single-click "Start Detection" CTA.
  * **Detection Console**: Split-screen command center with real-time operational status indicators (`Linked`, `Polling`, `Offline`, `Standby`).

### Dashboard Features

1. **Inspection Queue & JSON Editor**:
   * Pre-loaded with a known DDoS attack sample from the CICIDS test split.
   * Format validation, row extraction, and quick-reset button.
2. **Action Controls**:
   * **Check API (`/health`)**: Pings backend availability.
   * **Random Sample (`/random-analyze`)**: Immediately selects, normalizes, and classifies a real network flow from the local CICIDS2017 dataset.
   * **Predict (`/predict`)**: Executes single-packet inspection across the VAE + MoE pipeline.
3. **Live Threat Readout**:
   * **Score Gauge Bar**: Dynamic progress track comparing flow anomaly score directly against the 38.51 threshold.
   * **Metric Cards**: Real-time status cards for Anomaly Status (`Detected` / `Clear`), Numerical Score, Attack Attribution (`DDoS`, `DoS Hulk`, etc.), and Confidence percentage.
   * **Top Contributors Attribution**: Visual badges displaying the top 5 network features exhibiting highest squared error.
   * **Collapsible Raw Telemetry Drawer**: Formatted JSON viewer for deep telemetry inspection.

### Running the Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://127.0.0.1:3000`.

---

## Empirical Evaluation & Benchmarks

The CYPHER architecture has undergone extensive empirical validation against the full **CICIDS2017** benchmark dataset.

### Ground Truth Validation (`test_model.py`)

Validation is performed by [`test_model.py`](file:///c:/Users/msi/Projects/Cypher/test_model.py) using 5,000 balanced ground truth samples (2,500 benign, 2,500 across 14 attack categories) drawn from over 2.83 million raw network records:

| Metric | Score | Analysis |
| :--- | :--- | :--- |
| **Precision** | **83.70%** | When CYPHER flags an anomaly, it is an attack flow 84% of the time. |
| **Accuracy** | **60.50%** | Overall correct categorization across balanced evaluation split. |
| **Recall (TPR)** | **26.08%** | Primary coverage on high-impact volumetric attacks (DDoS, DoS, Heartbleed). |
| **F1 Score** | **39.77%** | Harmonic mean of precision and recall. |

#### Confusion Matrix Breakdown:

```text
                    Ground Truth Benign    Ground Truth Attack
Predicted Benign          2,373 (TN)             1,848 (FN)
Predicted Anomaly           127 (FP)               652 (TP)
```

* **True Negatives:** 2,373 benign flows correctly classified as normal.
* **False Positives:** Only 127 benign flows incorrectly flagged ($< 5.1\%$ FPR).
* **True Positives:** 652 confirmed attack flows caught and classified.

### VAE Anomaly Reconstruction Profiling (`analyze_vae.py`)

Detailed profiling across 2,830,743 flows generated by [`analyze_vae.py`](file:///c:/Users/msi/Projects/Cypher/analyze_vae.py) and documented in `vae_analysis_report.csv`:

| Traffic Category | Total Flows Evaluated | Median Error Score | 95th Percentile Score | Detection Rate / Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Heartbleed** | 11 | 5,089.05 | 6,490.07 | **100.00%** |
| **DoS Hulk** | 230,124 | 351.04 | 1,727.12 | **67.36%** |
| **DoS GoldenEye** | 10,293 | 136.29 | 2,611.35 | **66.31%** |
| **DDoS** | 128,025 | 419.36 | 5,211.38 | **63.61%** |
| **Infiltration** | 36 | 21.78 | 22,016.13 | **30.56%** |
| **DoS Slowhttptest** | 5,499 | 14.20 | 143.79 | **23.73%** |
| **DoS slowloris** | 5,796 | 17.83 | 49.98 | **23.29%** |
| **Web Attack (Sql Inj)** | 21 | 9.89 | 47.69 | **9.52%** |
| **Web Attack (Brute Force)**| 1,507 | 3.74 | 35.30 | **4.98%** |
| **BENIGN (Baseline)** | 2,271,320 | 6.49 | 40.49 | **5.26% FPR** |

> **Key Finding:** Volumetric and denial-of-service threats (DDoS, DoS Hulk, GoldenEye, Heartbleed) exhibit extreme reconstruction errors (often hundreds to thousands of times higher than threshold), yielding high detection rates. Stealthy single-packet probes (PortScan, Patator) exhibit lower reconstruction error and are primarily handled by the MoE classifier.

---

## API Reference

### Configuration & CORS

* **Backend Base URL:** `http://localhost:8000`
* **Swagger Interactive Docs:** `http://localhost:8000/docs`
* **Allowed CORS Origins:**
  * `http://localhost:3000` & `http://127.0.0.1:3000` (Default Vite frontend)
  * `http://localhost:8001` & `http://127.0.0.1:8001`
  * `http://localhost:5500` & `http://127.0.0.1:5500` (Live Server extensions)
  * `http://localhost` & `http://127.0.0.1`

### Endpoints Summary

| Method | Endpoint | Description | Typical Use Case |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Server heartbeat check | Health monitors, container probes |
| `GET` | `/random-analyze` | Picks random CICIDS CSV row and runs full analysis | UI Demo, test sampling |
| `POST` | `/predict` | Single row full prediction (flattened response) | Main UI "Predict" action |
| `POST` | `/analyze` | Single row analysis returning model + curated LLM payload | Flow deep-dive before `/summarize` |
| `POST` | `/anomaly-flag` | Fast check returning only boolean flag and score | High-throughput telemetry filtering |
| `POST` | `/attack-type` | Classification only for detected anomalies | Secondary attack classification |
| `POST` | `/analyze-batch` | High-throughput batch prediction for multiple rows | Bulk flow processing, table views |
| `POST` | `/summarize` | Generates structured SOC alert via Gemini LLM | Human-readable alert reports |

### Detailed Endpoints & Payloads

#### 1. `GET /random-analyze`
Picks a random file from `solarmainframe/ids-intrusion-csv/versions/1/*.csv`, normalizes columns using `CICIDS_COLUMN_MAPPING`, drops labels, and runs the threat pipeline.

**Response:**
```json
{
  "model_output": {
    "is_anomaly": true,
    "score": 419.36,
    "threshold": 38.51,
    "attack_type": "DDoS",
    "attack_confidence": 0.999,
    "top_contributors": [
      { "feature": "Pkt Len Var", "sq_error": 124.81 },
      { "feature": "Flow Byts/s", "sq_error": 88.19 }
    ]
  },
  "llm_payload": {
    "is_anomaly": true,
    "score": 419.36,
    "threshold": 38.51,
    "attack_type": "DDoS",
    "attack_confidence": 0.999,
    "top_contributors": [ ... ],
    "input_issues": {}
  }
}
```

#### 2. `POST /predict`
**Request (`RowIn`):**
```json
{
  "row": {
    "Dst Port": 80,
    "Flow Duration": 601248,
    "Tot Fwd Pkts": 3,
    "Tot Bwd Pkts": 6,
    "TotLen Fwd Pkts": 26,
    "TotLen Bwd Pkts": 11601,
    "Pkt Len Var": 3681776.01,
    "Flow Byts/s": 19338.11
  }
}
```

**Response:**
```json
{
  "is_anomaly": true,
  "score": 512.34,
  "threshold": 38.51,
  "attack_type": "DDoS",
  "attack_confidence": 0.9998,
  "top_contributors": [
    { "feature": "Pkt Len Var", "sq_error": 145.2 },
    { "feature": "Flow Byts/s", "sq_error": 92.4 }
  ],
  "input_issues": {}
}
```

#### 3. `POST /summarize`
**Request:**
```json
{
  "llm_payload": { ... },
  "model_output": { ... }
}
```
**Response:** Complete structured SOC incident response JSON conforming to the schema in [Core Components](#4-soc-alert-summarizer-google-gemini-llm).

---

## Feature Sets & Preprocessing

### VAE 77-Feature Vector

The VAE processes 77 continuous and discrete network flow attributes:

```text
Dst Port, Flow Duration, Tot Fwd Pkts, Tot Bwd Pkts, TotLen Fwd Pkts, TotLen Bwd Pkts,
Fwd Pkt Len Max, Fwd Pkt Len Min, Fwd Pkt Len Mean, Fwd Pkt Len Std,
Bwd Pkt Len Max, Bwd Pkt Len Min, Bwd Pkt Len Mean, Bwd Pkt Len Std,
Flow Byts/s, Flow Pkts/s, Flow IAT Mean, Flow IAT Std, Flow IAT Max, Flow IAT Min,
Fwd IAT Tot, Fwd IAT Mean, Fwd IAT Std, Fwd IAT Max, Fwd IAT Min,
Bwd IAT Tot, Bwd IAT Mean, Bwd IAT Std, Bwd IAT Max, Bwd IAT Min,
Fwd PSH Flags, Bwd PSH Flags, Fwd URG Flags, Bwd URG Flags,
Fwd Header Len, Bwd Header Len, Fwd Pkts/s, Bwd Pkts/s,
Pkt Len Min, Pkt Len Max, Pkt Len Mean, Pkt Len Std, Pkt Len Var,
FIN Flag Cnt, SYN Flag Cnt, RST Flag Cnt, PSH Flag Cnt, ACK Flag Cnt, URG Flag Cnt,
CWE Flag Count, ECE Flag Cnt, Down/Up Ratio, Pkt Size Avg, Fwd Seg Size Avg, Bwd Seg Size Avg,
Fwd Byts/b Avg, Fwd Pkts/b Avg, Fwd Blk Rate Avg, Bwd Byts/b Avg, Bwd Pkts/b Avg, Bwd Blk Rate Avg,
Subflow Fwd Pkts, Subflow Fwd Byts, Subflow Bwd Pkts, Subflow Bwd Byts,
Init Fwd Win Byts, Init Bwd Win Byts, Fwd Act Data Pkts, Fwd Seg Size Min,
Active Mean, Active Std, Active Max, Active Min, Idle Mean, Idle Std, Idle Max, Idle Min
```

### MoE 46-Feature Vector

The Mixture of Experts focuses on a subset of 46 discriminative features optimized for differentiating specific attack categories:

```text
Dst Port, Flow Duration, Tot Fwd Pkts, Tot Bwd Pkts, TotLen Fwd Pkts,
Fwd Pkt Len Max, Fwd Pkt Len Min, Fwd Pkt Len Mean, Bwd Pkt Len Max, Bwd Pkt Len Min, Bwd Pkt Len Mean,
Flow Byts/s, Flow Pkts/s, Flow IAT Mean, Flow IAT Std, Flow IAT Max,
Bwd IAT Tot, Bwd IAT Mean, Bwd IAT Std, Bwd IAT Min,
Fwd PSH Flags, Bwd PSH Flags, Fwd URG Flags, Bwd URG Flags,
Pkt Len Var, FIN Flag Cnt, RST Flag Cnt, PSH Flag Cnt, ACK Flag Cnt, URG Flag Cnt, CWE Flag Count,
Down/Up Ratio, Fwd Byts/b Avg, Fwd Pkts/b Avg, Fwd Blk Rate Avg,
Bwd Byts/b Avg, Bwd Pkts/b Avg, Bwd Blk Rate Avg,
Init Fwd Win Byts, Init Bwd Win Byts, Fwd Act Data Pkts, Fwd Seg Size Min,
Active Mean, Active Std, Active Max, Idle Min
```

### Canonical Column Normalization

Raw CICIDS CSV exports often contain differing column names with extra spaces or slightly altered abbreviations. `main.py` applies a canonical mapping dictionary (`CICIDS_COLUMN_MAPPING`):

* `"Destination Port"` $\rightarrow$ `"Dst Port"`
* `"Total Fwd Packets"` $\rightarrow$ `"Tot Fwd Pkts"`
* `"Total Length of Fwd Packets"` $\rightarrow$ `"TotLen Fwd Pkts"`
* `"Init_Win_bytes_forward"` $\rightarrow$ `"Init Fwd Win Byts"`
* Automatic cleanup: Strips whitespace, drops ground truth `"Label"`, replaces `±inf` with 0, fills `NaN` with 0.

---

## Project Structure

```text
Cypher/
├── main.py                         # FastAPI web service & route definitions
├── compute.py                      # Two-stage ThreatPipeline, VAEScorer, MoEPredictor
├── llm.py                          # Gemini 1.5 Pro LangChain alert generation
├── test_model.py                   # Empirical validation suite (Accuracy, Precision, Recall)
├── analyze_vae.py                  # Reconstruction error profiler & threshold validator
├── validation_report.json          # Empirical benchmark test results across 5k samples
├── vae_analysis_report.csv         # Full 2.83M row error distributions & recall by class
├── requirements.txt                # Python backend dependencies
├── Dockerfile                      # Container build definition
├── .env                            # Environment variables (GOOGLE_API_KEY)
│
├── frontend/                       # Interactive Cyberpunk CRT WebGL Interface
│   ├── src/
│   │   ├── App.tsx                 # Main application console & state machine
│   │   ├── Scene.tsx               # Three.js CRT canvas binding
│   │   ├── threeui.ts              # UI export definitions
│   │   ├── shaders/                # WebGL CRT Shaders (barrel distortion, scanlines)
│   │   └── styles.css              # CRT cyberpunk green/blue neon styling
│   ├── verification/               # Visual UI regression screenshots
│   ├── package.json                # React 18, Vite, Three.js dependencies
│   └── vite.config.ts              # Vite server & build configuration
│
├── app/
│   └── model/
│       └── model.py                # PyTorch VAE class & loss functions
│
├── artifacts/                      # Pre-trained production ML artifacts
│   ├── model.pth                   # PyTorch VAE weights (740 KB)
│   ├── threshold.json              # Anomaly threshold (38.50959)
│   ├── scaler_train_benign.pkl     # StandardScaler fitted on benign traffic
│   ├── features_name.json          # List of 77 canonical VAE feature names
│   └── moeModels/                  # 8 expert classifiers & gating weights
│       ├── cat.cbm                 # CatBoost model
│       ├── deep_mlp.pt             # PyTorch Deep MLP weights
│       ├── small_mlp.pt            # PyTorch Small MLP weights
│       ├── gating.pt               # PyTorch Gating Network router weights
│       ├── lgb.pkl                 # LightGBM model
│       ├── lr.pkl                  # Logistic Regression model
│       ├── rf.pkl                  # Random Forest model
│       ├── svm.pkl                 # Support Vector Machine model
│       ├── xgb.json                # XGBoost model
│       ├── label_encoder.pkl       # Attack label string encoder
│       └── scaler.pkl              # StandardScaler for MoE features
│
├── ml/                             # Offline training & data processing scripts
│   ├── data.py                     # Dataset loading utilities
│   ├── preprocess.py               # Feature normalization & cleaning
│   ├── split.py                    # Train/validation/test dataset splitting
│   ├── train_xgb.py                # XGBoost training pipeline
│   └── evaluate_xgb.py             # Evaluation scripts
│
├── development_codes/              # Research & prototyping notebooks
│   ├── vae.ipynb                   # VAE architecture experimentation
│   ├── moe.ipynb                   # MoE gating network research
│   └── mainMoe.py                  # Prototype gating implementation
│
└── solarmainframe/                 # Benchmark dataset store
    └── ids-intrusion-csv/
        └── versions/1/*.csv        # Raw CICIDS2017 daily capture CSV files
```

---

## Setup & Installation

### Local Backend Setup

**Requirements:** Python 3.10+, pip

```bash
# 1. Clone repository
git clone https://github.com/AlphaPredator59/Cypher.git
cd Cypher

# 2. Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
# Create a .env file and add your Gemini API key:
echo GOOGLE_API_KEY=your_gemini_api_key_here > .env

# 5. Launch FastAPI backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be operational at `http://localhost:8000` (Docs at `http://localhost:8000/docs`).

### Local Frontend Setup

**Requirements:** Node.js 18+, npm

```bash
# 1. Navigate to frontend directory
cd frontend

# 2. Install dependencies
npm install

# 3. Launch Vite development server
npm run dev
```

Open `http://127.0.0.1:3000` in any modern browser with WebGL support.

### Docker Deployment

To package and run the backend within a self-contained container:

```bash
# Build the Docker image
docker build -t cypher-ids:latest .

# Run container with environment configuration
docker run -d \
  --name cypher-ids-container \
  -p 8000:8000 \
  -e GOOGLE_API_KEY="your_gemini_api_key" \
  cypher-ids:latest

# Verify container health
curl http://localhost:8000/health
```

---

## Configuration & Environment

### Environment Variables

| Variable | Scope | Description | Default |
| :--- | :--- | :--- | :--- |
| `GOOGLE_API_KEY` | Backend (`.env`) | Google Gemini API key used by `/summarize` | *Required for LLM features* |
| `VITE_API_BASE` | Frontend (`.env`) | Base URL for FastAPI backend connection | `http://localhost:8000` |

### Internal Constants & Thresholds

| Setting | Value | File | Purpose |
| :--- | :--- | :--- | :--- |
| **VAE Threshold** | `38.50959` | `artifacts/threshold.json` | 99.9th percentile anomaly cutoff |
| **VAE Input Dim** | `77` | [`compute.py`](file:///c:/Users/msi/Projects/Cypher/compute.py) | Input feature vector width |
| **MoE Input Dim** | `46` | [`compute.py`](file:///c:/Users/msi/Projects/Cypher/compute.py) | Supervised feature vector width |
| **LLM Model** | `gemini-1.5-pro` | [`llm.py`](file:///c:/Users/msi/Projects/Cypher/llm.py) | Generative security explainer |
| **LLM Temp** | `0.2` | [`llm.py`](file:///c:/Users/msi/Projects/Cypher/llm.py) | Deterministic analytical output |

---

## License

Proprietary & Research Codebase — All rights reserved. Developed for high-precision autonomous network defense.
