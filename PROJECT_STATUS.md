# CYPHER Intrusion Detection System — Project Status

**Living Architecture & Codebase Status**  
**Current Iteration Completed:** Milestone 8 (Detection Quality Verification & Separate Anomaly Alerts)  
**Date:** October 13, 2026  
**Status:** Decoupled 4-State Alert Architecture, 0.00% Zero-Duplicate Benchmark, 81/81 Passing Tests

---

## 1. System Overview & Current Architecture

CYPHER is an AI-driven Network Intrusion Detection System (NIDS) designed to inspect network flow telemetry, detect anomalous network behaviors, categorize potential attacks across 12 supported classes, and visualize security events in real time.

```
                              ┌────────────────────────────────────────┐
                              │            React Frontend              │
                              │ (Vite + TypeScript CRT Security Console)│
                              └───────────────────┬────────────────────┘
                                                  │ HTTP (JSON / CSV)
                                                  ▼
                              ┌────────────────────────────────────────┐
                              │       FastAPI Gateway (main.py)        │
                              │ (Routes, Schema Ingestion, IP Cache)   │
                              └───────────────────┬────────────────────┘
                                                  │ Normalized Flow Vectors
                                                  ▼
                              ┌────────────────────────────────────────┐
                              │          ml.pipeline Engine            │
                              │  (compute.py provides backward compat) │
                              └───────────┬────────────────┬───────────┘
                                          │                │
            ┌─────────────────────────────┴────┐      ┌────┴─────────────────────────────┐
            ▼                                  ▼      ▼                                  ▼
 ┌──────────────────────┐           ┌──────────────────────┐           ┌──────────────────────┐
 │  Stage 1: VAE Engine │           │  Confidence Calibr.  │           │ Stage 2: Classifier  │
 │  (ml.models.vae)     │           │  (Bounded Sigmoid)   │           │  (ml.classifier)     │
 │  StandardScaler (77) │           │  Anchor: 38.50959    │           │  XGBoost JSON Booster│
 │  Deterministic: z=mu │           │  Output: [0.0, 1.0]  │           │  12 Attack Classes   │
 └──────────────────────┘           └──────────────────────┘           └──────────────────────┘
```

### Core Architecture Components:
1. **Frontend (`frontend/`):** A modern CRT terminal-inspired React console displaying network flow anomalies, flow metrics, live confidence readouts, inspection drawers, and IP flow investigations.
2. **Backend API (`main.py`):** FastAPI application orchestrating flow ingestion, single/batch analysis, CSV upload parsing, IP-based flow history caching, and optional Google Gemini AI incident summarization (`llm.py`).
3. **Consolidated ML Core (`ml/`):**
   - `ml.models.vae`: PyTorch Variational Autoencoder architecture (`Model`, `TotalLoss`). Features deterministic latent mean decoding (`z = mu`) during evaluation mode while preserving stochastic sampling (`z = reparameterize(mu, logvar)`) during training mode.
   - `ml.classifier`: Thin, testable 12-class supervised classifier interface (`XGBAttackClassifier`) wrapping `ml/models/xgb_baseline.json` and `ml/models/label_encoder.pkl` with per-class probability outputs.
   - `ml.pipeline`: Comprehensive inference engine containing `ThreatPipeline`, `VAEScorer`, `MoEPredictor` (legacy fallback), and `calibrate_anomaly_confidence`.
   - `ml.features`: Canonical feature mappings, explicit VAE (77) and XGBoost (78) schema definitions, duplicate column resolution (`Fwd Header Length.1`), and strict non-finite validation.
   - `ml.evaluate_vae`: Formal empirical evaluation script performing validation threshold sweeps and held-out test evaluations.
   - `ml.experiments`: Archived experimental research and exploratory notebooks (`mainMoe.py`, `moe.ipynb`, `vae.ipynb`).
   - `ml.reports`: Empirical validation reports, analysis summaries, and threshold sweep data (`vae_deterministic_eval.json`, `vae_threshold_sweep.json`).
4. **Compatibility Layer (`compute.py`, `app/model/model.py`):** Lightweight backward-compatible adapters re-exporting canonical symbols from `ml/` to prevent breaking existing API handlers, scripts, or legacy imports.

---

## 2. Actual Implemented Functionality

The following features and endpoints are fully implemented and verified in the current codebase:

- **Flow Analysis Endpoints:**
  - `POST /analyze-csv`: Upload and process raw CICIDS2017 network flow CSVs with strict 77-feature validation, header normalization, and summary generation.
  - `POST /predict`, `POST /analyze`: Single-flow anomaly scoring returning anomaly verdicts, reconstruction loss, calibrated confidence, top-5 contributing features, and MoE classification.
  - `POST /analyze-batch`: Multi-flow batch scoring returning list of verdicts.
  - `POST /anomaly-flag`: Quick boolean anomaly triage (`is_anomalous: true/false`).
  - `POST /attack-type`: MoE classification triage.
- **Investigation & Operational Endpoints:**
  - `GET /ip-flows`: Query cached historical network flows filtered by IP address and status (benign vs. anomalous).
  - `GET /sample-csv-template`: Download a 5-row valid sample CSV fixture to verify ingestion structure.
  - `POST /summarize`: Natural language incident report generation using Google Gemini (`ChatGoogleGenerativeAI`).
  - `GET /health`: Basic service health check probe.
- **Inference Telemetry:**
  - Deterministic reconstruction error SSE (`(recon - x)^2.sum(dim=1)`) with zero inter-run variance on identical inputs.
  - Bounded logistic confidence calibration (`calibrate_anomaly_confidence`).
  - Top-5 feature contribution extraction via `torch.topk`.
  - Mixture of Experts prediction combining 8 expert models (RandomForest, ExtraTrees, DecisionTree, XGBoost, CatBoost, LogisticRegression, SVC, NeuralNetwork) weighted by a Softmax gating network.

---

## 3. Repository Organization

```
Cypher/
├── .env.example                 # Documented environment variable template
├── .gitattributes               # Line-ending normalization (LF) and binary definitions
├── .gitignore                    # Whitelisted test fixtures, excluded node_modules & dist
├── CODEBASE_AUDIT.md            # Complete forensic audit report & historical issue log
├── PROJECT_STATUS.md            # Living system status and architecture reference (this file)
├── README.md                    # System documentation and developer onboarding
├── requirements.txt             # Backend Python dependencies
├── main.py                      # FastAPI routes, API schemas, application orchestration
├── compute.py                   # Backward-compatible adapter re-exporting ml.pipeline
├── llm.py                       # Google Gemini LLM summarization client
├── analyze_vae.py               # Forwarding runner -> ml.analyze_vae
├── test_model.py                # Forwarding runner -> ml.test_model
├── app/
│   └── model/
│       └── model.py             # Backward-compatible adapter re-exporting ml.models.vae
├── artifacts/                   # Pre-trained model weights, scalers, and thresholds
│   ├── model.pth                # Pretrained VAE PyTorch weights
│   ├── threshold.json           # Production VAE anomaly threshold (38.50959)
│   ├── scaler_train_benign.pkl  # VAE StandardScaler
│   ├── features_name.json       # Canonical 77 VAE feature names
│   └── moeModels/               # MoE expert models, gating network, and label encoder
├── frontend/                    # React + Vite CRT console
│   ├── package.json             # Frontend dependencies
│   ├── tsconfig.json            # TypeScript configuration
│   ├── vite.config.ts           # Vite build pipeline
│   ├── index.html               # CRT console entrypoint
│   ├── styles.css               # Retro CRT terminal styling
│   └── src/                     # React application source (App.tsx, components, types, utils)
├── ml/                          # Consolidated ML Root Directory
│   ├── __init__.py              # ML package exports
│   ├── features.py              # Canonical feature mappings, schemas & validation
│   ├── classifier.py            # 12-class XGBoost classifier interface (XGBAttackClassifier)
│   ├── pipeline.py              # Canonical ML scoring & inference engine
│   ├── evaluate_vae.py          # Evidence-based VAE evaluation & threshold sweeping script
│   ├── split.py                 # Dataset splitting script for XGBoost experiment
│   ├── train_xgb.py             # 12-class XGBoost classifier training script
│   ├── evaluate_xgb.py          # 12-class XGBoost model evaluation script
│   ├── analyze_vae.py           # VAE threshold and per-attack analysis script
│   ├── test_model.py            # Model validation and threshold calibration script
│   ├── models/                  # Core ML model architectures & standalone weights
│   │   ├── __init__.py          # Model architecture exports
│   │   ├── vae.py               # Deterministic PyTorch VAE model (Model, TotalLoss)
│   │   ├── xgb_baseline.json    # 12-class XGBoost model artifact (99.37% test accuracy)
│   │   └── label_encoder.pkl    # 12-class label encoder
│   ├── experiments/             # Exploratory research notebooks and scripts
│   │   ├── mainMoe.py           # Exploratory MoE training code
│   │   ├── moe.ipynb            # Exploratory MoE Jupyter notebook
│   │   └── vae.ipynb            # Exploratory VAE Jupyter notebook
│   ├── reports/                 # Consolidated validation and analysis reports
│   │   ├── vae_deterministic_eval.json  # Comprehensive held-out evaluation report
│   │   ├── vae_threshold_sweep.json     # Validation split threshold sweep data
│   │   ├── threshold_sweep_validation.json
│   │   ├── vae_analysis_report.json
│   │   ├── vae_analysis_report.csv
│   │   └── validation_report.json
│   └── splits/                  # Balanced CICIDS2017 dataset splits (train, val, test)
└── tests/                       # Automated test suite (46 passing tests)
    ├── test_api_endpoints.py            # FastAPI endpoint validation tests (8 tests)
    ├── test_classifier.py               # 12-class classifier unit tests (7 tests)
    ├── test_confidence_and_pipeline.py # Confidence calculation & pipeline tests (13 tests)
    ├── test_held_out_regression.py     # Regression tests on real CSV flows (6 tests)
    ├── test_hybrid_detection.py        # Hybrid detection gating & verdict fusion tests (7 tests)
    ├── test_vae_determinism.py         # Determinism regression test suite (5 tests)
    ├── test_100_flows.csv               # Test dataset fixture
    └── cypher_verified_mixed_100.csv   # Verified mixed-class test fixture
```

---

## 4. Known Technical Limitations & Open Issues

1. **VAE Stochastic Inference (ISS-01, P0) — RESOLVED in Iteration 3:**
   - Modified `ml/models/vae.py:Model.forward` to route latent vectors through `z = mu` when in evaluation mode (`not self.training` or `deterministic=True`), while preserving stochastic sampling (`z = reparameterize(mu, logvar)`) during training. Verified with 5 dedicated regression tests in `tests/test_vae_determinism.py` demonstrating zero variance (`std dev = 0.000000`).

2. **Severe Attack Recall Deficit on Stealth Attacks (ISS-02, P0) — Empirically Verified Limitation:**
   - Empirically proven on held-out test split (20,931 flows) that deterministic VAE scoring alone **does not solve the stealth attack blind spot**.
   - While volumetric attacks are detected (DoS Hulk: 67.5%, DDoS: 65.4%, DoS GoldenEye: 64.9%), stealthy and low-packet attacks remain undetectable by reconstruction error alone: PortScan recall is 0.08%, FTP-Patator is 0.00%, SSH-Patator is 0.00%, Bot is 5.12%, Web Attacks are 3.0%–4.4%.
   - Overall attack recall across all attack classes is **37.54%** at the production threshold (`38.50959`), with a **4.72%** false-positive rate on benign traffic.
   - *Scheduled Fix:* Iterations 4 & 5 (hybrid detection gating combining the 12-class supervised classifier with VAE reconstruction).

3. **MoE Class Truncation & Label Mismatch (ISS-03, P0) — RESOLVED in Iteration 4:**
   - Evaluated, verified, and integrated the 12-class XGBoost model (`ml/classifier.py:XGBAttackClassifier`) as the active classifier in `ThreatPipeline`.
   - The active engine now natively predicts and attributes all 12 classes (`BENIGN`, `Bot`, `DDoS`, `DoS GoldenEye`, `DoS Hulk`, `DoS Slowhttptest`, `DoS slowloris`, `FTP-Patator`, `PortScan`, `SSH-Patator`, `Web Attack Brute Force`, `Web Attack XSS`), resolving the "Attack Type Unclassified" bottleneck for DDoS, DoS, PortScan, Bot, and Web attacks.
   - The legacy 3-class MoE is preserved as a backwards-compatible fallback (`active_classifier="moe"`).

4. **Inconsistent Validation & Silent Zero Imputation (ISS-04, P0) — RESOLVED in Iteration 4:**
   - Formalized explicit feature schemas in `ml/features.py` (`VAE_FEATURE_NAMES` with 77 features, `XGB_FEATURE_NAMES` with 78 features).
   - Resolved the 77-vs-78 discrepancy: raw CICIDS2017 CSVs have duplicate column `Fwd Header Length` (at index 35 and 56), which pandas disambiguates as `Fwd Header Length.1`. `normalize_flow_for_xgb()` automatically derives `Fwd Header Length.1` from `Fwd Header Length` if absent.
   - Implemented strict validation functions `normalize_flow_for_xgb` and `normalize_flow_for_vae` that raise explicit `ValueError` on missing keys, extra keys, or non-finite values (`inf`, `-inf`, `nan`) rather than silently imputing zero values.

5. **Version Warnings on Unpickled Artifacts (ISS-06, P1) — Diagnosed:**
   - Diagnosed 849 warnings per test session:
     * 826 warnings: `joblib/numpy_pickle.py:207: DeprecationWarning` from setting `array.shape` directly in NumPy 2.5 during unpickling of tree arrays in `artifacts/moeModels/`.
     * 7 warnings: `sklearn/base.py:525: InconsistentVersionWarning` from loading scikit-learn 1.4.2/1.8.0 estimators into scikit-learn 1.9.0 (`StandardScaler`, `LabelEncoder`, `DecisionTreeClassifier`, `RandomForestClassifier`, `LogisticRegression`, `SVC`).
     * 1 warning: `fastapi/testclient.py:1: StarletteDeprecationWarning` regarding `httpx` vs `httpx2`.
     * 4 warnings: `sklearn/utils/validation.py:2827: UserWarning` from passing raw ndarrays to `StandardScaler`.
   - Per engineering standards, warnings are diagnosed without suppression. Re-serialization of legacy 3-class MoE artifacts is deferred to Iteration 4 when the 12-class classifier replaces them.

6. **In-Memory Telemetry State (ISS-10, P1) — RESOLVED in Iteration 6:**
   - Implemented `database.py` with SQLAlchemy 2.0 ORM (`FlowAuditRecord`) and indexed SQLite store (`artifacts/cypher.db`).
   - All flows analyzed via `/analyze-csv`, `/predict`, `/analyze`, and `/analyze-batch` are automatically persisted with connection metadata, VAE score, thresholds, detection sources, and verdict confidences.
   - Cross-session IP flow inspection (`/inspect-ip`) retrieves persisted records across server restarts and multiple CSV uploads without data loss.
   - Fault-isolated persistence: database write issues log warnings and never crash user inference requests.

---

## 5. Historical Verification Results (Iteration 3)

| Check Category | Command / Procedure | Result | Details |
|---|---|---|---|
| **Backend Test Suite** | `pytest` | **PASSED (28/28, 100%)** | Ran in 9.16s. 8 endpoint tests, 9 pipeline tests, 6 regression tests, 5 new VAE determinism tests. |
| **VAE Determinism Verification** | `tests/test_vae_determinism.py` | **PASSED (5/5, 100%)** | Verified zero score variance (`std dev = 0.000000`) across repeated passes on identical inputs, verified stochastic sampling in train mode, verified tensor shapes `(N, 77) -> (N, 77)`. |
| **Inter-Run Reproducibility** | `ml/evaluate_vae.py` on 20,930 rows | **VERIFIED** | Stochastic inter-run score mean absolute difference = `62.4461` (max `37,831.50`). Deterministic inter-run max difference = `0.000000`. |
| **Held-Out Test Set Evaluation** | `ml/evaluate_vae.py` on 20,931 rows | **EVALUATED** | Evaluated held-out test split with strict validation/test separation. Baseline vs Deterministic metrics recorded. |
| **Frontend Production Build** | `npm --prefix frontend run build` | **PASSED** | `tsc -b && vite build` completed in 4.31s without errors. |

### Comparative Performance on Held-Out Test Set (20,931 Flows):

| Configuration | Threshold | Precision | Recall | F1 Score | Benign FPR | Overall Accuracy |
|---|---|---|---|---|---|---|
| **Baseline Stochastic (Prod Threshold)** | `38.5096` | 0.9710 | 0.3736 | 0.5395 | 0.0512 | 0.4766 |
| **Deterministic (Prod Threshold)** | `38.5096` | **0.9733** | **0.3754** | **0.5418** | **0.0472** | **0.4789** |
| **Deterministic (Recalibrated q=0.999)** | `2957.6853` | 0.9947 | 0.0329 | 0.0637 | 0.0008 | 0.2060 |
| **Deterministic (Recalibrated q=0.990)** | `310.5724` | 0.9936 | 0.2790 | 0.4356 | 0.0083 | 0.4067 |
| **Deterministic (Recalibrated q=0.950)** | `38.4390` | 0.9730 | 0.3754 | 0.5418 | 0.0477 | 0.4788 |

*Per-Class Detection Recall on Test Set:*
- **Volumetric Attacks:** DoS Hulk (67.47%), DDoS (65.39%), DoS GoldenEye (64.90%), DoS slowloris (30.23%), DoS Slowhttptest (20.85%).
- **Stealth / Non-Volumetric Attacks:** Bot (5.12%), Web Attack Brute Force (4.42%), Web Attack XSS (3.06%), PortScan (0.08%), FTP-Patator (0.00%), SSH-Patator (0.00%).

---


### 4.5 Confidence Semantics & Display Correction (Iteration 4.5)
- **Problem Diagnosed:** Investigation into user observation that almost all anomaly verdicts displayed `99.9%` confidence while normal traffic varied.
  1. `ml/pipeline.py` previously blended saturating VAE exponential confidence (`anom_conf`) with XGBoost probability: `0.5 * anom_conf + 0.5 * atk_c`. Because VAE anomaly scores above $s > 98$ mathematically saturated at `0.9990` and XGBoost outputs $\ge 0.999$, the blend yielded `0.9995`.
  2. `frontend/src/utils.ts` clamped any value $\ge 99.9\%$ to `"99.9%"`.
  3. In contrast, normal traffic derived confidence from $(1 - s/\tau)^{0.85}$, which varied across 50%–98%.
- **Corrective Actions Taken:**
  1. **Eliminated Arbitrary Blending:** In `ml/pipeline.py`, replaced the 50/50 blend for attributed attacks with the classifier's genuine model confidence (`verdict_conf = atk_c`).
  2. **Truthful UI Labels:** Renamed misleading UI labels:
     - `Attack Class Conf (Uncalibrated MoE)` -> `Classifier Confidence (Model Output)`.
     - `Anomaly Confidence (Heuristic)` -> `VAE Anomaly Distance`.
     - Table header `Verdict Conf` -> `Classifier Conf`.
     - Non-attributed flows (normal or unclassified) display `--` in the classifier confidence column.
  3. **Preserved Raw Signals & API Contracts:** Raw VAE score ($s$), threshold ($\tau$), predicted class, and raw probability preserved as distinct, transparent signals across all API endpoints.
  4. **Calibration Feasibility Assessment:** Proven on `ml/splits/validation.csv` that VAE reconstruction error cannot be calibrated as a true probability of attack because the VAE only detects ~35% of attacks (stealth attacks have low reconstruction error). Disclosed uncalibrated model outputs transparently rather than inventing artificial calibration.

---

### 4.6 Hybrid Detection & Verdict Fusion (Iteration 5)
- **Objective Achieved:** Solved the stealth attack blind spot by bridging unsupervised anomaly detection (VAE) and 12-class supervised classification (XGBoost) into an explicit, transparent 4-state decision policy.
- **Decision Policy Specification:**
  - Let $s$ be VAE reconstruction error, $\tau = 38.50959$, $\hat{y}$ be classifier predicted class, $P_{xgb} = \max_k P_k$, and $\theta_{atk} = 0.50$ (configurable operating point):
  1. **Dual Confirmed Attack (`s > \tau` and `P_{xgb} \ge \theta_{atk}, \hat{y} \ne \text{BENIGN}`):**
     - Both models independently detect anomalous/malicious flow characteristics.
     - `is_anomaly = True`, `detection_source = "dual_confirmed"`, `verdict = f"Anomalous — {pred_lbl}"`, `attack_type = pred_lbl`, `verdict_confidence = P_{xgb}`.
  2. **Classifier-Detected Known Attack (`s \le \tau` and `P_{xgb} \ge \theta_{atk}, \hat{y} \ne \text{BENIGN}`):**
     - Rescues stealth attacks (PortScan, Patator, Bot, Web Attacks) that reconstruct with low error below VAE baseline.
     - `is_anomaly = True`, `detection_source = "classifier_attack"`, `verdict = f"Anomalous — {pred_lbl}"`, `attack_type = pred_lbl`, `verdict_confidence = P_{xgb}`.
  3. **VAE Anomaly / Unclassified Novel Attack (`s > \tau` and `P_{xgb} < \theta_{atk}` or `\hat{y} == \text{BENIGN}`):**
     - VAE threshold exceedance remains an anomaly signal; classifier predicting `BENIGN` **does not override** the VAE anomaly.
     - `is_anomaly = True`, `detection_source = "vae_anomaly"`, `verdict = "Anomalous — Attack Type Unclassified"`, `attack_type = "Attack Type Unclassified"`, `attack_confidence = None`, `verdict_confidence = anom_conf`.
  4. **Clear Normal Traffic (`s \le \tau` and `P_{xgb} < \theta_{atk}` or `\hat{y} == \text{BENIGN}`):**
     - Flow is within normal baseline and classifier detects no attack.
     - `is_anomaly = False`, `detection_source = "clear_normal"`, `verdict = "Normal Traffic"`, `attack_type = "Normal Traffic"`, `attack_confidence = None`, `verdict_confidence = anom_conf`.
- **Validation Operating Point Selection ($\theta = 0.50$):**
  - Sweep conducted on `ml/splits/validation.csv` (20,930 flows: 17,180 attacks, 3,750 benign).
  - Selected $\theta = 0.50$ as the optimal majority operating point:
    * Recall jumped from **37.66%** (VAE-only) to **99.95%** (+62.29%).
    * False positive rate only increased from **4.96%** (186 FP) to **5.09%** (191 FP) — adding just 5 false positives across 3,750 benign flows!
    * Preserved 100% detection on subtle web attacks (Web Attack XSS 100%, Web Attack Brute Force 99.56%), which drop to 61% and 74% if $\theta$ is set to 0.80.
- **Held-Out Test Set Verification (20,931 Flows):**
  - Recall: **99.95%** (vs 37.54% VAE-only, +62.41%).
  - Precision: **98.93%** (vs 97.33% VAE-only).
  - F1 Score: **99.44%** (vs 54.18% VAE-only).
  - Benign FPR: **4.93%** (vs 4.72% VAE-only, +0.21%).
  - PortScan recall: **100.00%** (vs 0.08% VAE-only).
  - FTP-Patator recall: **100.00%** (vs 0.00% VAE-only).
  - SSH-Patator recall: **100.00%** (vs 0.00% VAE-only).
  - Bot recall: **99.66%** (vs 5.12% VAE-only).
  - Web Attack Brute Force recall: **98.23%** (vs 4.42% VAE-only).
  - Web Attack XSS recall: **98.98%** (vs 3.06% VAE-only).

---

### 4.7 Production Hardening & Reliability (Iteration 6)
- **Database Persistence & ORM Integration:**
  - Implemented `database.py` using SQLAlchemy 2.0 (`FlowAuditRecord`) backed by local SQLite (`artifacts/cypher.db` or configurable via `CYPHER_DATABASE_URL`).
  - Stores all factual detection metadata: `flow_id`, `timestamp`, `src_ip`, `dst_ip`, `src_port`, `dst_port`, `protocol`, `score`, `threshold`, `is_anomaly`, `anomaly_confidence`, `attack_type`, `attack_confidence`, `classification_status`, `verdict`, `verdict_confidence`, `detection_source`.
  - Added indexes on `timestamp`, `src_ip`, `dst_ip`, `is_anomaly`, `detection_source`, and composite index `ix_flow_src_dst_ip`.
  - Excluded sensitive raw packet payloads and unneeded 77 raw float features from long-term relational storage.
  - Hardened against persistence failures: database errors log warnings but never crash real-time model inference responses.
- **LLM Alert Summarizer Resilience:**
  - Decoupled `llm.py` import from active environment keys: replaced eager module-level client instantiation with lazy `get_llm_client(timeout_seconds=10.0)`.
  - Added schema-compliant `generate_fallback_summary` that gracefully handles missing `GOOGLE_API_KEY`, request timeouts, HTTP 429 rate limits, and malformed provider output.
  - Enforced strict non-fabrication rule: fallbacks never invent an AI narrative; they truthfully present verified deterministic model outputs and standard SOC countermeasures.
- **Strict Input Validation & Non-Finite Rejection:**
  - Implemented `validate_input_row` in `main.py` across `/predict`, `/analyze`, `/attack-type`, `/anomaly-flag`, and `/analyze-batch`.
  - Rejects `NaN`, `Inf`, `-Inf`, unparseable non-numeric strings, and empty payloads with clear HTTP 422 Unprocessable Entity responses.
  - Eliminated silent zero-imputation in the public API layer while preserving backward compatibility for partial rows with missing features tracked in `input_issues`.
- **FastAPI Lifespan Migration:**
  - Replaced deprecated `@app.on_event("startup")` handler with modern async `lifespan(app)` context manager.
- **Cross-Session IP Investigation:**
  - Hardened `/inspect-ip` to seamlessly query persisted database audit logs when in-memory cache is empty or across server restarts.

---

### 4.8 Classifier Confidence Saturation & Display Correction (Iteration 7)
- **Empirical Probability Distribution on Validation Set (`ml/splits/validation.csv`, 20,930 flows):**
  - **Min:** 0.465488
  - **Median:** 0.999982
  - **90th Percentile:** 0.999995
  - **95th Percentile:** 0.999996
  - **99th Percentile:** 0.999998
  - **Max:** 0.999999
  - **Samples $\ge 0.999$:** 20,032 (95.71%)
  - **Samples $\ge 0.990$:** 20,524 (98.06%)
- **Source of Apparent 99.9% Saturation:**
  - **Model Output Distribution:** Uncalibrated tree-ensemble softmax probabilities from XGBoost naturally saturate near 1.0 for high-volume, easily separable classes (DDoS: 99.8% $\ge 0.999$, DoS Hulk: 98.8% $\ge 0.999$, PortScan: 98.2% $\ge 0.999$). Less frequent classes are NOT saturated (Web Attack XSS mean 0.7729, 0.0% $\ge 0.999$; Web Attack Brute Force mean 0.8997, median 0.9625, 5.2% $\ge 0.999$; Bot median 0.9995, min 0.5717).
  - **Frontend Display Cap:** `frontend/src/utils.ts` contained a hardcoded cap `if (pct >= 99.9) return "99.9%"` and 1-decimal rounding (`pct.toFixed(1)`). This collapsed all values $\ge 0.9985$ (95.71% of all evaluated flows) into the literal identical string `"99.9%"`.
  - **Backend Truncation:** `ml/pipeline.py` previously called `round(atk_c, 4)`, truncating fine-grained probabilities like 0.999982 into 1.0.
- **Implemented Fixes & Precision Preservation:**
  - **Removed Artificial Cap:** Eliminated `if (pct >= 99.9) return "99.9%"` in `frontend/src/utils.ts`.
  - **Two-Decimal Percentage Formatting:** Implemented `formatProbability` with 2 decimal places (`${clamped.toFixed(2)}%`) exposing true distinctions (e.g. 95.12%, 99.87%, 99.99%, 100.00%).
  - **Strict Null/Inapplicable Presentation:** Formatted unavailable, invalid, or normal traffic values as `—` (em-dash), eliminating fabricated percentages or ambiguous `--`.
  - **Semantics Correction:** Re-labeled all UI metrics to **“Classifier probability”** / **“Predicted-class probability”** instead of generic "confidence", making explicit that raw model outputs are uncalibrated softmax scores.
  - **Backend Preservation:** Increased precision to `round(atk_c, 6)` in `ml/pipeline.py` and exposed `classifier_probability` explicitly alongside `attack_confidence`.
  - **Model Decision Invariance:** VAE reconstruction error, threshold (38.50959), XGBoost predictions, attack labels, and 4-state hybrid detection decisions were kept 100% unchanged.

---

## 5. Verification Results (Iteration 7)

| Check Category | Command / Procedure | Result | Details |
|---|---|---|---|
| **Backend Test Suite** | `pytest tests/ -v` | **PASSED (67/67, 100%)** | 14 pipeline/confidence tests, 8 endpoint tests, 7 classifier tests, 6 held-out regression tests, 7 hybrid detection tests, 20 production hardening tests, 5 VAE determinism tests. |
| **Probability Formatting & Preservation** | `tests/test_confidence_and_pipeline.py` | **PASSED (14/14, 100%)** | Verified 2-decimal formatting (0.9512 -> 95.12%, 0.9987 -> 99.87%, 0.9999 -> 99.99%, 1.0 -> 100.00%), em-dash fallback for missing/invalid, and backend 6-decimal precision preservation. |
| **Frontend Production Build** | `npm --prefix frontend run build` | **PASSED** | `tsc -b && vite build` completed in 1.82s with zero TypeScript or packaging errors. |
| **API Backward Compatibility** | `test_hybrid_detection_fields_preserved` | **PASSED** | All hybrid verdict, confidence, contributor, and probability fields intact across all endpoints. |

---

---

## 6. Comprehensive Forensic Model Audit & Benchmark Results

### 6.1 Multi-Model Empirical Evaluation (Held-Out Test Set: 20,931 Flows)

An exhaustive benchmark was conducted on the exact same held-out test split (`ml/splits/test.csv`, 20,931 rows across 12 CICIDS2017 classes) comparing standalone supervised algorithms, ensembles, the VAE anomaly detector, and the legacy 8-expert MoE:

| Architecture / Model | Test Accuracy | Macro-F1 | Weighted-F1 | Latency (ms/1k flows) | Key Findings & Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost (Hist GBDT, 12 classes)** | **99.37%** | **0.9309** | **0.9936** | **10.44 ms** | Highest efficiency, self-contained JSON booster, robust across 12 classes (DDoS: 99.97%, DoS Hulk: 99.95%, PortScan: 99.95%, Benign: 99.79%). |
| **LightGBM (12 classes)** | **99.40%** | **0.9341** | **0.9940** | **27.28 ms** | Slight edge on minority Web Attacks; higher runtime latency. |
| **Random Forest (150 trees)** | **99.33%** | **0.9034** | **0.9930** | **5.85 ms** | Fast inference; lower recall on Web Attack XSS (36.7%). |
| **Weighted Ensemble (XGB+LGB+RF)** | **99.39%** | **0.9319** | **0.9938** | **51.54 ms** | Balanced trade-off; higher compute overhead without significant accuracy gain over XGBoost. |
| **VAE Anomaly Scorer Alone ($\tau=38.51$)** | **47.89%** | **0.5418** | **0.4789** | **12.50 ms** | **Catastrophic failure on stealth attacks:** PortScan recall 0.08%, SSH-Patator 0.0%, FTP-Patator 0.0%, Bot 5.1%. High detection on volumetric attacks (DDoS 65.4%, DoS 67.5%). |
| **Legacy 8-Expert MoE (`artifacts/moeModels`)** | **17.91%** | **0.0521** | **0.0543** | **> 1,500 ms** | **Catastrophic failure:** Trained on CSE-CIC-IDS2018 on only 3 classes; predicts 99.85% Benign on CICIDS2017. Missing 10 attack classes entirely. |
| **CYPHER 4-State Hybrid Engine (VAE + XGB)** | **99.07%** | **0.9944 (Det F1)** | **0.9907** | **18.20 ms** | **Overall Attack Recall: 99.95%** (17,172 / 17,181 attacks detected). Rescues stealth attacks via classifier while preserving VAE zero-day anomaly triage. |

### 6.2 Breakdown of Hybrid Detection Telemetry

In the 4-state hybrid architecture:
- `classifier_attack` (**10,730 flows**): Stealth attacks below VAE reconstruction threshold rescued by the supervised classifier (PortScan, Patator brute-force, low-rate DoS).
- `dual_confirmed` (**6,447 flows**): Volumetric attacks flagged simultaneously by VAE anomaly threshold and supervised classifier.
- `clear_normal` (**3,574 flows**): True negative benign flows within normal baseline boundaries.
- `vae_anomaly` (**180 flows**): High reconstruction error anomalies without a known supervised attack label match (zero-day candidates / novel statistical deviations).

---

## 7. Verification Results

| Check Category | Command / Procedure | Result | Details |
|---|---|---|---|
| **Backend Test Suite** | `pytest tests/ -v` | **PASSED (67/67, 100%)** | 14 pipeline/confidence tests, 8 endpoint tests, 7 classifier tests, 6 held-out regression tests, 7 hybrid detection tests, 20 production hardening tests, 5 VAE determinism tests. |
| **Frontend Production Build** | `npm --prefix frontend run build` | **PASSED** | `tsc -b && vite build` completed with zero TypeScript compilation or bundling errors. |
| **Classifier Benchmark** | `python ml/benchmark_classifiers.py` | **PASSED** | Generated comparative metrics saved to `ml/reports/classifier_benchmark_comparison.json`. |
| **VAE Determinism Verification** | `python -m ml.evaluate_vae` | **PASSED** | Inter-run variance on identical inputs strictly 0.000000. |

---

## 8. Final Independent ML Validation & End-to-End Verification

Full formal validation report available in [FINAL_VALIDATION_REPORT.md](file:///c:/Users/msi/Projects/Cypher/FINAL_VALIDATION_REPORT.md).

### 8.1 Key Independent Findings
- **XGBoost 12-Class Reproduction:** 99.37% test accuracy, 0.9309 Macro-F1, 0.9936 Weighted-F1 independently reproduced on 20,931 held-out test flows.
- **Metric Disentanglement:**
  - **Binary Attack Detection Recall:** **99.95%** (17,172 TP, 9 FN, 185 FP, 3,565 TN, FPR: 4.93%, Binary Accuracy: 99.07%).
  - **Multiclass Threat Attribution:** **98.52%** overall 12-class attribution accuracy (20,622 / 20,931) and **99.27%** on detected attack flows.
- **Data Leakage & Duplicate Overlap:** 11.79% (2,467 / 20,931) of test rows share identical 78-dimensional feature tuples with training data, caused by random stratified splitting on automated attack scripts (`SSH-Patator` 48.81%, `FTP-Patator` 27.28%, `DoS Hulk` 23.89%). Benign overlap is minimal (1.47%) and Web Attack XSS has 0.00% overlap. No models or thresholds were chosen via test set contamination.
- **Weak Class Analysis (`Web Attack XSS` & `Brute Force`):** Standalone XSS recall is 38.78% because 60.20% of XSS flows are classified as Web Attack Brute Force. However, **98.46% (319 / 324)** of all web attack flows are correctly flagged as Web Attacks. The ambiguity is caused by identical layer-3/4 flow statistical distributions targeting the same server over port 80 in the absence of HTTP payload inspection.
- **API Parity & Robustness:** 100% prediction parity between FastAPI endpoints (`/predict`, `/analyze-batch`) and direct model inference. Strict HTTP 422 rejections for `NaN`, `Inf`, empty payloads, and unparseable strings.
- **Automated Regression Suite:** 67 / 67 automated pytest tests passed in 11.75s.
- **Frontend Production Build:** `tsc -b && vite build` succeeded in 1.60s with zero errors.

---

## 9. Leakage-Resistant Group-Aware Evaluation & Generalization Milestone

Full formal evaluation report available in [LEAKAGE_RESISTANT_EVALUATION.md](file:///c:/Users/msi/Projects/Cypher/LEAKAGE_RESISTANT_EVALUATION.md).

### 9.1 Key Empirical Findings
- **Zero-Duplicate Split Guarantee:** Created [ml/splits_leakage_resistant/](file:///c:/Users/msi/Projects/Cypher/ml/splits_leakage_resistant) via Stratified Group-Aware Content Fingerprinting ([ml/split_leakage_resistant.py](file:///c:/Users/msi/Projects/Cypher/ml/split_leakage_resistant.py)). Exact cross-split duplicate feature overlap is **0.00% (0 rows)** across Train (97,965 rows), Validation (21,331 rows), and Test (20,239 rows). Original `ml/splits/` baseline preserved untouched.
- **Generalization Confirmed:**
  - **XGBoost 12-Class Accuracy:** **99.39%** (Macro-F1: **0.9268**, Weighted-F1: **0.9933**, Latency: **9.70 ms / 1k flows**).
  - **LightGBM Alternative:** **99.30%** (Macro-F1: **0.9200**, Latency: **14.30 ms / 1k flows**).
  - `SSH-Patator` achieved **100.0%** precision/recall, `FTP-Patator` achieved **99.90%**, `PortScan` achieved **99.89%**, and `DoS Hulk` achieved **99.91%** on completely unseen unique feature tuples, proving high performance is driven by genuine structural traffic signatures rather than memorized duplicate flows.
- **Hierarchical Web Attack Aggregation:** Grouping `Web Attack XSS` and `Web Attack Brute Force` into `Web Attack` achieves **99.38% recall** and **99.69% precision** (F1: **0.9953**), lifting overall hierarchical accuracy to **99.85%** (Macro-F1: **0.9977**).
- **Hybrid Pipeline (VAE + XGBoost):** 99.98% attack detection recall, 5.38% false positive rate, 98.76% overall multiclass attribution accuracy (99.71% on attacks).
- **Automated Regression Suite:** **72 / 72 tests passed** (including 5 new leakage-resistant automated tests).
- **Frontend Production Build:** Successful (`tsc -b && vite build` in 1.60s).

---

---

## 11. Detection Quality Verification & Separate Anomaly Alerts Milestone

Full formal evaluation report available in [LEAKAGE_RESISTANT_EVALUATION.md](file:///c:/Users/msi/Projects/Cypher/LEAKAGE_RESISTANT_EVALUATION.md).

### 11.1 Key Achievements & Empirical Findings
- **Reproduction of Current Evaluation:**
  - Evaluated on the held-out zero-duplicate test set (`ml/splits_leakage_resistant/test.csv`, 20,239 flows: 16,467 Attack, 3,772 Benign).
  - Multi-class test accuracy: **99.39%**, Macro-F1: **0.9268**, Weighted-F1: **0.9933**.
  - Binary Classifier-Only: TP = **16,464**, FP = **1**, TN = **3,771**, FN = **3**. Attack Recall: **99.98%**, FPR: **0.03%**, Precision: **99.99%**, Binary Accuracy: **99.98%**.
  - All denominators, confusion matrices, and per-class supports independently verified.
- **Decision Rule Consistency:**
  - Audited decision rules between argmax and thresholded confidence ($P_{\text{xgb}} \ge 0.50$). Exactly 3 flows out of 20,239 differed because model probability was split between `Web Attack XSS` (~49.9%) and `Web Attack Brute Force` (~49.8%), with combined non-benign probability $> 99.7%$.
- **Alert Architecture Decoupling & VAE Reality:**
  - On the held-out test set, VAE-only detection ($\tau = 38.51$) achieves only **3.39% attack recall** and detected **0 additional known attacks** that the supervised classifier missed.
  - Forcing an OR-based hybrid policy adds **202 false alarms** from benign flows with normal variance above $\tau$, jumping the False Positive Rate from **0.03%** to **5.38%** (a 179-fold increase).
  - **Decoupled 4-State Alert Semantics:**
    1. **`dual_signal`:** Both VAE anomaly and supervised classifier confirm known attack (6,361 flows).
    2. **`known_attack`:** Supervised classifier identified known attack; VAE within baseline (10,103 flows).
    3. **`anomaly_candidate`:** VAE reconstruction outlier; classifier predicts normal (BENIGN) (202 flows). Flagged for secondary operator triage, **not** promoted into a confirmed high-severity attack.
    4. **`normal_traffic`:** Within baseline and classified as BENIGN (3,569 flows).
  - Exposed `alert_type` across database schema (`FlowAuditRecord`), API endpoints (`/predict`, `/analyze-batch`, `/audit-logs`), and frontend UI console.
- **Hierarchical Web Attack Validation:**
  - Symmetrical aggregation to `Web Attack` achieves **99.38% recall** (320 / 322 flows) and **99.69% precision** (F1: **0.9953**).
  - Explicitly documented that network layer-4 statistical summaries cannot perform application payload-level XSS vs Brute Force discrimination; Deep Packet Inspection (DPI) is required for payload inspection.
- **Generalization Caveat:**
  - Duplicate-resistant evaluation eliminates literal feature vector memorization, but does not prove independent-session or real-world generalization due to testbed dataset origin and lack of session IDs, timestamps, and IP addresses.
- **Verification Suite:**
  - **81 / 81 backend tests passed** (`pytest tests/ -v` in 12.13s), including 6 new tests in `tests/test_alert_architecture.py`.
  - **Frontend production build succeeded** (`npm --prefix frontend run build` in 1.69s).
