# MoonWire Signal Engine – Backend

This is the backend engine for **MoonWire** — a real-time crypto signal intelligence platform that blends social sentiment, news narratives, and market behavior into **actionable indicators**.

The backend now powers:
- **Trigger detection & scoring** (statistical + ML ensemble)
- **Feedback-driven retraining** (continuous learning loop)
- **Per-origin / per-version analytics** (precision, recall, F1)
- **Threshold optimization & backtesting**
- **Automated CI summaries with charts, metrics, and recommendations**

---

## 🚀 Overview

MoonWire’s backend ingests candidate events (Twitter, Reddit, news, market feeds), scores them through ML models, and produces **signal triggers** with rich provenance. Every inference, label, retrain, and threshold decision is logged, versioned, and surfaced in CI.

The system is fully modular:
- Each diagnostic (accuracy, drift, coverage, suppression, precision, etc.) lives in its own section.
- CI runs assemble these into a **demo summary report** with charts and artifacts.
- Demo mode seeds synthetic rows for visibility even when logs are sparse.

---

## ✨ Key Features

- **FastAPI server** — lightweight async API for scoring and feedback
- **Trigger + label logs** — append-only JSONL for complete provenance
- **On-demand retraining** — Logistic, RF, and GB ensembles retrain from feedback logs
- **Training metadata ledger** — every retrain is logged with metrics, features, and version
- **Per-origin & per-version analytics** — precision/recall/F1, score distributions, drift splits
- **Threshold engine** — per-origin thresholds, recommendations, and backtests
- **Coverage & suppression metrics** — track firing rates vs. held-back signals
- **Automated CI summaries** — rich markdown + charts + artifacts on every run
- **Demo mode** — seeded plausible data for visibility in early or sparse runs

---

## 🛠 Tech Stack

- **Python 3.10+**
- **FastAPI + Uvicorn** — async inference + feedback endpoints
- **scikit-learn** — ML models (logistic, RF, gradient boosting)
- **matplotlib** — CI-safe chart rendering (Agg backend)
- **pytest** — full test suite with green CI gating
- **GitHub Actions** — CI/CD, artifact publishing, job summaries

---

## 📂 Core Modules

| Module | Purpose |
|--------|---------|
| `main.py` | FastAPI app entrypoint, mounts inference + feedback routes |
| `src/infer.py` | Ensemble inference, now logs `model_version` on every trigger |
| `src/retrain_from_log.py` | Retrains models from `training_data.jsonl`, saves versioned artifacts |
| `src/training_metadata.py` | Appends structured metadata ledger for every training run |
| `src/ml/metrics.py` | Accuracy, precision/recall/F1 (per-version, per-origin) |
| `scripts/mw_demo_summary.py` | Orchestrator: calls modular sections to build CI markdown |
| `scripts/summary_sections/*` | Modular summary blocks (accuracy, thresholds, drift, coverage, suppression, precision, etc.) |
| `src/paths.py` | Robust path/env handling for logs, models, and artifacts |

---

## 🌐 API Endpoints

| Route | Purpose |
|-------|---------|
| `/internal/trigger-likelihood/score?use=ensemble` | Run inference on candidate features |
| `/internal/trigger-likelihood/feedback` | Submit label feedback for a scored trigger (auto-tags version) |
| (more internal routes exist for retraining, testing, and CI hooks) |

---

## 📊 CI Summary (Artifacts)

Every CI run produces:
- **Markdown job summary** — readable report with metrics, tables, and charts
- **Artifacts/** — PNG trend charts (score distributions, signal quality, coverage, suppression, etc.)
- **Models/** — versioned models, training metadata, accuracy snapshots, threshold recommendations, backtests

Examples:
- Signal Quality (per-origin, per-version, and trend charts)
- Score Distributions with drift overlays
- Threshold Quality + Recommendations + Backtest uplift report
- Trigger Coverage + Suppression metrics and trends

---

## ⚙️ How It Works

1. **Inference**  
   Candidates (from logs or live sources) → ensemble models → trigger scores.  
   Every prediction logged with model version.

2. **Feedback**  
   Human (or simulated) labels posted via API.  
   Joined to triggers → written to `label_feedback.jsonl`.

3. **Training Data Log**  
   Continuous join of triggers + labels → `training_data.jsonl`.

4. **Retraining**  
   Retrains models on demand or in CI.  
   Saves versioned artifacts and appends metadata to `training_runs.jsonl`.

5. **CI Summary**  
   Orchestrator assembles modular diagnostics into a full report.  
   Artifacts (charts, JSON) uploaded for visibility and reproducibility.

---

## 🧪 Development & Testing

- **Run tests:**  
  ```bash
  pytest -q
