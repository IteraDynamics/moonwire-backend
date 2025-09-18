# MoonWire Signal Engine – Development Plan (Sept 2025)

This document outlines MoonWire’s current backend development strategy and next-phase priorities.  
The engine has matured from early mock-mode sentiment APIs into a **fully modular, ledger-first ML system** with versioned retraining, explainability, and CI-driven diagnostics.

---

## 🎯 Strategic Goal

Build a backend-first framework that:

- Preserves every inference, label, and retrain in **append-only ledgers**  
- Surfaces **transparent, reproducible metrics** (precision, recall, F1) by origin and model version  
- Produces **explainable triggers** with drift/volatility context  
- Supports **continuous retraining** from feedback logs  
- Generates CI artifacts (JSON + charts) for human and machine consumption  
- Keeps frontend integration stable while backend grows more powerful

---

## ✅ Phase 1 – Foundations (completed)

- Structured ledgers: `trigger_history.jsonl`, `label_feedback.jsonl`, `training_data.jsonl`, `training_runs.jsonl`  
- Model version tagging across inferences, labels, and retrains  
- Volatility- and drift-aware thresholds  
- Modularized CI summary (`scripts/summary_sections/`) with >15 diagnostic sections  
- Demo seeding for CI visibility without live data  

---

## 🚀 Phase 2 – Performance & Quality Monitoring (in progress)

- Per-origin and per-version accuracy snapshots (precision, recall, F1)  
- Signal quality summaries (batch, per-origin, per-version)  
- Trend charts for accuracy, coverage, suppression, and quality  
- Threshold quality analysis and guardrailed recommendations  
- Score distribution overlays with drift splits  

---

## 🔮 Phase 3 – Towards Auto-Adaptive Models (upcoming)

- Live integration of real social/news/market APIs (Twitter/X, Reddit, CoinGecko, etc.)  
- Continuous retraining pipelines gated by quality thresholds  
- Auto-application of recommended thresholds (with reviewer override)  
- Alerting on regressions (e.g., F1 drop > X% for a version or origin)  
- Suppression/coverage/precision dashboards for operators  

---

## 🧱 Phase 4 – Scaling & Governance

- Persistent artifact store for long-term analytics (beyond JSONL/PNG)  
- Comparison dashboards across training runs and model versions  
- Human-in-the-loop workflows for reviewer confirmation/rejection  
- Drift/volatility-aware retrain scheduling  
- Enterprise-friendly governance (audit exports, provenance checks, alerts)  

---

## 📂 Current Branch Structure

- `main` → stable, CI-green, demo-safe  
- `feature/*` → individual feature branches (merged after test + review)  
- Artifacts written to `/models/` (JSON, ledgers, metadata) and `/artifacts/` (charts, histograms, summaries)  

---

## 🔒 Guiding Principle

> **Every signal must be explainable, versioned, and reproducible.**

MoonWire is not just an engine for signals — it is a system for **proving signal quality** in real time.

---

## 📜 Archive – Original Dev Plan (May 2024)

This section preserves the original foundation plan for context and historical reference.

### MoonWire Signal Engine Foundation – Dev Plan (May 2024)

This document outlines the foundational engineering work for MoonWire’s AI/ML signal engine. All tasks are focused on data integrity, internal traceability, and model-readiness without breaking the current frontend or requiring immediate ML infrastructure.

---

### Strategic Goal

Build a backend-first framework that:
- Logs every signal event with metadata
- Captures fallback types and price context
- Enables internal testing of composite signal formats
- Keeps frontend stable and unaware of backend transitions

---

### Phase 1 – Signal Logging Infrastructure

Purpose: Create structured signal logs for every backend-generated score.

Tasks:
- Create `signal_logger.py` in `src/`
- Define `SignalLog` schema:
  - asset
  - timestamp
  - source (twitter, news, etc.)
  - score (float)
  - fallback_type (mock, cached, live)
  - price_at_score (optional, USD)
- Add logging call inside `/sentiment/twitter` and `/news-sentiment`
- For now, print logs to console or write to JSON file

---

### Phase 2 – Market Price Capture

Purpose: Enrich logs with real-time asset price to support future signal scoring.

Tasks:
- Add helper in `market_price.py` to fetch price via CoinGecko
- Cache last known price to avoid rate limits
- Integrate into signal logging flow

---

### Phase 3 – Fallback Type Tagging

Purpose: Enable trust analysis and training segmentation for mock vs. real data.

Tasks:
- Add `source_type` field to all sentiment responses
- Return value as mock, cached, or live
- Include in `SignalLog` output

---

### Phase 4 – Mock Signal Renderer

Purpose: Prototype what a future model output might look like.

Tasks:
- Define static `/signals/mock` endpoint
- Return JSON like:

```json
{
  "asset": "BTC",
  "score": 0.72,
  "confidence": 0.81,
  "trend": "strengthening",
  "label": "Bullish Momentum",
  "top_drivers": ["etf", "approval", "breakout"]
}
