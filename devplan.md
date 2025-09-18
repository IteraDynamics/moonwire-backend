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

**Next milestone:** Integrate live data feeds, close the feedback loop with real-world validation, and transition from **proven prototype** → **validated signal engine**.
