# MoonWire Backend – Philosophy & Signal Governance

## 🧭 Guiding Belief

MoonWire is built on a simple truth:

> **"Signals must be auditable before they are actionable."**

Every log, chart, and recommendation in this backend exists to enforce that principle.  
No silent heuristics. No unexplained triggers. Every decision is accompanied by proof.

---

## 🎯 System Design Principles

### 1. **Feedback is the ground truth.**
Labels are not optional or cosmetic — they are the backbone of retraining.  
Every user-confirmed or rejected signal is logged, joined to its trigger, and preserved in append-only ledgers.

### 2. **Provenance is non-negotiable.**
- Every inference is stamped with the model version that produced it.  
- Every label records the version it judged.  
- Every retrain is logged with dataset stats, metrics, and top features.  
You can always trace *which model made which call, on which data*.

### 3. **Explainability is a feature, not an afterthought.**
Each trigger includes:
- Adjusted score vs threshold  
- Drift penalties and volatility multipliers  
- Top contributing features  
The “why” is never hidden.

### 4. **Quality is measured in public.**
MoonWire continuously surfaces:
- Precision, recall, and F1 by origin  
- Accuracy snapshots by version  
- Coverage vs suppression rates  
- Trend charts across time windows  
The system inspects itself so humans don’t have to guess.

### 5. **Thresholds are guided, not guessed.**
Recommendations come from real precision–recall sweeps, with guardrails to prevent wild shifts.  
Decisions are explainable, reproducible, and demo-safe.

### 6. **Everything is structured. Everything is logged.**
- `trigger_history.jsonl` → every inference  
- `label_feedback.jsonl` → every ground-truth correction  
- `training_data.jsonl` → joined, model-ready rows  
- `training_runs.jsonl` → retrain metadata and metrics  

MoonWire’s ledger-first design means nothing disappears, and every artifact can be replayed.

---

## 📤 Outputs That Matter

MoonWire produces:
- Transparent CI summaries for every run  
- Append-only JSONL ledgers for inference, feedback, training, and retrains  
- Machine-readable JSON artifacts for coverage, precision, and threshold quality  
- Headless charts (PNG) for trends and distributions  
- Threshold recommendations with clear objectives and guardrails  

---

## 🔒 Why This Exists

Because crypto is noisy. AI is messy. And most “signal engines” are black boxes with unverifiable claims.  

MoonWire is not that.  

It doesn’t just generate signals.  
It explains them, measures them, and proves when they’re worth trusting.  

---

> We didn’t just build a signal engine.  
> We built one that can prove itself, version by version, signal by signal.
