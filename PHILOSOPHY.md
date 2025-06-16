# MoonWire Backend – Philosophy & Signal Governance

## 🧭 Guiding Belief

MoonWire is built from the belief that:

> **"Signals must be trustworthy before they are useful."**

This backend architecture reflects that truth. Every signal, score, feedback loop, and tuning simulation exists to support a single goal:

> **"All data is innocent until proven biased — and all signals are guilty until they can explain themselves."**

---

## 🎯 System Design Principles

### 1. **Feedback is not cosmetic — it’s core.**
Feedback is not a UI feature. It's a signal correction vector. Every user response is logged, scored for reliability, and attached to its parent signal with traceability.

### 2. **Not all feedback is equal.**
MoonWire includes a feedback scoring engine that evaluates confidence-weighted reliability. The system is designed to listen to the *most certain, consistent corrections* — not just raw counts.

### 3. **Volatility is an input.**
Signal disagreement and confidence variance are measured and surfaced as part of system QA. If an asset is unstable, MoonWire knows it — and adjusts accordingly.

### 4. **Everything is structured. Everything is logged.**
- Signals are logged to `signal_history.jsonl`
- Feedback entries are timestamped and typed
- Every trend, override, or score adjustment is traceable and exportable

### 5. **This system is built to learn.**
MoonWire’s backend isn't just about real-time signals — it's about creating labeled, exportable, high-quality datasets for future supervised training. Every feature is constructed with ML readiness in mind.

---

## 📤 Outputs That Matter

MoonWire produces:
- Composite signals with explainable trend deltas
- Trust-weighted feedback data
- Label datasets in CSV/Parquet format for ML
- Threshold simulation tools for signal QA
- Disagreement volatility metrics by asset

---

## 🔒 Why This Exists

Because crypto is noisy. AI is messy. And most “signal engines” are black boxes with no accountability.

MoonWire is not that.

It’s not just here to show you signals.  
It’s here to explain them, evolve them, and prove they’re worth trusting.

---

> We didn’t waste time building a product.  
> We spent time building a product that won’t waste its own time later.