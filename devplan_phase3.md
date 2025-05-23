# MoonWire Signal Engine – Phase 3: Composite Signal Scoring & Feedback Prep

This phase transitions the MoonWire backend from raw sentiment logging to structured signal generation, feedback readiness, and user-facing interpretability.

---

## 1. Composite Signal Function (Core)

**File:** `src/signal_composer.py`  
**Function:** `generate_signal(asset, sentiment_score, price_at_score, fallback_type)`

### Returns:
```json
{
  "id": "sig_004",
  "asset": "ETH",
  "score": 0.78,
  "confidence": "medium",
  "label": "Bullish Sentiment Divergence",
  "trend": "upward",
  "top_drivers": ["social volume", "price breakout"],
  "timestamp": "ISO 8601",
  "fallback_type": "mock"
}