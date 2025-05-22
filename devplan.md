MoonWire Signal Engine Foundation – Dev Plan (May 2024)

This document outlines the foundational engineering work for MoonWire’s AI/ML signal engine. All tasks are focused on data integrity, internal traceability, and model-readiness without breaking the current frontend or requiring immediate ML infrastructure.

⸻

Strategic Goal

Build a backend-first framework that:
	•	Logs every signal event with metadata
	•	Captures fallback types and price context
	•	Enables internal testing of composite signal formats
	•	Keeps frontend stable and unaware of backend transitions

⸻

Phase 1 – Signal Logging Infrastructure

Purpose: Create structured signal logs for every backend-generated score.

Tasks:
	•	Create signal_logger.py in src/
	•	Define SignalLog schema:
	•	asset
	•	timestamp
	•	source (twitter, news, etc.)
	•	score (float)
	•	fallback_type (mock, cached, live)
	•	price_at_score (optional, USD)
	•	Add logging call inside /sentiment/twitter and /news-sentiment
	•	For now, print logs to console or write to JSON file

⸻

Phase 2 – Market Price Capture

Purpose: Enrich logs with real-time asset price to support future signal scoring.

Tasks:
	•	Add helper in market_price.py to fetch price via CoinGecko
	•	Cache last known price to avoid rate limits
	•	Integrate into signal logging flow

⸻

Phase 3 – Fallback Type Tagging

Purpose: Enable trust analysis and training segmentation for mock vs. real data.

Tasks:
	•	Add source_type field to all sentiment responses
	•	Return value as mock, cached, or live
	•	Include in SignalLog output

⸻

Phase 4 – Mock Signal Renderer

Purpose: Prototype what a future model output might look like.

Tasks:
	•	Define static /signals/mock endpoint
	•	Return JSON like:

{
  "asset": "BTC",
  "score": 0.72,
  "confidence": 0.81,
  "trend": "strengthening",
  "label": "Bullish Momentum",
  "top_drivers": ["etf", "approval", "breakout"]
}

	•	Hook into future private frontend testing page

⸻

Notes
	•	No database required at this phase (JSON logs or console is fine)
	•	Everything should degrade gracefully for frontend compatibility
	•	This branch should not alter live production views unless explicitly opted in

⸻

Branch: signal-engine-foundation-may24

Start here. Set the groundwork for everything MoonWire’s signal engine will need to scale.