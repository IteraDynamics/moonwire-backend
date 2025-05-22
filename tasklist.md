MoonWire Signal Engine – Tasklist (May 2024)

This doc tracks tactical engineering steps for MoonWire’s signal engine as we move from passive sentiment logging to active signal generation and future model integration.

⸻

✅ Phase 1 Complete: Logging Infrastructure
	•	signal_log.py module created
	•	Twitter + News endpoints integrated with logger
	•	Render-safe environment check added
	•	Fallback types and timestamps logged

⸻

Phase 2 – Price Enrichment Layer (High Priority)

Goal: Attach price_at_score to every signal log

Tasks:
	•	Create price_fetcher.py in src/
	•	Use CoinGecko API to get USD price per asset
	•	Add in-memory cache to prevent rate limit hits
	•	Update log_signal() call in Twitter and News routers to include price
	•	Gracefully handle price fetch failures (fallback to None)

Value: Enables market-aware signal logs, future divergence detection
Risk: Low – isolated enrichment step

⸻

Phase 3 – Signal Mock Output View (Frontend Dev-Only)

Goal: Visualize a signal card powered by fake logic

Tasks:
	•	Create MockSignalCard.jsx
	•	Display: score, label, confidence, top drivers, trend arrow
	•	Populate from static object or dev-only API route
	•	Add toggleable view to frontend UI

Value: UX validation, label design testing
Risk: None – isolated from production

⸻

Phase 4 – Feedback Form Integration

Goal: Replace placeholder with real user input capture

Tasks:
	•	Create live Google Form or Typeform
	•	Update feedback button in App.jsx with real link
	•	Optional: track asset name in form metadata (manual for now)

Value: Opens feedback loop for UX + signal perception
Risk: None – frontend only

⸻

Phase 5 – Composite Signal Staging Function

Goal: Internal-only logic engine to simulate model output

Tasks:
	•	Create signal_composer.py
	•	Define generate_mock_composite() with:
	•	Inputs: asset, sentiment score, fallback flag, optional price delta
	•	Output: JSON with label, score, confidence, trend, top_drivers
	•	Log or print result only (not connected to frontend/API yet)

Value: Starts real signal shaping logic
Risk: None – internal test only

⸻

Notes
	•	All current features degrade gracefully
	•	No DB or persistent infrastructure required
	•	All work is tracked under signal-engine-foundation-may24

Build lean. Log smart. Score later.