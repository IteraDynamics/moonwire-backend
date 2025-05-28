# src/signal_log.py

import json
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "signal_history.jsonl"

# Ensure logs directory exists
LOG_DIR.mkdir(exist_ok=True)

def log_signal(signal_data: dict = None, **kwargs):
    try:
        entry = signal_data or {
            "id": kwargs["id"],
            "timestamp": kwargs["timestamp"],
            "asset": kwargs["asset"],
            "score": kwargs["score"],
            "confidence": kwargs["confidence"],
            "label": kwargs["label"],
            "trend": kwargs["trend"],
            "top_drivers": kwargs["top_drivers"],
            "fallback_type": kwargs["fallback_type"]
        }
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[Signal Logged] {entry}")
    except Exception as e:
        print(f"[Log Error] Could not log signal: {e}")
