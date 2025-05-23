# src/signal_log.py

import json
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "signal_history.jsonl"

# Ensure logs directory exists
LOG_DIR.mkdir(exist_ok=True)

def log_signal(
    id: str,
    asset: str,
    score: float,
    confidence: str,
    label: str,
    trend: str,
    top_drivers: list,
    timestamp: str,
    fallback_type: str
):
    try:
        entry = {
            "id": id,
            "timestamp": timestamp,
            "asset": asset,
            "score": score,
            "confidence": confidence,
            "label": label,
            "trend": trend,
            "top_drivers": top_drivers,
            "fallback_type": fallback_type
        }
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[Signal Logged] {entry}")
    except Exception as e:
        print(f"[Log Error] Could not log signal for {asset}: {e}")