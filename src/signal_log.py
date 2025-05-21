import json
import os
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "signal_history.jsonl"
IS_LOCAL = os.environ.get("RENDER") is None

# Only attempt to create logs directory if local
if IS_LOCAL:
    LOG_DIR.mkdir(exist_ok=True)

def log_signal(
    asset,
    source,
    score,
    confidence=None,
    fallback_type=None,
    price_at_score=None,
    movement=None,
    volume=None,
    timestamp=None
):
    try:
        entry = {
            "timestamp": (timestamp or datetime.utcnow()).strftime("%Y-%m-%d %H:%M:%S"),
            "asset": asset,
            "source": source,
            "score": round(score, 4) if score is not None else None,
            "confidence": round(confidence, 4) if confidence is not None else None,
            "fallback_type": fallback_type,
            "price_at_score": round(price_at_score, 2) if price_at_score is not None else None,
            "movement_percent": round(movement, 4) if movement is not None else None,
            "volume_usd": round(volume, 2) if volume is not None else None
        }

        # Always print to console
        print("[Signal Logged]", json.dumps(entry, indent=2))

        # Only write to file locally
        if IS_LOCAL:
            with open(LOG_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")

    except Exception as e:
        print(f"[Log Error] Could not log signal for {asset}: {e}")