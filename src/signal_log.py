import json
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "signal_history.jsonl"

# Ensure logs directory exists
LOG_DIR.mkdir(exist_ok=True)

def log_signal(asset, movement, volume, sentiment, confidence, timestamp):
    try:
        entry = {
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "asset": asset,
            "movement_percent": movement,
            "volume_usd": volume,
            "sentiment_score": round(sentiment, 4),
            "confidence_score": round(confidence, 4)
        }
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[Signal Logged] {entry}")
    except Exception as e:
        print(f"[Log Error] Could not log signal for {asset}: {e}")

def log_composite_signal(signal: dict):
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(signal) + "\n")
        print(f"[Composite Signal Logged] {signal}")
    except Exception as e:
        print(f"[Composite Log Error] {e}")