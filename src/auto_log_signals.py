# src/auto_log_signals.py

from datetime import datetime
from src.signal_log import log_signal

# === Mock signal snapshot (can later be replaced by cache fetch) ===
mock_signals = [
    {
        "asset": "BTC",
        "score": 0.48,
        "confidence": 0.87,
        "label": "Positive",
        "trend": "Stable",
        "top_drivers": ["news", "price"],
        "fallback_type": "mock"
    },
    {
        "asset": "ETH",
        "score": 0.42,
        "confidence": 0.81,
        "label": "Neutral",
        "trend": "Moving",
        "top_drivers": ["social", "price"],
        "fallback_type": "mock"
    }
]

def run_signal_logger():
    now = datetime.utcnow().isoformat()
    for signal in mock_signals:
        entry = {
            "type": "signal",
            "timestamp": now,
            **signal
        }
        log_signal(entry)

    print(f"[✔] Logged {len(mock_signals)} signal(s) at {now}")

if __name__ == "__main__":
    run_signal_logger()