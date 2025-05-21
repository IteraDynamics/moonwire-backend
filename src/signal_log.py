from datetime import datetime
import json
import os

# Optional: Path for local logging during development
LOG_PATH = "signal_logs.json"

class SignalLog:
    def __init__(self, asset, source, score, fallback_type=None, price_at_score=None):
        self.asset = asset
        self.source = source
        self.score = score
        self.fallback_type = fallback_type
        self.price_at_score = price_at_score
        self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self):
        return {
            "asset": self.asset,
            "timestamp": self.timestamp,
            "source": self.source,
            "score": self.score,
            "fallback_type": self.fallback_type,
            "price_at_score": self.price_at_score
        }


def log_signal(signal: SignalLog):
    entry = signal.to_dict()
    
    # Console log for quick feedback
    print("[SignalLog]", json.dumps(entry, indent=2))

    # Optional: append to local JSON file for persistent logs
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(entry)

    with open(LOG_PATH, "w") as f:
        json.dump(logs, f, indent=2)


# Example usage:
# log_signal(SignalLog(asset="BTC", source="twitter", score=0.42, fallback_type="mock", price_at_score=67420.00))
