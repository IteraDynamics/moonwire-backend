# src/trend_deltas.py

import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

LOG_FILE = Path("logs/signal_history.jsonl")

def load_signal_history():
    signals_by_asset = defaultdict(list)
    if not LOG_FILE.exists():
        print(f"[Warning] Log file not found: {LOG_FILE}")
        return load_mock_history()  # ✅ NEW: fallback to mock

    with open(LOG_FILE, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                asset = entry.get("asset")
                ts = entry.get("timestamp")
                if asset and ts:
                    entry["parsed_ts"] = datetime.fromisoformat(ts)
                    signals_by_asset[asset].append(entry)
            except json.JSONDecodeError:
                continue

    if not signals_by_asset:
        print("[Warning] No signal data found. Falling back to mock.")
        return load_mock_history()  # ✅ NEW: fallback if log is empty

    return signals_by_asset

def load_mock_history():
    now = datetime.utcnow()
    return {
        "BTC": [
            {"score": 0.42, "parsed_ts": now - timedelta(hours=24)},
            {"score": 0.45, "parsed_ts": now - timedelta(hours=6)},
            {"score": 0.5, "parsed_ts": now}
        ],
        "ETH": [
            {"score": 0.35, "parsed_ts": now - timedelta(hours=24)},
            {"score": 0.4, "parsed_ts": now - timedelta(hours=6)},
            {"score": 0.39, "parsed_ts": now}
        ]
    }

def compute_trend_deltas(signals_by_asset):
    now = datetime.utcnow()
    delta_6h = timedelta(hours=6)
    delta_24h = timedelta(hours=24)

    output = {}

    for asset, entries in signals_by_asset.items():
        sorted_entries = sorted(entries, key=lambda x: x["parsed_ts"])
        latest = sorted_entries[-1] if sorted_entries else None
        if not latest:
            continue

        score_now = latest.get("score") or latest.get("confidence") or 0

        score_6h_ago = find_closest_score(sorted_entries, now - delta_6h)
        score_24h_ago = find_closest_score(sorted_entries, now - delta_24h)

        output[asset] = {
            "trend_6h": round(score_now - score_6h_ago, 4),
            "trend_24h": round(score_now - score_24h_ago, 4)
        }

    return output

def find_closest_score(entries, target_time):
    closest = None
    min_diff = timedelta.max
    for entry in entries:
        diff = abs(entry["parsed_ts"] - target_time)
        if diff < min_diff:
            min_diff = diff
            closest = entry
    return closest.get("score") or closest.get("confidence") or 0

if __name__ == "__main__":
    history = load_signal_history()
    deltas = compute_trend_deltas(history)
    print(json.dumps(deltas, indent=2))
