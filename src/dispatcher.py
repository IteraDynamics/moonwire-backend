# src/dispatcher.py

import logging
import json
from pathlib import Path
from datetime import datetime
from src.cache import SignalCache
from src.emailer import send_email_alert

logger = logging.getLogger(__name__)

STANDARD_SIGNALS_FILE = Path('models/standard/signals.jsonl')
ELITE_SIGNALS_FILE = Path('models/elite/signals.jsonl')

def dispatch_alerts(asset: str, signal: dict, cache: SignalCache):
    print(f"[Dispatch] Alert triggered for {asset}: {signal}")
    
    # Get current price from cache (stored by ingest_discovery)
    asset_data = cache.get_signal(asset)
    print(f"[DEBUG] Asset data from cache for {asset}: {asset_data}")
    current_price = asset_data.get("current_price", 0) if isinstance(asset_data, dict) else 0
    print(f"[DEBUG] Extracted current_price: {current_price}")

    # Save signal to cache
    cache.set_signal(asset, signal)

    # Also save to history (as a separate entry)
    history_key = f"{asset}_history"
    history_entry = {
        "price_change": signal["price_change"],
        "volume": signal["volume"],
        "sentiment": signal["sentiment"],
        "confidence_score": signal["confidence_score"],
        "confidence_label": signal.get("confidence_label", "Unknown"),
        "timestamp": signal["timestamp"]
    }
    cache.set_signal(history_key, history_entry)
    print(f"[Dispatch] Saved to history: {history_key}")

    # Write to JSONL files for Discord bot
    # Determine tier based on confidence (can adjust thresholds later)
    confidence = signal.get("confidence_score", 0)
    direction = "long" if signal["price_change"] > 0 else "short"
    
    # Generate unique ID from timestamp + asset
    signal_id = f"{asset}_{int(datetime.utcnow().timestamp() * 1000)}"
    
    signal_entry = {
        "id": signal_id,
        "symbol": asset,
        "direction": direction,
        "confidence": abs(confidence),
        "price": current_price,
        "ts": datetime.utcnow().isoformat() + 'Z'
    }
    
    # Write to standard tier (for now, write all signals to both)
    STANDARD_SIGNALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STANDARD_SIGNALS_FILE, 'a') as f:
        f.write(json.dumps(signal_entry) + '\n')
    print(f"[Dispatch] Written to {STANDARD_SIGNALS_FILE}")
    
    # Write to elite tier as well
    ELITE_SIGNALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ELITE_SIGNALS_FILE, 'a') as f:
        f.write(json.dumps(signal_entry) + '\n')
    print(f"[Dispatch] Written to {ELITE_SIGNALS_FILE}")

    # Email disabled for testing - was causing hangs
    # send_email_alert(subject, body)
    print(f"[Dispatch] Complete for {asset}")