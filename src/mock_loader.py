# src/mock_loader.py

from datetime import datetime, timedelta
from src.cache_instance import cache

def load_mock_cache_data():
    now = datetime.utcnow()
    mock_history = [
        {
            "price_change": 2.1,
            "sentiment": 0.42,
            "confidence_score": 0.8,
            "timestamp": (now - timedelta(hours=24)).isoformat()
        },
        {
            "price_change": 2.8,
            "sentiment": 0.45,
            "confidence_score": 0.82,
            "timestamp": (now - timedelta(hours=6)).isoformat()
        },
        {
            "price_change": 3.2,
            "sentiment": 0.5,
            "confidence_score": 0.85,
            "timestamp": now.isoformat()
        }
    ]

    for asset in ["BTC", "ETH"]:
        cache.set_signal(f"{asset}_history", mock_history.copy())
        cache.set_signal(asset, mock_history[-1])
        print(f"[Mock Loaded] {asset} -> {len(mock_history)} entries")