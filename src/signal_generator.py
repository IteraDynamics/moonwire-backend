# src/signal_generator.py

from datetime import datetime
from src.signal_filter import is_signal_valid
from src.cache_instance import cache
from src.sentiment_blended import blend_sentiment_scores
from src.dispatcher import dispatch_alerts
from src.ingest_discovery import ingest_market_data

def generate_signals():
    print(f"[{datetime.utcnow()}] Running signal generation...")
    
    # Ensure fresh market data is available
    ingest_market_data(cache)

    stablecoins = {"USDC", "USDT", "DAI", "TUSD", "BUSD"}
    valid_signals = []

    try:
        sentiment_scores = blend_sentiment_scores()
        assets = [k for k in cache.keys() if not k.endswith('_signals') and not k.endswith('_sentiment')]
        
        print(f"[DEBUG] Found assets in cache: {assets}")

        for asset in assets:
            if asset in stablecoins:
                continue

            data = cache.get_signal(asset)
            if not isinstance(data, dict):
                print(f"[ERROR] Expected dict for signal data, got {type(data)} — asset: {asset}")
                continue
            
            print(f"[DEBUG] {asset} data: {data}")

            price_change = data.get("price_change_24h")
            volume = data.get("volume_now")

            if price_change is None or volume is None:
                print(f"[DEBUG] {asset} missing required fields - price_change_24h: {price_change}, volume_now: {volume}")
                continue

            sentiment = sentiment_scores.get(asset, 0.0)
            confidence = round(((price_change / 10) + sentiment) / 2, 2)

            signal = {
                "asset": asset,
                "price_change": price_change,
                "volume": volume,
                "sentiment": sentiment,
                "confidence_score": confidence,
                "confidence_label": label_confidence(confidence),
                "timestamp": datetime.utcnow()
            }

            is_valid = is_signal_valid(signal)
            print(f"[DEBUG] {asset} signal valid: {is_valid} | movement: {price_change}% | confidence: {confidence}")
            
            if is_valid:
                dispatch_alerts(asset, signal, cache)
                valid_signals.append(signal)
                print(f"[SIGNAL GENERATED] {asset}: {signal}")

    except Exception as e:
        print(f"[ERROR] Failed to generate signals: {e}")

    print(f"[SIGNAL GENERATION COMPLETE] Generated {len(valid_signals)} valid signals")
    return valid_signals

def label_confidence(score):
    if score >= 0.66:
        return "High Confidence"
    elif score >= 0.33:
        return "Medium Confidence"
    else:
        return "Low Confidence"