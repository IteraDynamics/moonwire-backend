# src/signal_generator.py

from datetime import datetime
from src.signal_filter import is_signal_valid
from src.cache_instance import cache
from src.sentiment_blended import blend_sentiment_scores
from src.dispatcher import dispatch_alerts

def generate_signals():
    print(f"[{datetime.utcnow()}] Running signal generation...")

    stablecoins = {"USDC", "USDT", "DAI", "TUSD", "BUSD"}
    valid_signals = []

    try:
        sentiment_scores = blend_sentiment_scores()
        assets = [k for k in cache.keys() if not k.endswith('_signals') and not k.endswith('_sentiment')]

        for asset in assets:
            if asset in stablecoins:
                continue

            data = cache.get_signal(asset)
            if not isinstance(data, dict):
                print(f"[ERROR] Expected dict for signal data, got {type(data)} — asset: {asset}")
                continue

            price_change = data.get("price_change_24h")
            volume = data.get("volume_now")

            if price_change is None or volume is None:
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

            # --- ML Inference Bridge (shadow mode) ----------------------------
            # Logs a model-driven "what would we have predicted?" record
            # to logs/signal_inference_shadow.jsonl when MW_INFERENCE_BRIDGE is
            # set to 'shadow' or 'on'. No behavior change to current flow.
            try:
                from src.inference.bridge_adapter import run_bridge_shadow
                run_bridge_shadow(asset)  # safe no-op if disabled or unavailable
            except Exception:
                # Never allow shadow logging to impact production flow
                pass
            # ------------------------------------------------------------------

            if is_signal_valid(signal):
                dispatch_alerts(asset, signal, cache)
                valid_signals.append(signal)

    except Exception as e:
        print(f"[ERROR] Failed to generate signals: {e}")

    return valid_signals

def label_confidence(score):
    if score >= 0.66:
        return "High Confidence"
    elif score >= 0.33:
        return "Medium Confidence"
    else:
        return "Low Confidence"