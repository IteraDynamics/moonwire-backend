from fastapi import APIRouter, Query
from src.cache_instance import cache
from src.trend_deltas import load_signal_history, compute_trend_deltas  # ✅ NEW

router = APIRouter()

def get_movement_label(change: float) -> str:
    if change >= 10:
        return "Exploding"
    elif change >= 5:
        return "Surging"
    elif change >= 2:
        return "Moving"
    elif change > -2:
        return "Stable"
    elif change >= -5:
        return "Falling"
    else:
        return "Crashing"

@router.get("/leaderboard")
def leaderboard(
    sort_by: str = Query("price_change", enum=[
        "price_change", "sentiment", "confidence_score", "timestamp", "trend_6h", "trend_24h"  # ✅ Added
    ]),
    descending: bool = Query(True)
):
    output = []

    # ✅ Load trend deltas from log (real or mock fallback)
    trend_map = compute_trend_deltas(load_signal_history())

    for key in cache.keys():
        if not key.endswith("_history"):
            history = cache.get_signal(f"{key}_history")
            if isinstance(history, list) and history:
                latest = history[-1]
                asset_trends = trend_map.get(key, {})
                output.append({
                    "asset": key,
                    "price_change": latest.get("price_change", 0),
                    "sentiment": latest.get("sentiment", 0),
                    "confidence_score": latest.get("confidence_score", 0),
                    "timestamp": latest.get("timestamp", ""),
                    "movement_label": get_movement_label(latest.get("price_change", 0)),
                    "trend_6h": asset_trends.get("trend_6h", 0),     # ✅ New
                    "trend_24h": asset_trends.get("trend_24h", 0)    # ✅ New
                })
            else:
                output.append({
                    "asset": key,
                    "price_change": 0,
                    "sentiment": 0,
                    "confidence_score": 0,
                    "timestamp": "",
                    "movement_label": "No Data",
                    "trend_6h": 0,
                    "trend_24h": 0
                })

    return sorted(output, key=lambda x: x.get(sort_by, 0) or 0, reverse=descending)
