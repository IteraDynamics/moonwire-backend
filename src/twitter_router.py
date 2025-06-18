from fastapi import APIRouter, Query
from src.twitter_ingestor import fetch_tweets_and_analyze
from src.signal_log import log_signal
from src.price_fetcher import get_price_usd
from datetime import datetime

router = APIRouter()

@router.get("/sentiment/twitter")
def get_twitter_sentiment(
    asset: str = Query("BTC"),
    method: str = Query("snscrape", enum=["snscrape", "api"]),
    limit: int = Query(10, ge=10, le=100)
):
    result = fetch_tweets_and_analyze(asset, method=method, limit=limit)

    score = result.get("average_sentiment")
    fallback_type = result.get("source", "mock")
    price = get_price_usd(asset)

    log_signal({
        "timestamp": datetime.utcnow().isoformat(),
        "asset": asset,
        "score": score,
        "label": (
            "Positive" if score > 0.3 else
            "Negative" if score < -0.3 else
            "Neutral"
        ),
        "confidence": None,
        "fallback_type": fallback_type,
        "source": "twitter",
        "price_at_score": price
    })

    return result
