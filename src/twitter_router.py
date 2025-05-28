from fastapi import APIRouter, Query
from src.twitter_ingestor import fetch_tweets_and_analyze
from src.signal_log import log_signal
from src.signal_composer import generate_signal
from src.price_fetcher import get_price_usd

router = APIRouter()

@router.get("/sentiment/twitter")
def get_twitter_sentiment(
    asset: str = Query("BTC"),
    method: str = Query("snscrape", enum=["snscrape", "api"]),
    limit: int = Query(10, ge=10, le=100)
):
    result = fetch_tweets_and_analyze(asset, method=method, limit=limit)

    sentiment_score = result.get("average_sentiment", 0)
    fallback_type = result.get("source_type", "mock")

    # Generate full signal
    signal = generate_signal(
        asset=asset,
        sentiment_score=sentiment_score,
        fallback_type=fallback_type,
        top_drivers=["twitter sentiment"]
    )

    # Log to signal history
    log_signal(**signal)

    # Return frontend-friendly fields
    return {
        "sentiment_scores": [
            {
                "asset": signal["asset"],
                "sentiment_score": signal["score"],
                "confidence": signal["confidence"],
                "timestamp": signal["timestamp"]
            }
        ]
    }
