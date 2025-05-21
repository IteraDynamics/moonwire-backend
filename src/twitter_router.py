from fastapi import APIRouter, Query
from src.twitter_ingestor import fetch_tweets_and_analyze
from src.signal_log import log_signal

router = APIRouter()

@router.get("/sentiment/twitter")
def get_twitter_sentiment(
    asset: str = Query("BTC"),
    method: str = Query("snscrape", enum=["snscrape", "api"]),
    limit: int = Query(10, ge=10, le=100)
):
    result = fetch_tweets_and_analyze(asset, method=method, limit=limit)

    # Extract average sentiment and fallback label
    score = result.get("average_sentiment")
    fallback_type = result.get("source_type", "mock")  # Default to mock if not provided

    # Log sentiment signal
    log_signal(
        asset=asset,
        source="twitter",
        score=score,
        fallback_type=fallback_type,
        confidence=None,
        price_at_score=None
    )

    return result