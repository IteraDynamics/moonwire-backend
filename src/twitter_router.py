from fastapi import APIRouter, Query
from src.twitter_ingestor import fetch_tweets_and_analyze
from src.signal_log import log_signal
from src.price_fetcher import get_price_usd
from src.signal_utils import generate_signal  # ✅ for structured signal output

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

    log_signal(
        asset=asset,
        source="twitter",
        score=score,
        fallback_type=fallback_type,
        confidence=None,
        price_at_score=price
    )

    return result

@router.get("/signals/twitter")
def get_twitter_signals(
    asset: str = Query("BTC"),
    method: str = Query("snscrape", enum=["snscrape", "api"]),
    limit: int = Query(10, ge=10, le=100)
):
    tweets_result = fetch_tweets_and_analyze(asset, method=method, limit=limit)

    score = tweets_result.get("average_sentiment", 0)
    fallback_type = tweets_result.get("source", "mock")

    signal = generate_signal(
        asset=asset,
        source="twitter",
        score=score,
        fallback_type=fallback_type,
        confidence=None,
        price_at_score=None,
        top_drivers=["twitter sentiment"]
    )

    return {"signals": [signal]}
