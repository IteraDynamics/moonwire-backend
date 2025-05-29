# src/composite_router.py

from fastapi import APIRouter, Query
from datetime import datetime
from src.twitter_ingestor import fetch_tweets_and_analyze
from src.sentiment_news import fetch_news_sentiment_scores
from src.signal_log import log_signal
from src.signal_utils import generate_composite_signal

router = APIRouter()

@router.get("/signals/composite")
def get_composite_signal(
    asset: str = Query("BTC"),
    twitter_method: str = Query("snscrape", enum=["snscrape", "api"]),
    limit: int = Query(10, ge=10, le=100)
):
    # Fetch Twitter sentiment
    twitter_result = fetch_tweets_and_analyze(asset, method=twitter_method, limit=limit)
    twitter_score = twitter_result.get("average_sentiment", 0.0)
    twitter_fallback = twitter_result.get("source", "mock")

    # Fetch News sentiment
    news_scores = fetch_news_sentiment_scores()
    news_score = news_scores.get(asset, 0.0)
    news_fallback = "mock" if news_score == 0.0 else "live"

    # Composite signal generation
    timestamp = datetime.utcnow().isoformat()
    composite_signal = generate_composite_signal(
        asset=asset,
        twitter_score=twitter_score,
        news_score=news_score,
        timestamp=timestamp
    )

    # Add fallback sources to signal payload
    composite_signal["fallback_type"] = {
        "twitter": twitter_fallback,
        "news": news_fallback
    }

    # Log composite signal
    log_signal(**composite_signal)

    return {"signal": composite_signal}