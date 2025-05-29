# src/composite_router.py

from fastapi import APIRouter, Query
from datetime import datetime
from src.twitter_ingestor import fetch_tweets_and_analyze
from src.news_ingestor import fetch_news_sentiment
from src.signal_utils import generate_composite_signal

router = APIRouter()

@router.get("/signals/composite")
def get_composite_signal(
    asset: str = Query("BTC"),
    tweet_method: str = Query("api", enum=["api", "snscrape"]),
    tweet_limit: int = Query(10, ge=10, le=100)
):
    timestamp = datetime.utcnow().isoformat()

    twitter_result = fetch_tweets_and_analyze(asset, method=tweet_method, limit=tweet_limit)
    twitter_score = twitter_result.get("average_sentiment", 0)

    news_result = fetch_news_sentiment(asset)
    news_score = news_result.get("average_sentiment", 0)

    signal = generate_composite_signal(
        asset=asset,
        twitter_score=twitter_score,
        news_score=news_score,
        timestamp=timestamp
    )

    return {
        "timestamp": timestamp,
        "twitter_score": twitter_score,
        "news_score": news_score,
        "composite_signal": signal
    }