# src/composite_router.py

from fastapi import APIRouter, Query
from src.twitter_ingestor import fetch_tweets_and_analyze
from src.sentiment_news import fetch_news_sentiment_scores
from src.signal_utils import generate_composite_signal
from src.signal_log import log_signal
from src.price_fetcher import get_price_usd

router = APIRouter()

@router.get("/signals/composite")
def get_composite_signal(
    asset: str = Query("BTC"),
    twitter_limit: int = Query(10, ge=10, le=100)
):
    twitter_result = fetch_tweets_and_analyze(asset, method="snscrape", limit=twitter_limit)
    twitter_score = twitter_result.get("average_sentiment", 0.0)
    twitter_source = twitter_result.get("source", "mock")

    news_scores = fetch_news_sentiment_scores()
    news_score = news_scores.get(asset, 0.0)

    price = get_price_usd(asset)

    signal = generate_composite_signal(
        asset=asset,
        twitter_score=twitter_score,
        news_score=news_score,
        fallback_types={
            "twitter": twitter_source,
            "news": "mock" if asset not in news_scores else None
        },
        price=price
    )

    log_signal(
        asset=asset,
        source="composite",
        score=signal["score"],
        fallback_type="composite",
        confidence=signal.get("confidence"),
        price_at_score=price
    )

    return {"signals": [signal]}
