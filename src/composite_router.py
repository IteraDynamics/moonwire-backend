from fastapi import APIRouter, Query
from datetime import datetime
from src.twitter_ingestor import fetch_tweets_and_analyze
from src.sentiment_news import fetch_news_sentiment_scores
from src.signal_utils import generate_composite_signal
from src.signal_log import log_signal

router = APIRouter()

@router.get("/signals/composite")
def get_composite_signal(asset: str = Query("BTC")):
    twitter_result = fetch_tweets_and_analyze(asset, method="snscrape", limit=10)
    news_scores = fetch_news_sentiment_scores()
    news_score = news_scores.get(asset, 0.0)

    score = round((twitter_result["average_sentiment"] * 0.6 + news_score * 0.4), 4)

    signal = generate_composite_signal(
        asset=asset,
        twitter_score=twitter_result["average_sentiment"],
        news_score=news_score,
        timestamp=datetime.utcnow().isoformat()
    )

    signal["label"] = (
        "Positive" if signal["score"] > 0.3 else
        "Negative" if signal["score"] < -0.3 else
        "Neutral"
    )
    signal["fallback_type"] = twitter_result.get("source", "mock")
    signal["source"] = "composite"
    signal["price_at_score"] = None  # Optional: populate if needed

    log_signal(signal)

    return {"signals": [signal]}
