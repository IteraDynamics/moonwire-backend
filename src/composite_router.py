from fastapi import APIRouter, Query
from src.twitter_ingestor import fetch_tweets_and_analyze
from src.sentiment_news import fetch_news_sentiment_scores
from src.signal_utils import generate_composite_signal

router = APIRouter()

@router.get("/signals/composite")
def get_composite_signal(
    asset: str = Query("BTC"),
    twitter_method: str = Query("snscrape", enum=["snscrape", "api"]),
    twitter_limit: int = Query(10, ge=10, le=100)
):
    twitter_data = fetch_tweets_and_analyze(asset, method=twitter_method, limit=twitter_limit)
    news_scores = fetch_news_sentiment_scores()
    news_score = news_scores.get(asset, 0)

    twitter_score = twitter_data.get("average_sentiment", 0)
    timestamp = twitter_data.get("timestamp")

    signal = generate_composite_signal(
        asset=asset,
        twitter_score=twitter_score,
        news_score=news_score,
        timestamp=timestamp
    )

    return {"signals": [signal]}
