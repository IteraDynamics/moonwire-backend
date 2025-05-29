from fastapi import APIRouter, Query
from src.twitter_ingestor import fetch_tweets_and_analyze
from src.sentiment_news import fetch_news_sentiment_scores
from src.signal_utils import generate_signal
from src.price_fetcher import get_price_usd

router = APIRouter()

@router.get("/signals/composite")
def get_composite_signal(
    asset: str = Query("BTC"),
    method: str = Query("snscrape", enum=["snscrape", "api"]),
    limit: int = Query(10, ge=10, le=100)
):
    twitter_result = fetch_tweets_and_analyze(asset, method=method, limit=limit)
    twitter_score = twitter_result.get("average_sentiment", 0.0)
    twitter_source = twitter_result.get("source", "mock")
    twitter_is_mock = twitter_source == "mock"

    try:
        news_scores = fetch_news_sentiment_scores()
        news_score = news_scores.get(asset, 0.0)
        news_is_mock = False  # Assume real unless scores are zeros or empty
        if news_score == 0.0:
            news_is_mock = True
    except Exception:
        news_score = 0.0
        news_is_mock = True

    # Fallback if both are mock or missing
    if twitter_is_mock and news_is_mock:
        avg_score = 0.0
        fallback_type = "composite-mock"
        top_drivers = ["twitter", "news"]
    else:
        # Weighting: Twitter 60%, News 40%
        avg_score = round(twitter_score * 0.6 + news_score * 0.4, 4)
        fallback_type = "partial-mock" if twitter_is_mock or news_is_mock else None
        top_drivers = []
        if not twitter_is_mock:
            top_drivers.append("twitter")
        if not news_is_mock:
            top_drivers.append("news")

    try:
        price = get_price_usd(asset)
    except Exception:
        price = None

    signal = generate_signal(
        asset=asset,
        score=avg_score,
        source="composite",
        fallback_type=fallback_type,
        top_drivers=top_drivers,
        timestamp=None
    )

    return {"signals": [signal], "price": price}