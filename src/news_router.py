from fastapi import APIRouter
from src.sentiment_news import fetch_news_sentiment_scores
from src.signal_log import log_signal
from src.price_fetcher import get_price_usd

router = APIRouter()

@router.get("/sentiment/news")
def get_news_sentiment_scores():
    scores = fetch_news_sentiment_scores()

    for asset, score in scores.items():
        price = get_price_usd(asset)

        log_signal(
            asset=asset,
            source="news",
            score=score,
            fallback_type="mock",
            confidence=None,
            price_at_score=price
        )

    return {"sentiment_scores": scores}