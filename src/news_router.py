from fastapi import APIRouter
from src.sentiment_news import fetch_news_sentiment_scores
from src.signal_log import log_signal

router = APIRouter()

@router.get("/sentiment/news")
def get_news_sentiment_scores():
    scores = fetch_news_sentiment_scores()

    # Each score is a float, not a dict
    for asset, score in scores.items():
        log_signal(
            asset=asset,
            source="news",
            score=score,
            fallback_type="mock",
            confidence=None,
            price_at_score=None
        )

    return {"sentiment_scores": scores}