from fastapi import APIRouter
from src.sentiment_news import fetch_news_sentiment_scores
from src.signal_log import log_signal

router = APIRouter()

@router.get("/sentiment/news")
def get_news_sentiment_scores():
    scores = fetch_news_sentiment_scores()

    # Log each asset's news sentiment score
    for asset, data in scores.items():
        log_signal(
            asset=asset,
            source="news",
            score=data.get("score"),
            fallback_type="mock",  # Update if live later
            confidence=None,       # Placeholder for future confidence logic
            price_at_score=None    # Optional: price fetch integration later
        )

    return {"sentiment_scores": scores}