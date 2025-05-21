from fastapi import APIRouter
from src.sentiment_news import fetch_news_sentiment_scores
from src.signal_log import log_signal
from src.price_fetcher import bulk_price_fetch

router = APIRouter()

@router.get("/sentiment/news")
def get_news_sentiment_scores():
    scores = fetch_news_sentiment_scores()
    asset_list = list(scores.keys())

    # Fetch all prices at once
    price_map = bulk_price_fetch(asset_list)

    for asset, score in scores.items():
        price = price_map.get(asset)

        log_signal(
            asset=asset,
            source="news",
            score=score,
            fallback_type="mock",
            confidence=None,
            price_at_score=price
        )

    return {"sentiment_scores": scores}