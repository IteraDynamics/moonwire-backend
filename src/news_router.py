from fastapi import APIRouter
from datetime import datetime
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

    formatted_scores = []

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

        # Append structured object for frontend
        formatted_scores.append({
            "asset": asset,
            "sentiment_score": score,
            "confidence": 0.85,  # Static mock confidence
            "timestamp": datetime.utcnow().isoformat()
        })

    return {"sentiment_scores": formatted_scores}
