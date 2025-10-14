from fastapi import APIRouter
from datetime import datetime
from src.sentiment_news import fetch_news_sentiment_scores
from src.signal_log import log_signal
from src.price_fetcher import bulk_price_fetch
from src.signal_composer import generate_signal

router = APIRouter()

@router.get("/sentiment/news")
def get_news_sentiment_scores():
    scores = fetch_news_sentiment_scores()
    asset_list = list(scores.keys())
    price_map = bulk_price_fetch(asset_list)

    formatted_scores = []

    for asset, score in scores.items():
        price = price_map.get(asset)

        # Generate full signal using standard composer
        signal = generate_signal(
            asset=asset,
            sentiment_score=score,
            fallback_type="news-mock",
            top_drivers=["news sentiment"]
        )

        # Log it (full-schema)
        log_signal(**signal)

        # Extract frontend fields
        formatted_scores.append({
            "asset": signal["asset"],
            "sentiment_score": signal["score"],
            "confidence": signal["confidence"],
            "timestamp": signal["timestamp"]
        })

    return {"sentiment_scores": formatted_scores}
