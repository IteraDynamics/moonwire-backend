from fastapi import APIRouter, Query
from datetime import datetime
from src.twitter_ingestor import fetch_tweets_and_analyze
from src.sentiment_news import fetch_news_sentiment_scores
from src.signal_utils import generate_composite_signal, compute_trust_scores
from src.signal_log import log_signal
import requests

router = APIRouter()

@router.get("/signals/composite")
def get_composite_signal(
    asset: str = Query("BTC"),
    min_trust_score: float = Query(0.3),
    force_include: bool = Query(False)
):
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

    # === Trust score logic ===
    def fetch_disagreement_prediction(payload):
        try:
            response = requests.post(
                "http://localhost:8000/internal/predict-feedback-risk",
                json=payload
            )
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return {"probability": 0.5}

    trust_insights = compute_trust_scores(fetch_disagreement_prediction)
    trust_lookup = {entry["signal_id"]: entry for entry in trust_insights}
    signal_id = signal["id"]

    trust_data = trust_lookup.get(signal_id)

    if trust_data:
        signal["trust_score"] = trust_data["trust_score"]
        signal["trust_label"] = trust_data["trust_label"]
        signal["predicted_disagreement_prob"] = trust_data["predicted_disagreement_prob"]
        signal["agreement_rate"] = trust_data["agreement_rate"]
    else:
        signal["trust_score"] = 0.5  # fallback
        signal["trust_label"] = "Unknown"

    # === Filter logic ===
    if not force_include and signal["trust_score"] < min_trust_score:
        return {"signals": []}

    log_signal(signal)

    return {"signals": [signal]}