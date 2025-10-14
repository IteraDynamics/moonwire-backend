from src.sentiment_reddit import fetch_sentiment_scores
from src.cache_instance import cache
from datetime import datetime

def update_sentiment_cache():
    print(f"[{datetime.utcnow()}] Updating sentiment cache...")
    sentiment = fetch_sentiment_scores()
    for asset, score in sentiment.items():
        cache.set_signal(f"{asset}_sentiment", score)
        print(f"[Sentiment Cached] {asset}: {score}")
