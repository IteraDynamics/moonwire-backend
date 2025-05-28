# src/twitter_ingestor.py

import snscrape.modules.twitter as sntwitter
import os
import logging
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.cache_instance import cache
from src.signal_utils import generate_signal  # ✅ Make sure this import exists
from fastapi import APIRouter, Query
import ssl
import tweepy

ssl._create_default_https_context = ssl._create_unverified_context
router = APIRouter()

logging.basicConfig(level=logging.INFO)

MOCK_TWEETS = [
    "Bitcoin is doing great today!",
    "People are hyped about BTC again.",
    "Market momentum looks strong for crypto.",
    "Feeling neutral on Bitcoin right now.",
    "Could go either way, but BTC sentiment seems positive."
]

def fetch_from_snscrape(query: str, limit: int = 10):
    try:
        tweets = []
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            if i >= limit:
                break
            tweets.append(tweet.content)
        return tweets
    except Exception as e:
        logging.error({
            "event": "twitter_fetch_error",
            "method": "snscrape",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        })
        return []

def fetch_from_twitter_api(query: str, limit: int = 10):
    try:
        bearer = os.getenv("TWITTER_BEARER_TOKEN")
        client = tweepy.Client(bearer_token=bearer)
        resp = client.search_recent_tweets(
            query=query,
            tweet_fields=["created_at", "lang"],
            max_results=max(limit, 10)
        )

        logging.info({
            "event": "twitter_api_response_debug",
            "query": query,
            "raw_response": str(resp.data),
            "timestamp": datetime.utcnow().isoformat()
        })

        return [t.text for t in resp.data or []]

    except tweepy.TooManyRequests:
        logging.warning({
            "event": "twitter_fetch_error",
            "method": "api",
            "error": "RateLimitExceeded",
            "timestamp": datetime.utcnow().isoformat()
        })
        return []
    except Exception as e:
        logging.error({
            "event": "twitter_fetch_error",
            "method": "api",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        })
        return []

def score_tweets(tweets: list[str]):
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(t)["compound"] for t in tweets]
    return round(sum(scores) / len(scores), 4) if scores else 0.0

def fetch_tweets_and_analyze(asset: str, method="api", limit=10):
    query = f"{asset} OR #{asset}"  # Cashtag (${asset}) removed to avoid unsupported operator

    tweets = []
    if method == "snscrape":
        tweets = fetch_from_snscrape(query, limit)
    elif method == "api":
        tweets = fetch_from_twitter_api(query, limit)

    fallback_type = None
    if not tweets:
        logging.warning({
            "event": "twitter_fallback_used",
            "reason": "No tweets returned from either method",
            "timestamp": datetime.utcnow().isoformat()
        })
        tweets = MOCK_TWEETS
        fallback_type = "mock"

    avg_sentiment = score_tweets(tweets)
    timestamp = datetime.utcnow().isoformat()

    cache.set_signal(f"{asset}_twitter_sentiment", {
        "sentiment": avg_sentiment,
        "source": "Twitter",
        "timestamp": timestamp
    })

    signal = generate_signal(
        asset=asset,
        score=avg_sentiment,
        source="twitter",
        fallback_type=fallback_type,
        top_drivers=["twitter sentiment"],
        timestamp=timestamp
    )

    return {
        "signals": [signal]
    }

@router.get("/test-twitter")
def test_twitter(
    asset: str = Query("BTC"),
    method: str = Query("snscrape", enum=["snscrape", "api"]),
    limit: int = Query(10, ge=10, le=100)
):
    return fetch_tweets_and_analyze(asset, method=method, limit=limit)
