# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.twitter_router import router as twitter_router
from src.news_router import router as news_router
from src.composite_router import router as composite_router
from src.feedback_router import router as feedback_router
import asyncio
import httpx
import logging

app = FastAPI()

# Allow cross-origin requests (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(twitter_router)
app.include_router(news_router)
app.include_router(composite_router)
app.include_router(feedback_router)

# Log setup
logging.basicConfig(level=logging.INFO)

# Background signal updater
async def fetch_signals_periodically():
    await asyncio.sleep(5)  # Let the server boot first
    while True:
        try:
            async with httpx.AsyncClient() as client:
                for asset in ["BTC", "ETH", "SOL"]:
                    twitter = await client.get(f"http://localhost:8000/signals/twitter?asset={asset}")
                    news = await client.get("http://localhost:8000/sentiment/news")
                    composite = await client.get(f"http://localhost:8000/signals/composite?asset={asset}")
                    logging.info(f"[{asset}] Twitter: {twitter.status_code}, News: {news.status_code}, Composite: {composite.status_code}")
        except Exception as e:
            logging.error(f"Background task error: {str(e)}")

        await asyncio.sleep(300)  # Wait 5 minutes before next fetch

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(fetch_signals_periodically())
