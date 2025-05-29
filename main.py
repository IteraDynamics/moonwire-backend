# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.twitter_router import router as twitter_router
from src.news_router import router as news_router
from src.composite_router import router as composite_router  # ✅ New router import

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(twitter_router)
app.include_router(news_router)
app.include_router(composite_router)  # ✅ Add composite router
