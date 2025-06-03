# src/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.twitter_router import router as twitter_router
from src.news_router import router as news_router
from src.composite_router import router as composite_router
from src.feedback_router import router as feedback_router
from src.health_router import router as health_router

app = FastAPI()

# CORS config — updated to explicitly handle all origins and methods
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or explicitly ["https://<your-vercel-app>.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(twitter_router)
app.include_router(news_router)
app.include_router(composite_router)
app.include_router(feedback_router)
app.include_router(health_router)

# HEAD support for uptime checks
@app.head("/ping", include_in_schema=False)
async def ping_head():
    return {"status": "ok"}