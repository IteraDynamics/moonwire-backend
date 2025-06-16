from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.twitter_router import router as twitter_router
from src.news_router import router as news_router
from src.composite_router import router as composite_router
from src.feedback_router import router as feedback_router
from src.health_router import router as health_router
from src.admin_router import router as admin_router
from src.trend_router import router as trend_router
from src.leaderboard import router as leaderboard_router
from src.mock_loader import load_mock_cache_data
from src.feedback_analysis_router import router as feedback_analysis_router  # ✅ NEW

app = FastAPI()

# ✅ CORS middleware added before router includes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load mock cache data
load_mock_cache_data()

# ✅ Include all routers
app.include_router(twitter_router)
app.include_router(news_router)
app.include_router(composite_router)
app.include_router(feedback_router)
app.include_router(health_router)
app.include_router(admin_router)
app.include_router(trend_router)
app.include_router(leaderboard_router)
app.include_router(feedback_analysis_router)  # ✅ NEW

# ✅ HEAD route for uptime checks
@app.head("/ping", include_in_schema=False)
async def ping_head():
    return {"status": "ok"}