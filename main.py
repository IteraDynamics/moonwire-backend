# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.twitter_router import router as twitter_router
from src.news_router import router as news_router
from src.composite_router import router as composite_router
from src.feedback_router import router as feedback_router  # ✅ Feedback route added

app = FastAPI()

# CORS middleware (adjust allowed origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all API routes
app.include_router(twitter_router)
app.include_router(news_router)
app.include_router(composite_router)
app.include_router(feedback_router)  # ✅ Register feedback endpoint