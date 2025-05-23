from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.news_router import router as news_router
from src.twitter_router import router as twitter_router
from src.signal_router import router as signal_router

app = FastAPI()

# Enable CORS if needed for frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all routers
app.include_router(news_router)
app.include_router(twitter_router)
app.include_router(signal_router)

@app.get("/")
def read_root():
    return {"message": "MoonWire Signal Engine API is live."}