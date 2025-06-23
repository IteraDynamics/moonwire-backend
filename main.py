from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.ping_router import router as ping_router
from src.feedback_ingestion_router import router as feedback_ingestion_router
from src.feedback_aggregate_router import router as feedback_aggregate_router
from src.adjustment_trigger_router import router as adjustment_trigger_router
from src.high_disagreement_summary_router import router as high_disagreement_summary_router  # ✅ NEW

app = FastAPI()

# Allow CORS from any origin for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(ping_router)
app.include_router(feedback_ingestion_router)
app.include_router(feedback_aggregate_router)
app.include_router(adjustment_trigger_router)
app.include_router(high_disagreement_summary_router)  # ✅ NEW