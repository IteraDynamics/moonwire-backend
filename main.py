import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.paths import LOGS_DIR, REVIEWER_IMPACT_LOG_PATH, REVIEWER_SCORES_PATH

# Ensure logging directory and files exist at boot
os.makedirs(LOGS_DIR, exist_ok=True)
REVIEWER_IMPACT_LOG_PATH.touch(exist_ok=True)
REVIEWER_SCORES_PATH.touch(exist_ok=True)

print("📁 Log directory initialized:")
print(f"  - Impact log path: {REVIEWER_IMPACT_LOG_PATH}")
print(f"  - Scores path:     {REVIEWER_SCORES_PATH}")

from src.twitter_router import router as twitter_router
from src.news_router import router as news_router
from src.composite_router import router as composite_router
from src.health_router import router as health_router
from src.admin_router import router as admin_router
from src.trend_router import router as trend_router
from src.leaderboard import router as leaderboard_router
from src.mock_loader import load_mock_cache_data
from src.feedback_analysis_router import router as feedback_analysis_router
from src.label_export_router import router as label_export_router
from src.internal_log_router import router as internal_log_router
from src.threshold_simulator_router import router as threshold_simulator_router
from src.feedback_volatility_router import router as feedback_volatility_router
from src.training_pair_router import router as training_pair_router
from src.model_training_router import router as model_training_router
from src.feedback_prediction_router import router as feedback_prediction_router
from src.model_signal_adjust_router import router as model_signal_adjust_router
from src.export_training_router import router as export_training_router
from src.adjustment_router import router as adjustment_router
from src.adjustment_trigger_router import router as adjustment_trigger_router
from src.internal_router import router as internal_router
from src.feedback_ingestion_router import router as feedback_ingestion_router
from src.high_disagreement_router import router as high_disagreement_router
from src.feedback_insights_router import router as feedback_insights_router
from src.internal_trusted_signals_router import router as trust_intelligence_router
from src.signal_review_router import router as signal_review_router
from src.trust_asset_pulse_router import router as trust_asset_pulse_router
from src.trust_volatility_spike_router import router as trust_volatility_spike_router
from src.reviewer_impact_scorer_router import router as reviewer_impact_scorer_router
from src.consensus_router import router as consensus_router   # <-- import it here

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load mocks
load_mock_cache_data()

# Public routers
app.include_router(twitter_router)
app.include_router(news_router)
app.include_router(composite_router)
app.include_router(health_router)
app.include_router(admin_router)
app.include_router(trend_router)
app.include_router(leaderboard_router)
app.include_router(feedback_analysis_router)
app.include_router(label_export_router)
app.include_router(internal_log_router)
app.include_router(threshold_simulator_router)
app.include_router(feedback_volatility_router)
app.include_router(training_pair_router)
app.include_router(model_training_router)
app.include_router(feedback_prediction_router)
app.include_router(model_signal_adjust_router)
app.include_router(export_training_router)
app.include_router(adjustment_router)
app.include_router(adjustment_trigger_router)
app.include_router(internal_router)
app.include_router(feedback_ingestion_router)
app.include_router(high_disagreement_router)
app.include_router(feedback_insights_router)
app.include_router(trust_intelligence_router)
app.include_router(signal_review_router)
app.include_router(trust_asset_pulse_router)
app.include_router(trust_volatility_spike_router)

# Internal routers, all under /internal
app.include_router(reviewer_impact_scorer_router, prefix="/internal")
app.include_router(consensus_router,               prefix="/internal")  # <-- mount here

@app.head("/ping", include_in_schema=False)
async def ping_head():
    return {"status": "ok"}

@app.get("/debug/routes")
def list_routes():
    return [{"path": route.path, "methods": route.methods} for route in app.routes]