from fastapi import APIRouter, Query
from typing import Optional
from backend.schemas import CompositeSignal
from backend.data_store import composite_signal_store
from backend.logic.trust_logic import apply_trust_filtering

router = APIRouter()

@router.get("/composite")
def get_composite_signals(
    asset: str,
    twitter_score: Optional[float] = None,
    news_score: Optional[float] = None,
    filter: bool = False,
):
    signals = composite_signal_store.get_signals_for_asset(asset)

    # If disagreement prediction scores are available, annotate likely_disagreed
    if twitter_score is not None and news_score is not None:
        for sig in signals:
            if sig.confidence is not None:
                try:
                    from backend.logic.disagreement import predict_disagreement
                    likely_disagreed = predict_disagreement(
                        twitter_score, news_score, sig.confidence
                    )
                    sig.likely_disagreed = likely_disagreed
                except Exception as e:
                    print(f"WARNING: Failed to run disagreement prediction: {e}")

    # Apply trust filtering if enabled
    if filter:
        signals = apply_trust_filtering(signals)

    return {"signals": [s.dict() for s in signals]}