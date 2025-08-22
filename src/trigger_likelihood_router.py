# src/trigger_likelihood_router.py
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from src.ml.infer import infer_score, model_metadata

router = APIRouter()


class ScoreBody(BaseModel):
    origin: Optional[str] = None
    timestamp: Optional[str] = None
    features: Optional[Dict[str, Any]] = None


@router.post("/trigger-likelihood/score")
async def score_endpoint(
    body: ScoreBody,
    request: Request,
    explain: bool = Query(False, description="Return top-N feature contributions"),
    top_n: int = Query(5, ge=1, le=20),
):
    try:
        payload = body.dict(exclude_none=True)
        res = infer_score(payload, explain=explain, top_n=top_n)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trigger-likelihood/metadata")
async def metadata_endpoint():
    meta = model_metadata()
    if not meta:
        # Demo-friendly shape if artifacts missing
        return {
            "demo": True,
            "metrics": {"roc_auc_va": 0.99},
            "feature_order": [
                "count_1h", "count_6h", "count_24h", "count_72h",
                "burst_z", "regime_calm", "regime_normal", "regime_turbulent",
                "precision_7d", "recall_7d", "leadership_max_r",
            ],
            "feature_coverage_summary": {},
            "top_features": [{"feature": "burst_z", "coef": 0.1}],
        }
    return meta
