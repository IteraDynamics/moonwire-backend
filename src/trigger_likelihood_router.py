from __future__ import annotations
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from src.ml.infer import score as infer_score, metadata as infer_metadata

router = APIRouter(prefix="/internal", tags=["internal"])

@router.post("/trigger-likelihood/score")
def trigger_likelihood_score(body: Dict[str, Any]):
    try:
        return infer_score(body or {})
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/trigger-likelihood/metadata")
def trigger_likelihood_metadata():
    try:
        return infer_metadata()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
