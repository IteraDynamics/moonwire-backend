# src/ml/infer_asset.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import joblib

# Reuse your existing ML pipeline bits
from scripts.ml.data_loader import load_prices
from scripts.ml.feature_builder import build_features

# Where we expect the manifest/model (from train_predict.py outputs)
_MANIFEST = Path("models/ml_model_manifest.json")

def _read_manifest() -> Dict[str, Any]:
    try:
        if _MANIFEST.exists():
            return json.loads(_MANIFEST.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def _model_path_from_manifest(man: Dict[str, Any]) -> Path | None:
    # Common keys we’ve seen in your artifacts
    for k in ("model_path", "model_file", "model_joblib"):
        p = man.get(k)
        if p:
            p = str(p)
            path = Path(p)
            if not path.is_absolute():
                path = Path("models") / path
            if path.exists():
                return path
    # fallback: try a joblib in ./models
    try:
        for cand in Path("models").glob("*.joblib"):
            return cand
    except Exception:
        pass
    return None

def _feature_names(man: Dict[str, Any]) -> List[str]:
    feats = man.get("features") or man.get("feature_list") or []
    # guarantee list[str]
    return [str(f) for f in feats]

def _align_one_symbol_frame(df, need_cols: List[str]):
    """
    Given a features DF for one symbol, ensure all needed columns exist (fill 0.0),
    order them to match training, and drop rows with NA to avoid model errors.
    """
    import pandas as pd

    if df is None or len(df) == 0:
        return None
    # Ensure all columns present
    for c in need_cols:
        if c not in df.columns:
            df[c] = 0.0
    # Order columns
    df = df[need_cols + [c for c in df.columns if c not in need_cols]]
    # Last row = most recent
    df = df.sort_values("ts") if "ts" in df.columns else df
    xrow = df.tail(1).copy()
    # Keep only model columns for inference
    xrow = xrow[need_cols]
    # Drop if NA
    xrow = xrow.replace([np.inf, -np.inf], np.nan).dropna()
    if len(xrow) == 0:
        return None
    return xrow

def infer_asset_signal(asset: str) -> Dict[str, Any]:
    """
    Returns:
      {
        "direction": "long"/"short",
        "confidence": float in [0,1],
        "raw": {...}    # debugging info
      }
    On failure, returns {"error": "...", "raw": {...}}.
    """
    man = _read_manifest()
    feats = _feature_names(man)
    model_path = _model_path_from_manifest(man)

    raw = {
        "manifest_found": bool(man),
        "feature_names": feats,
        "model_path": str(model_path) if model_path else None,
        "asset": asset,
    }

    if not man or not feats or not model_path:
        return {"error": "model_unavailable", "raw": raw}

    # Load model
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raw["load_error"] = f"{type(e).__name__}"
        return {"error": "model_load_failed", "raw": raw}

    # Build features for this asset (one-symbol frame)
    try:
        prices = load_prices([asset], lookback_days=int(os.getenv("MW_ML_LOOKBACK_DAYS", "180")))
        all_feats = build_features(prices)  # dict: {symbol -> DataFrame}
        df = all_feats.get(asset)
    except Exception as e:
        raw["build_features_error"] = f"{type(e).__name__}"
        return {"error": "feature_build_failed", "raw": raw}

    xrow = _align_one_symbol_frame(df, feats)
    if xrow is None:
        raw["note"] = "no_row_for_inference"
        return {"error": "no_recent_features", "raw": raw}

    X = xrow.values.astype(float)  # shape (1, n_features)

    # Predict proba -> confidence
    try:
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X)[0, 1])
        elif hasattr(model, "predict"):
            # Calibrated or regression-ish fallback: map to [0,1]
            y = float(model.predict(X)[0])
            proba = 1.0 / (1.0 + np.exp(-y))
        else:
            return {"error": "model_interface_unknown", "raw": raw}
    except Exception as e:
        raw["predict_error"] = f"{type(e).__name__}"
        return {"error": "predict_failed", "raw": raw}

    direction = "long" if proba >= 0.5 else "short"
    return {
        "direction": direction,
        "confidence": proba,
        "raw": raw,
    }