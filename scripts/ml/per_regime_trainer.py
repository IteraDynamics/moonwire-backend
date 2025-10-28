# scripts/ml/per_regime_trainer.py
"""
Per-regime model training: Train separate models for different market conditions.

Strategy:
- Train Model A for trending markets (high directional movement)
- Train Model B for choppy markets (ranging/whipsaw)
- At inference, detect current regime and use appropriate model
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .regime_detector import detect_market_regime, get_regime_config


def split_by_regime(
    df: pd.DataFrame,
    prices_df: pd.DataFrame,
    symbol: str = ""
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split feature DataFrame into trending vs choppy regimes.

    Parameters:
    -----------
    df : pd.DataFrame
        Feature DataFrame with all features
    prices_df : pd.DataFrame
        Price data with 'close' column for regime detection
    symbol : str
        Symbol name for logging

    Returns:
    --------
    (trending_df, choppy_df) : Tuple[pd.DataFrame, pd.DataFrame]
        Two DataFrames split by regime
    """
    regime_config = get_regime_config()
    regime = detect_market_regime(prices_df, symbol=symbol, **regime_config)

    # Align regime with feature df
    regime_aligned = regime.reindex(df.index, fill_value="choppy")

    trending_mask = regime_aligned == "trending"
    choppy_mask = regime_aligned == "choppy"

    trending_df = df[trending_mask].copy()
    choppy_df = df[choppy_mask].copy()

    print(f"[per_regime] {symbol}: {len(trending_df)} trending, {len(choppy_df)} choppy samples")

    return trending_df, choppy_df


def train_per_regime_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    df_train: pd.DataFrame,
    prices_df: pd.DataFrame,
    model_type: str = "hybrid",
    symbol: str = ""
) -> Dict[str, object]:
    """
    Train separate models for each market regime.

    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    df_train : pd.DataFrame
        Training DataFrame (for regime detection)
    prices_df : pd.DataFrame
        Price data for regime detection
    model_type : str
        Type of model to train ('logreg', 'hybrid', etc.)
    symbol : str
        Symbol name for logging

    Returns:
    --------
    dict with keys:
        - 'trending': model trained on trending markets
        - 'choppy': model trained on choppy markets
        - 'fallback': model trained on all data (for edge cases)
    """
    from .model_runner import train_model

    models = {}

    # Detect regime for training data
    regime_config = get_regime_config()
    regime = detect_market_regime(prices_df, symbol=symbol, **regime_config)
    regime_aligned = regime.reindex(df_train.index, fill_value="choppy")

    # Split by regime
    trending_mask = regime_aligned == "trending"
    choppy_mask = regime_aligned == "choppy"

    X_trending = X_train[trending_mask]
    y_trending = y_train[trending_mask]
    X_choppy = X_train[choppy_mask]
    y_choppy = y_train[choppy_mask]

    print(f"[per_regime] {symbol}: Training on {len(X_trending)} trending, {len(X_choppy)} choppy samples")

    # Train trending model (if sufficient data)
    if len(X_trending) >= 10:
        try:
            models["trending"] = train_model(X_trending, y_trending, model_type=model_type)
            print(f"[per_regime] {symbol}: Trending model trained successfully")
        except Exception as e:
            print(f"[per_regime] {symbol}: Failed to train trending model: {e}")
            models["trending"] = None
    else:
        print(f"[per_regime] {symbol}: Insufficient trending data, skipping trending model")
        models["trending"] = None

    # Train choppy model (if sufficient data)
    if len(X_choppy) >= 10:
        try:
            models["choppy"] = train_model(X_choppy, y_choppy, model_type=model_type)
            print(f"[per_regime] {symbol}: Choppy model trained successfully")
        except Exception as e:
            print(f"[per_regime] {symbol}: Failed to train choppy model: {e}")
            models["choppy"] = None
    else:
        print(f"[per_regime] {symbol}: Insufficient choppy data, skipping choppy model")
        models["choppy"] = None

    # Train fallback model on all data (for safety)
    try:
        models["fallback"] = train_model(X_train, y_train, model_type=model_type)
        print(f"[per_regime] {symbol}: Fallback model trained on all {len(X_train)} samples")
    except Exception as e:
        print(f"[per_regime] {symbol}: Failed to train fallback model: {e}")
        models["fallback"] = None

    return models


def predict_per_regime(
    models: Dict[str, object],
    X_test: np.ndarray,
    df_test: pd.DataFrame,
    prices_df: pd.DataFrame,
    symbol: str = ""
) -> np.ndarray:
    """
    Make predictions using regime-specific models.

    Parameters:
    -----------
    models : dict
        Dict with 'trending', 'choppy', 'fallback' models
    X_test : np.ndarray
        Test features
    df_test : pd.DataFrame
        Test DataFrame (for regime detection)
    prices_df : pd.DataFrame
        Price data for regime detection
    symbol : str
        Symbol name for logging

    Returns:
    --------
    np.ndarray
        Predictions for test set
    """
    from .model_runner import predict_proba

    # Detect regime for test data
    regime_config = get_regime_config()
    regime = detect_market_regime(prices_df, symbol=symbol, **regime_config)
    regime_aligned = regime.reindex(df_test.index, fill_value="choppy")

    # Initialize predictions
    preds = np.zeros(len(X_test))

    # Predict with regime-specific models
    trending_mask = regime_aligned == "trending"
    choppy_mask = regime_aligned == "choppy"

    # Trending predictions
    if trending_mask.sum() > 0:
        model = models.get("trending") or models.get("fallback")
        if model is not None:
            try:
                preds[trending_mask] = predict_proba(model, X_test[trending_mask])
            except Exception as e:
                print(f"[per_regime] {symbol}: Failed to predict trending, using fallback: {e}")
                if models.get("fallback"):
                    preds[trending_mask] = predict_proba(models["fallback"], X_test[trending_mask])

    # Choppy predictions
    if choppy_mask.sum() > 0:
        model = models.get("choppy") or models.get("fallback")
        if model is not None:
            try:
                preds[choppy_mask] = predict_proba(model, X_test[choppy_mask])
            except Exception as e:
                print(f"[per_regime] {symbol}: Failed to predict choppy, using fallback: {e}")
                if models.get("fallback"):
                    preds[choppy_mask] = predict_proba(models["fallback"], X_test[choppy_mask])

    return preds


def save_per_regime_models(
    models: Dict[str, object],
    symbol: str,
    feature_list: List[str],
    models_dir: Path = Path("models/current")
):
    """
    Save per-regime models to disk.

    Creates files:
    - models/current/{symbol}_trending_model.joblib
    - models/current/{symbol}_choppy_model.joblib
    - models/current/{symbol}_fallback_model.joblib
    - models/current/{symbol}_regime_manifest.json
    """
    import joblib

    models_dir.mkdir(parents=True, exist_ok=True)

    for regime_type, model in models.items():
        if model is not None:
            model_path = models_dir / f"{symbol}_{regime_type}_model.joblib"
            joblib.dump(model, model_path)
            print(f"[per_regime] Saved {regime_type} model to {model_path}")

    # Save manifest
    manifest = {
        "symbol": symbol,
        "regime_types": list(models.keys()),
        "features": feature_list,
        "regime_config": get_regime_config(),
    }
    manifest_path = models_dir / f"{symbol}_regime_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[per_regime] Saved manifest to {manifest_path}")


def load_per_regime_models(
    symbol: str,
    models_dir: Path = Path("models/current")
) -> Optional[Dict[str, object]]:
    """
    Load per-regime models from disk.

    Returns:
    --------
    dict with 'trending', 'choppy', 'fallback' models or None if not found
    """
    import joblib

    models = {}

    for regime_type in ["trending", "choppy", "fallback"]:
        model_path = models_dir / f"{symbol}_{regime_type}_model.joblib"
        if model_path.exists():
            try:
                models[regime_type] = joblib.load(model_path)
                print(f"[per_regime] Loaded {regime_type} model from {model_path}")
            except Exception as e:
                print(f"[per_regime] Failed to load {regime_type} model: {e}")
                models[regime_type] = None
        else:
            models[regime_type] = None

    if not any(models.values()):
        return None

    return models


# Environment variable helpers
def per_regime_enabled() -> bool:
    """Check if per-regime models are enabled via MW_PER_REGIME_MODELS env var."""
    return str(os.getenv("MW_PER_REGIME_MODELS", "0")).lower() in {"1", "true", "yes"}
