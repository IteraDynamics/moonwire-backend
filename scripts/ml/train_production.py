# scripts/ml/train_production.py
"""
Train production models (no CV splits - train on all available data)
Saves models to models/standard/ or models/elite/
"""
from __future__ import annotations
import os
import pathlib
import joblib
from typing import List

from .utils import env_str, env_int, to_list
from .data_loader import load_prices
from .feature_builder import build_features
from .labeler import label_next_horizon
from .model_runner import train_model

ROOT = pathlib.Path(".").resolve()


def _feature_matrix(df):
    """Extract feature matrix and labels"""
    feature_cols = [
        "r_1h", "r_3h", "r_6h", "vol_6h", "atr_14h", 
        "sma_gap", "high_vol", "social_score"
    ]
    X = df[feature_cols].values.astype(float)
    y = df["y_long"].values.astype(int)
    return X, y, feature_cols


def train_production_models(
    symbols: List[str],
    lookback_days: int,
    model_type: str,
    output_dir: str,
    horizon_h: int = 1
):
    """
    Train production models on all available data
    
    Args:
        symbols: List of symbols to train (e.g. ['BTC', 'ETH'])
        lookback_days: How many days of historical data to use
        model_type: 'logreg', 'rf', 'gbm', or 'hybrid'
        output_dir: Where to save models (e.g. 'models/standard')
        horizon_h: Prediction horizon in hours
    """
    output_path = ROOT / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training Production Models")
    print(f"{'='*60}")
    print(f"Symbols: {symbols}")
    print(f"Lookback: {lookback_days} days")
    print(f"Model: {model_type}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading price data...")
    prices = load_prices(symbols, lookback_days=lookback_days)
    
    print("Building features...")
    feats = build_features(prices)
    
    # Train each symbol
    for sym in symbols:
        print(f"\n--- Training {sym} ---")
        
        # Label data
        df = label_next_horizon(feats[sym], horizon_h=horizon_h)
        X, y, feature_cols = _feature_matrix(df)
        
        print(f"  Training samples: {len(X)}")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Label distribution: {y.mean():.2%} positive")
        
        # Train on ALL data (no train/test split for production)
        print(f"  Training {model_type} model...")
        model = train_model(X, y, model_type=model_type)
        
        # Save model
        model_path = output_path / f"{sym}_model.joblib"
        joblib.dump(model, model_path)
        print(f"  ✓ Saved: {model_path}")
    
    print(f"\n{'='*60}")
    print(f"✓ All models trained and saved to {output_dir}")
    print(f"{'='*60}\n")


def main():
    """Entry point - reads config from environment variables"""
    
    # Read config from environment
    symbols = to_list(env_str("MW_ML_SYMBOLS", "BTC,ETH"))
    lookback_days = env_int("MW_ML_LOOKBACK_DAYS", 270)
    model_type = env_str("MW_ML_MODEL", "hybrid")
    horizon_h = env_int("MW_HORIZON_H", 1)
    
    # Determine output directory based on lookback
    # 270 days = Standard tier, 365 days = Elite tier
    if lookback_days == 270:
        output_dir = "models/standard"
    elif lookback_days == 365:
        output_dir = "models/elite"
    else:
        output_dir = "models/current"
    
    train_production_models(
        symbols=symbols,
        lookback_days=lookback_days,
        model_type=model_type,
        output_dir=output_dir,
        horizon_h=horizon_h
    )


if __name__ == "__main__":
    main()