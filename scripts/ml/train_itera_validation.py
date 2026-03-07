#!/usr/bin/env python3
"""
train_itera_validation.py

Full-history training and validation for MoonWire signal engine using Itera data.

This script:
1. Loads full BTC/ETH historical data (2019-2025)
2. Runs in PRICE-ONLY mode (no social features)
3. Executes walk-forward validation
4. Generates comprehensive metrics
5. Trains final production models on all data
6. Saves models and metrics for signal export

Usage:
    python train_itera_validation.py --symbols BTC ETH --output validation_results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add parent to path for local imports
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Import using relative paths
from data_loader_itera import load_prices_itera
from feature_builder import build_features
from labeler import label_next_horizon


# ============================================================================
# Configuration
# ============================================================================

# Disable social features (PRICE-ONLY MODE)
os.environ["MW_SOCIAL_ENABLED"] = "0"

# Model configuration
MODEL_TYPE = "hybrid"  # Uses ensemble of LogReg + RF + GBM
HORIZON_HOURS = 1  # 1-hour prediction horizon

# Walk-forward validation config
INITIAL_TRAIN_DAYS = 365  # Start with 1 year of training data
VALIDATION_WINDOW_DAYS = 90  # Test on 90-day windows
STEP_DAYS = 90  # Step forward 90 days each iteration


# ============================================================================
# Model Training
# ============================================================================

def train_hybrid_model(X_train: np.ndarray, y_train: np.ndarray) -> Dict:
    """
    Train hybrid ensemble model (LogReg + RF + GBM).
    Returns dict with individual models and ensemble predictor.
    """
    print("  Training Logistic Regression...")
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_train, y_train)
    
    print("  Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    print("  Training Gradient Boosting...")
    gbm = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    gbm.fit(X_train, y_train)
    
    return {
        "logreg": logreg,
        "rf": rf,
        "gbm": gbm,
    }


def predict_hybrid(models: Dict, X: np.ndarray) -> np.ndarray:
    """
    Ensemble prediction: average probabilities from all models.
    """
    p_logreg = models["logreg"].predict_proba(X)[:, 1]
    p_rf = models["rf"].predict_proba(X)[:, 1]
    p_gbm = models["gbm"].predict_proba(X)[:, 1]
    
    # Average probabilities
    p_ensemble = (p_logreg + p_rf + p_gbm) / 3.0
    
    return p_ensemble


# ============================================================================
# Feature Engineering
# ============================================================================

def extract_features_and_labels(
    df: pd.DataFrame,
    horizon_h: int = 1
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """
    Extract feature matrix, labels, feature names, and full dataframe.
    """
    # Add labels
    df_labeled = label_next_horizon(df, horizon_h=horizon_h)
    
    # Feature columns (price-only, no social)
    feature_cols = [
        "r_1h", "r_3h", "r_6h", 
        "vol_6h", "atr_14h", 
        "sma_gap", "high_vol"
    ]
    
    # Check if social_score exists (it will be 0.5 constant in price-only mode)
    if "social_score" in df_labeled.columns:
        feature_cols.append("social_score")
    
    X = df_labeled[feature_cols].values.astype(float)
    y = df_labeled["y_long"].values.astype(int)
    
    return X, y, feature_cols, df_labeled


# ============================================================================
# Walk-Forward Validation
# ============================================================================

def walk_forward_validation(
    df: pd.DataFrame,
    symbol: str,
    initial_train_days: int = 365,
    validation_window_days: int = 90,
    step_days: int = 90,
    horizon_h: int = 1,
) -> Dict:
    """
    Perform walk-forward validation on full historical data.
    
    Returns:
        dict with validation results, per-window metrics, and aggregate stats
    """
    print(f"\n{'='*60}")
    print(f"Walk-Forward Validation: {symbol}")
    print(f"{'='*60}")
    
    # Extract features and labels
    X, y, feature_cols, df_labeled = extract_features_and_labels(df, horizon_h)
    
    # Get timestamps for windowing
    timestamps = df_labeled["ts"].values
    
    # Calculate window boundaries
    data_start = pd.to_datetime(timestamps[0])
    data_end = pd.to_datetime(timestamps[-1])
    
    print(f"Data range: {data_start} to {data_end}")
    print(f"Total bars: {len(df_labeled)}")
    print(f"Features: {feature_cols}")
    print(f"Label distribution: {y.mean():.2%} positive (up moves)")
    
    # Walk-forward splits
    results = []
    fold = 0
    
    # Start validation after initial training period
    current_val_start = data_start + pd.Timedelta(days=initial_train_days)
    
    while current_val_start < data_end - pd.Timedelta(days=validation_window_days):
        fold += 1
        
        # Define windows
        train_end = current_val_start
        train_start = data_start  # Expanding window (use all prior data)
        val_start = current_val_start
        val_end = min(
            val_start + pd.Timedelta(days=validation_window_days),
            data_end
        )
        
        # Get indices for this split
        train_mask = (pd.to_datetime(timestamps) >= train_start) & (pd.to_datetime(timestamps) < train_end)
        val_mask = (pd.to_datetime(timestamps) >= val_start) & (pd.to_datetime(timestamps) < val_end)
        
        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        
        if len(train_indices) < 100 or len(val_indices) < 10:
            # Skip if insufficient data
            current_val_start += pd.Timedelta(days=step_days)
            continue
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        print(f"\nFold {fold}")
        print(f"  Train: {train_start.date()} to {train_end.date()} ({len(train_indices)} bars)")
        print(f"  Val:   {val_start.date()} to {val_end.date()} ({len(val_indices)} bars)")
        
        # Train model
        models = train_hybrid_model(X_train, y_train)
        
        # Predict on validation set
        p_val = predict_hybrid(models, X_val)
        y_pred = (p_val >= 0.5).astype(int)
        
        # Calculate metrics
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1:        {f1:.4f}")
        print(f"  Mean p(long): {p_val.mean():.4f}")
        
        # Store results
        results.append({
            "fold": fold,
            "symbol": symbol,
            "train_start": str(train_start.date()),
            "train_end": str(train_end.date()),
            "val_start": str(val_start.date()),
            "val_end": str(val_end.date()),
            "train_bars": len(train_indices),
            "val_bars": len(val_indices),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "mean_p_long": float(p_val.mean()),
            "label_dist": float(y_val.mean()),
        })
        
        # Step forward
        current_val_start += pd.Timedelta(days=step_days)
    
    # Aggregate statistics
    if results:
        acc_mean = np.mean([r["accuracy"] for r in results])
        prec_mean = np.mean([r["precision"] for r in results])
        rec_mean = np.mean([r["recall"] for r in results])
        f1_mean = np.mean([r["f1"] for r in results])
        
        print(f"\n{'='*60}")
        print(f"Aggregate Statistics ({len(results)} folds)")
        print(f"{'='*60}")
        print(f"Mean Accuracy:  {acc_mean:.4f}")
        print(f"Mean Precision: {prec_mean:.4f}")
        print(f"Mean Recall:    {rec_mean:.4f}")
        print(f"Mean F1:        {f1_mean:.4f}")
    
    return {
        "symbol": symbol,
        "folds": results,
        "aggregate": {
            "num_folds": len(results),
            "mean_accuracy": float(np.mean([r["accuracy"] for r in results])) if results else 0.0,
            "mean_precision": float(np.mean([r["precision"] for r in results])) if results else 0.0,
            "mean_recall": float(np.mean([r["recall"] for r in results])) if results else 0.0,
            "mean_f1": float(np.mean([r["f1"] for r in results])) if results else 0.0,
        }
    }


# ============================================================================
# Final Model Training (on all data)
# ============================================================================

def train_final_model(
    df: pd.DataFrame,
    symbol: str,
    output_dir: Path,
    horizon_h: int = 1,
) -> Dict:
    """
    Train final production model on ALL available data.
    """
    print(f"\n{'='*60}")
    print(f"Training Final Production Model: {symbol}")
    print(f"{'='*60}")
    
    # Extract features and labels
    X, y, feature_cols, df_labeled = extract_features_and_labels(df, horizon_h)
    
    print(f"Training samples: {len(X)}")
    print(f"Features: {feature_cols}")
    print(f"Label distribution: {y.mean():.2%} positive")
    
    # Train hybrid model on ALL data
    models = train_hybrid_model(X, y)
    
    # Save model
    model_path = output_dir / f"{symbol}_model.joblib"
    joblib.dump(models, model_path)
    print(f"✓ Saved model: {model_path}")
    
    # Calculate in-sample metrics (just for reference, not for evaluation)
    p_train = predict_hybrid(models, X)
    y_pred = (p_train >= 0.5).astype(int)
    
    acc = accuracy_score(y, y_pred)
    
    return {
        "symbol": symbol,
        "model_path": str(model_path),
        "training_samples": len(X),
        "features": feature_cols,
        "label_distribution": float(y.mean()),
        "in_sample_accuracy": float(acc),
    }


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MoonWire Itera Validation - Full-History Training & Validation"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC", "ETH"],
        help="Symbols to process (default: BTC ETH)"
    )
    parser.add_argument(
        "--output",
        default="validation_results",
        help="Output directory for results (default: validation_results)"
    )
    parser.add_argument(
        "--start-date",
        help="Optional start date filter (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        help="Optional end date filter (YYYY-MM-DD)"
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("MoonWire Signal Engine - Itera Full-History Validation")
    print("="*60)
    print(f"Symbols: {args.symbols}")
    print(f"Output: {output_dir}")
    print(f"Mode: PRICE-ONLY (MW_SOCIAL_ENABLED=0)")
    print("="*60 + "\n")
    
    # Step 1: Load data
    print("Step 1: Loading Itera historical data...")
    prices = load_prices_itera(
        args.symbols,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Step 2: Build features
    print("\nStep 2: Building price-only features...")
    features = build_features(prices)
    
    # Step 3: Walk-forward validation for each symbol
    print("\nStep 3: Walk-forward validation...")
    validation_results = {}
    
    for symbol in args.symbols:
        if symbol not in features:
            print(f"⚠ Skipping {symbol} - no features available")
            continue
        
        val_result = walk_forward_validation(
            df=features[symbol],
            symbol=symbol,
            initial_train_days=INITIAL_TRAIN_DAYS,
            validation_window_days=VALIDATION_WINDOW_DAYS,
            step_days=STEP_DAYS,
            horizon_h=HORIZON_HOURS,
        )
        
        validation_results[symbol] = val_result
    
    # Step 4: Train final models on all data
    print("\nStep 4: Training final production models...")
    final_models = {}
    
    for symbol in args.symbols:
        if symbol not in features:
            continue
        
        model_result = train_final_model(
            df=features[symbol],
            symbol=symbol,
            output_dir=models_dir,
            horizon_h=HORIZON_HOURS,
        )
        
        final_models[symbol] = model_result
    
    # Step 5: Save results
    print("\nStep 5: Saving results...")
    
    # Validation results
    val_path = output_dir / "validation_results.json"
    with open(val_path, "w") as f:
        json.dump(validation_results, f, indent=2)
    print(f"✓ Saved: {val_path}")
    
    # Final model info
    models_path = output_dir / "final_models.json"
    with open(models_path, "w") as f:
        json.dump(final_models, f, indent=2)
    print(f"✓ Saved: {models_path}")
    
    # Training manifest
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "symbols": args.symbols,
        "mode": "price_only",
        "horizon_hours": HORIZON_HOURS,
        "model_type": MODEL_TYPE,
        "walk_forward_config": {
            "initial_train_days": INITIAL_TRAIN_DAYS,
            "validation_window_days": VALIDATION_WINDOW_DAYS,
            "step_days": STEP_DAYS,
        },
        "data_source": "itera_full_history",
        "validation_results": validation_results,
        "final_models": final_models,
    }
    
    manifest_path = output_dir / "training_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"✓ Saved: {manifest_path}")
    
    # Summary
    print("\n" + "="*60)
    print("✅ Validation Complete!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Review validation metrics in validation_results.json")
    print("  2. Use trained models in models/ for signal export")
    print("  3. Run export_signal_feed_itera.py to generate signals")
    print()


if __name__ == "__main__":
    main()
