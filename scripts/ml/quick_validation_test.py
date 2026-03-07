#!/usr/bin/env python3
"""
quick_validation_test.py

Quick validation test using a subset of data to verify the pipeline works.
This completes in minutes instead of hours.

Usage:
    python quick_validation_test.py --symbol BTC --output quick_test_results
"""
import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Disable social features
os.environ["MW_SOCIAL_ENABLED"] = "0"

from data_loader_itera import load_prices_itera
from feature_builder import build_features
from labeler import label_next_horizon


def train_hybrid_model(X, y):
    """Quick hybrid model training."""
    print("  Training models...")
    logreg = LogisticRegression(max_iter=500, random_state=42)
    logreg.fit(X, y)
    
    rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    gbm = GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42)
    gbm.fit(X, y)
    
    return {"logreg": logreg, "rf": rf, "gbm": gbm}


def predict_hybrid(models, X):
    """Ensemble prediction."""
    p1 = models["logreg"].predict_proba(X)[:, 1]
    p2 = models["rf"].predict_proba(X)[:, 1]
    p3 = models["gbm"].predict_proba(X)[:, 1]
    return (p1 + p2 + p3) / 3.0


def main():
    parser = argparse.ArgumentParser(description="Quick validation test")
    parser.add_argument("--symbol", default="BTC", help="Symbol to test")
    parser.add_argument("--output", default="quick_test_results", help="Output directory")
    parser.add_argument("--start", default="2024-01-01", help="Start date")
    parser.add_argument("--end", default="2024-12-31", help="End date")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Quick Validation Test - {args.symbol}")
    print(f"{'='*60}")
    print(f"Date range: {args.start} to {args.end}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading data...")
    prices = load_prices_itera([args.symbol], start_date=args.start, end_date=args.end)
    
    # Build features
    print("Building features...")
    features = build_features(prices)
    df = features[args.symbol]
    
    # Add labels
    df = label_next_horizon(df, horizon_h=1)
    
    # Extract features
    feature_cols = ["r_1h", "r_3h", "r_6h", "vol_6h", "atr_14h", "sma_gap", "high_vol"]
    if "social_score" in df.columns:
        feature_cols.append("social_score")
    
    X = df[feature_cols].values.astype(float)
    y = df["y_long"].values.astype(int)
    
    print(f"Total samples: {len(X)}")
    print(f"Features: {feature_cols}")
    print(f"Label distribution: {y.mean():.2%} positive")
    
    # Simple train/test split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\nTrain: {len(X_train)} samples")
    print(f"Test:  {len(X_test)} samples")
    
    # Train model
    print("\nTraining...")
    models = train_hybrid_model(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating...")
    p_test = predict_hybrid(models, X_test)
    y_pred = (p_test >= 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\nResults:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Mean p(long): {p_test.mean():.4f}")
    
    # Save model
    model_path = output_dir / "models" / f"{args.symbol}_model.joblib"
    joblib.dump(models, model_path)
    print(f"\n✓ Saved model: {model_path}")
    
    # Save results
    results = {
        "symbol": args.symbol,
        "date_range": f"{args.start} to {args.end}",
        "total_samples": len(X),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": feature_cols,
        "metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "mean_p_long": float(p_test.mean()),
        },
        "model_path": str(model_path),
        "generated_at": datetime.now().isoformat(),
    }
    
    results_path = output_dir / "quick_test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results: {results_path}")
    
    print(f"\n{'='*60}")
    print(f"✅ Quick validation test complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
