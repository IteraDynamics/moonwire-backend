#!/usr/bin/env python3
"""
export_signal_feed_itera.py

Export MoonWire signal feeds using Itera historical data and trained models.

Usage:
    python export_signal_feed_itera.py \\
        --symbol BTC \\
        --model validation_results/models/BTC_model.joblib \\
        --start 2019-01-01 \\
        --end 2025-12-30 \\
        --output feeds/btc_signals \\
        --format jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd

# Import from same directory
from data_loader_itera import load_prices_itera
from feature_builder import build_features
from labeler import label_next_horizon


def get_git_sha() -> str:
    """Get current git commit hash for provenance."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent.parent
        )
        return result.stdout.strip()[:12]
    except Exception:
        return "unknown"


def predict_hybrid(models: Dict, X: np.ndarray) -> np.ndarray:
    """
    Ensemble prediction for hybrid models.
    Averages probabilities from LogReg + RF + GBM.
    """
    p_logreg = models["logreg"].predict_proba(X)[:, 1]
    p_rf = models["rf"].predict_proba(X)[:, 1]
    p_gbm = models["gbm"].predict_proba(X)[:, 1]
    
    return (p_logreg + p_rf + p_gbm) / 3.0


def export_signals(
    symbol: str,
    model_path: str,
    start_date: str,
    end_date: str,
    output_path: str,
    format: str = "jsonl",
    horizon_hours: int = 1,
) -> None:
    """
    Export signal feed for a symbol.
    """
    print(f"\n{'='*60}")
    print(f"Exporting Signals: {symbol}")
    print(f"{'='*60}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    models = joblib.load(model_file)
    
    # Load data
    print(f"Loading data for {symbol}...")
    prices = load_prices_itera([symbol], start_date=start_date, end_date=end_date)
    
    if symbol not in prices:
        raise ValueError(f"No data loaded for {symbol}")
    
    # Build features
    print(f"Building features...")
    features = build_features(prices)
    df = features[symbol]
    
    # Add labels (needed for feature alignment, but we use all data)
    df_labeled = label_next_horizon(df, horizon_h=horizon_hours)
    
    print(f"  Total bars: {len(df_labeled)}")
    print(f"  Date range: {df_labeled['ts'].min()} to {df_labeled['ts'].max()}")
    
    # Extract features
    feature_cols = [
        "r_1h", "r_3h", "r_6h",
        "vol_6h", "atr_14h",
        "sma_gap", "high_vol"
    ]
    
    # Add social_score if exists (will be constant 0.5 in price-only mode)
    if "social_score" in df_labeled.columns:
        feature_cols.append("social_score")
    
    X = df_labeled[feature_cols].values.astype(float)
    
    # Generate predictions
    print(f"Generating predictions...")
    p_long = predict_hybrid(models, X)
    
    # Build output DataFrame
    result_df = pd.DataFrame({
        "ts_utc": df_labeled["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "product_id": f"{symbol}-USD",
        "p_long": p_long,
        "horizon_hours": horizon_hours,
        "close": df_labeled["close"],  # Include close price for reference
    })
    
    # Validation
    print(f"\nValidation:")
    print(f"  Bars: {len(result_df)}")
    print(f"  p_long range: [{p_long.min():.4f}, {p_long.max():.4f}]")
    print(f"  p_long mean: {p_long.mean():.4f}")
    print(f"  Duplicates: {result_df['ts_utc'].duplicated().sum()}")
    print(f"  Nulls: {result_df.isnull().sum().sum()}")
    
    # Export based on format
    output_base = Path(output_path)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "jsonl":
        feed_file = output_base.with_suffix(".jsonl")
        records = result_df.to_dict("records")
        with feed_file.open("w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        print(f"\n✓ Exported {len(records)} signals to {feed_file}")
    
    elif format == "csv":
        feed_file = output_base.with_suffix(".csv")
        result_df.to_csv(feed_file, index=False)
        print(f"\n✓ Exported {len(result_df)} signals to {feed_file}")
    
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    # Generate manifest
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "product_id": f"{symbol}-USD",
        "model_path": str(model_path),
        "start_date": start_date,
        "end_date": end_date,
        "total_bars": len(result_df),
        "first_bar": result_df["ts_utc"].iloc[0],
        "last_bar": result_df["ts_utc"].iloc[-1],
        "horizon_hours": horizon_hours,
        "git_sha": get_git_sha(),
        "data_source": "itera_full_history",
        "mode": "price_only",
        "features": feature_cols,
        "statistics": {
            "p_long_min": float(p_long.min()),
            "p_long_max": float(p_long.max()),
            "p_long_mean": float(p_long.mean()),
            "p_long_std": float(p_long.std()),
            "bars_p_long_gt_0.6": int((p_long > 0.6).sum()),
            "bars_p_long_lt_0.4": int((p_long < 0.4).sum()),
        },
        "schema": {
            "ts_utc": "ISO8601 timestamp (bar close time, UTC)",
            "product_id": "Asset identifier (e.g., BTC-USD)",
            "p_long": "Probability of upward move over next horizon_hours (0.0-1.0)",
            "horizon_hours": "Prediction horizon in hours",
            "close": "Close price at bar (for reference)",
        },
        "notes": [
            "Closed-bar semantics: decision at bar t applies to bar t+1",
            "All timestamps are bar close times (end of hourly period)",
            "p_short = 1 - p_long (not included to save space)",
            "Timestamps are monotonically increasing and deduplicated",
            "NO LOOKAHEAD: Features at bar t only use data up to and including bar t",
        ],
        "alignment": {
            "prediction_timing": "t → t+1",
            "bar_alignment": "close",
            "warmup_trimmed": "Yes (initial bars with insufficient history for features)",
        }
    }
    
    manifest_file = output_base.with_suffix(".manifest.json")
    manifest_file.write_text(json.dumps(manifest, indent=2))
    print(f"✓ Wrote manifest to {manifest_file}")
    
    print(f"\n{'='*60}")
    print(f"✅ Signal export complete for {symbol}")
    print(f"{'='*60}\n")
    
    return feed_file, manifest_file


def main():
    parser = argparse.ArgumentParser(description="Export MoonWire signal feed (Itera data)")
    parser.add_argument("--symbol", required=True, help="Symbol (e.g., BTC, ETH)")
    parser.add_argument("--model", required=True, help="Path to trained model (.joblib)")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", required=True, help="Output path (without extension)")
    parser.add_argument("--format", default="jsonl", choices=["jsonl", "csv"], help="Output format")
    parser.add_argument("--horizon-hours", type=int, default=1, help="Prediction horizon")
    
    args = parser.parse_args()
    
    export_signals(
        symbol=args.symbol.upper(),
        model_path=args.model,
        start_date=args.start,
        end_date=args.end,
        output_path=args.output,
        format=args.format,
        horizon_hours=args.horizon_hours,
    )


if __name__ == "__main__":
    main()
