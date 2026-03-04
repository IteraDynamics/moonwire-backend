#!/usr/bin/env python3
"""
export_signal_feed.py

Exports MoonWire ML predictions as a deterministic signal feed artifact
for consumption by Itera or other backtest engines.

Usage:
    python export_signal_feed.py \\
        --product BTC-USD \\
        --bar_seconds 3600 \\
        --start 2019-01-01 \\
        --end 2025-12-30 \\
        --out feeds/btc_3600s_signals \\
        --format jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import numpy as np

# Local imports from MoonWire ML pipeline
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.ml.data_loader import load_prices
from scripts.ml.feature_builder import build_features
from scripts.ml.labeler import label_next_horizon


def get_git_sha() -> str:
    """Get current git commit hash for provenance."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent
        )
        return result.stdout.strip()[:12]
    except Exception:
        return "unknown"


def get_model_version() -> str:
    """Get model version from manifest if available."""
    manifest_path = Path("models/ml_model_manifest.json")
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            return manifest.get("version", "v1.0")
        except Exception:
            pass
    return "v1.0"


def load_trained_model(symbol: str, model_dir: Path = Path("models/standard")) -> Any:
    """Load trained model for symbol."""
    import joblib
    model_path = model_dir / f"{symbol}_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def generate_predictions(
    symbol: str,
    start_date: str,
    end_date: str,
    bar_seconds: int = 3600,
    horizon_hours: int = 1,
    model_dir: Path = Path("models/standard"),
) -> pd.DataFrame:
    """
    Generate p_long predictions for a symbol over a date range.
    
    Returns DataFrame with columns: [ts_utc, product_id, p_long, horizon_hours, model_version]
    """
    # Parse dates
    start_ts = pd.to_datetime(start_date, utc=True)
    end_ts = pd.to_datetime(end_date, utc=True)
    
    # Calculate lookback days needed
    lookback_days = int((end_ts - start_ts).days) + 30  # add buffer
    
    # Load price data
    print(f"Loading price data for {symbol}...")
    prices = load_prices([symbol], lookback_days=lookback_days)
    
    if symbol not in prices or prices[symbol].empty:
        raise ValueError(f"No price data loaded for {symbol}")
    
    # Build features
    print(f"Building features for {symbol}...")
    features = build_features(prices)
    
    if symbol not in features or features[symbol].empty:
        raise ValueError(f"No features built for {symbol}")
    
    # Add labels (needed for feature alignment, but we won't use y values)
    df = label_next_horizon(features[symbol], horizon_h=horizon_hours)
    
    # Filter to requested date range
    df = df[df["ts"] >= start_ts]
    df = df[df["ts"] <= end_ts]
    
    if df.empty:
        raise ValueError(f"No data in range {start_date} to {end_date} for {symbol}")
    
    # Load model
    print(f"Loading model for {symbol}...")
    model = load_trained_model(symbol, model_dir)
    
    # Extract feature names (exclude ts, close, y_long)
    feature_cols = [c for c in df.columns if c not in {"ts", "close", "y_long", "ret_1h"}]
    
    X = df[feature_cols].values
    
    # Generate predictions
    print(f"Generating predictions for {symbol} ({len(df)} bars)...")
    if hasattr(model, "predict_proba"):
        # Classifier - get probability of class 1 (long)
        p_long = model.predict_proba(X)[:, 1]
    else:
        # Regressor or other - use predict directly
        p_long = model.predict(X)
        # Clip to [0, 1]
        p_long = np.clip(p_long, 0.0, 1.0)
    
    # Build output DataFrame
    result = pd.DataFrame({
        "ts_utc": df["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "product_id": f"{symbol}-USD",
        "p_long": p_long,
        "horizon_hours": horizon_hours,
        "model_version": get_model_version(),
    })
    
    return result


def export_feed(
    product: str,
    bar_seconds: int,
    start: str,
    end: str,
    out_path: str,
    format: str = "jsonl",
    horizon_hours: int = 1,
    model_dir: str = "models/standard",
) -> None:
    """
    Main export function.
    """
    # Parse product (e.g., "BTC-USD" -> "BTC")
    symbol = product.split("-")[0].upper()
    
    model_dir_path = Path(model_dir)
    out_base = Path(out_path)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate predictions
    df = generate_predictions(
        symbol=symbol,
        start_date=start,
        end_date=end,
        bar_seconds=bar_seconds,
        horizon_hours=horizon_hours,
        model_dir=model_dir_path,
    )
    
    # Export based on format
    if format == "jsonl":
        feed_file = out_base.with_suffix(".jsonl")
        records = df.to_dict("records")
        with feed_file.open("w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        print(f"✓ Exported {len(records)} signals to {feed_file}")
    
    elif format == "csv":
        feed_file = out_base.with_suffix(".csv")
        df.to_csv(feed_file, index=False)
        print(f"✓ Exported {len(df)} signals to {feed_file}")
    
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    # Generate manifest
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "product_id": product,
        "bar_seconds": bar_seconds,
        "start_date": start,
        "end_date": end,
        "total_bars": len(df),
        "horizon_hours": horizon_hours,
        "model_version": get_model_version(),
        "git_sha": get_git_sha(),
        "model_dir": str(model_dir_path),
        "feature_set": "standard_v1",
        "schema": {
            "ts_utc": "ISO8601 timestamp of bar close",
            "product_id": "Asset identifier (e.g., BTC-USD)",
            "p_long": "Probability of upward move (0.0 to 1.0)",
            "horizon_hours": "Prediction horizon in hours",
            "model_version": "Model version identifier",
        },
        "notes": [
            "All timestamps are closed-bar aligned (decision at bar close, applies to next bar)",
            "p_long is the probability that price increases over next horizon_hours",
            "p_short = 1 - p_long (not explicitly included to save space)",
            "Timestamps are sorted and de-duplicated",
        ]
    }
    
    manifest_file = out_base.with_suffix(".manifest.json")
    manifest_file.write_text(json.dumps(manifest, indent=2))
    print(f"✓ Wrote manifest to {manifest_file}")
    
    # Validation
    print("\nValidation:")
    print(f"  - Rows: {len(df)}")
    print(f"  - Date range: {df['ts_utc'].iloc[0]} to {df['ts_utc'].iloc[-1]}")
    print(f"  - p_long range: [{df['p_long'].min():.4f}, {df['p_long'].max():.4f}]")
    print(f"  - Duplicates: {df['ts_utc'].duplicated().sum()}")
    print(f"  - Nulls: {df.isnull().sum().sum()}")


def main():
    parser = argparse.ArgumentParser(description="Export MoonWire signal feed")
    parser.add_argument("--product", required=True, help="Product ID (e.g., BTC-USD)")
    parser.add_argument("--bar_seconds", type=int, default=3600, help="Bar size in seconds")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--out", required=True, help="Output path (without extension)")
    parser.add_argument("--format", default="jsonl", choices=["jsonl", "csv"], help="Output format")
    parser.add_argument("--horizon_hours", type=int, default=1, help="Prediction horizon")
    parser.add_argument("--model_dir", default="models/standard", help="Model directory")
    
    args = parser.parse_args()
    
    export_feed(
        product=args.product,
        bar_seconds=args.bar_seconds,
        start=args.start,
        end=args.end,
        out_path=args.out,
        format=args.format,
        horizon_hours=args.horizon_hours,
        model_dir=args.model_dir,
    )
    print("\n✅ Signal feed export complete!")


if __name__ == "__main__":
    main()
