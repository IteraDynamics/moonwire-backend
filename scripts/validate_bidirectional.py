#!/usr/bin/env python3
"""
validate_bidirectional.py

Run bidirectional backtest validation (long-only, short-only, combined)
and save results to reports/ folder.
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import pandas as pd

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.ml.data_loader import load_prices
from scripts.ml.feature_builder import build_features
from scripts.ml.labeler import label_next_horizon
from scripts.ml.backtest_bidirectional import run_validation_report


def load_trained_model(symbol: str, model_dir: Path = Path("models/standard")):
    """Load trained model."""
    import joblib
    model_path = model_dir / f"{symbol}_model.joblib"
    if not model_path.exists():
        print(f"⚠ Model not found: {model_path}")
        print(f"   Using demo mode (will generate random predictions)")
        return None
    return joblib.load(model_path)


def generate_predictions_for_backtest(
    symbol: str,
    lookback_days: int = 90,
    horizon_h: int = 3,
    model_dir: Path = Path("models/standard"),
) -> pd.DataFrame:
    """Generate predictions for backtest."""
    import numpy as np
    
    print(f"\n🔄 Generating predictions for {symbol}...")
    
    # Load prices
    prices = load_prices([symbol], lookback_days=lookback_days)
    if symbol not in prices or prices[symbol].empty:
        raise ValueError(f"No price data for {symbol}")
    
    # Build features
    features = build_features(prices)
    if symbol not in features or features[symbol].empty:
        raise ValueError(f"No features for {symbol}")
    
    # Add labels
    df = label_next_horizon(features[symbol], horizon_h=horizon_h)
    
    # Load model
    model = load_trained_model(symbol, model_dir)
    
    # Extract features for prediction
    feature_cols = [c for c in df.columns if c not in {"ts", "close", "y_long", "ret_1h"}]
    X = df[feature_cols].values
    
    # Generate predictions
    if model is not None and hasattr(model, "predict_proba"):
        p_long = model.predict_proba(X)[:, 1]
    else:
        # Demo mode - random predictions for testing
        print(f"⚠ Using random predictions (no model found)")
        np.random.seed(42)
        p_long = np.random.uniform(0.3, 0.7, size=len(X))
    
    # Build prediction DataFrame
    pred_df = pd.DataFrame({
        "ts": df["ts"],
        "p_long": p_long,
    })
    
    return pred_df, prices[symbol]


def main():
    """Run validation report."""
    print("=" * 60)
    print("MoonWire Bidirectional Backtest Validation")
    print("=" * 60)
    
    # Configuration
    symbol = "BTC"
    lookback_days = 90  # Small window for quick validation
    horizon_h = 3
    model_dir = Path("models/standard")
    
    # Thresholds
    long_thresh = 0.65
    short_thresh = 0.35
    debounce_hours = 10
    
    # Generate predictions
    try:
        pred_df, prices_df = generate_predictions_for_backtest(
            symbol=symbol,
            lookback_days=lookback_days,
            horizon_h=horizon_h,
            model_dir=model_dir,
        )
    except Exception as e:
        print(f"❌ Error generating predictions: {e}")
        return
    
    print(f"\n✓ Generated {len(pred_df)} predictions")
    print(f"  p_long range: [{pred_df['p_long'].min():.3f}, {pred_df['p_long'].max():.3f}]")
    
    # Run validation report
    print(f"\n🔄 Running validation report...")
    results = run_validation_report(
        pred_df=pred_df,
        prices_df=prices_df,
        long_thresh=long_thresh,
        short_thresh=short_thresh,
        debounce_hours=debounce_hours,
        horizon_hours=horizon_h,
        fee_bps=1.0,  # 1 bps = 0.01%
        slippage_bps=2.0,  # 2 bps = 0.02%
        symbol=symbol,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    for mode in ["long_only", "short_only", "combined"]:
        metrics = results[mode]["metrics"]
        print(f"\n{mode.upper()}: {symbol}")
        print(f"  Win Rate:       {metrics['win_rate']:.2%}")
        print(f"  Profit Factor:  {metrics['profit_factor']:.2f}")
        print(f"  Max Drawdown:   {metrics['max_drawdown']:.2%}")
        print(f"  CAGR:           {metrics['cagr']:.2%}")
        print(f"  Signals/Day:    {metrics['signals_per_day']:.3f}")
        print(f"  Total Trades:   {metrics['n_trades']}")
        print(f"  Avg Hold (hrs): {metrics['avg_hold_hours']:.1f}")
    
    # Print summary
    summary = results["summary"]
    print(f"\n" + "-" * 60)
    print(f"Best Win Rate:      {summary['best_win_rate']:.2%} ({summary['best_mode_by_winrate']})")
    print(f"Best Profit Factor: {summary['best_profit_factor']:.2f} ({summary['best_mode_by_pf']})")
    print(f"Note: {summary['note']}")
    
    # Save results
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = reports_dir / f"bidirectional_validation_{symbol}_{timestamp}.json"
    
    # Convert timestamp objects to strings for JSON serialization
    results_serializable = {}
    for mode in ["long_only", "short_only", "combined", "summary"]:
        if mode == "summary":
            results_serializable[mode] = results[mode]
        else:
            results_serializable[mode] = {
                "metrics": results[mode]["metrics"],
                "trades": results[mode]["trades"],  # Already serializable
                # Skip equity curve for size (can add if needed)
            }
    
    report_file.write_text(json.dumps(results_serializable, indent=2))
    print(f"\n✅ Saved report to: {report_file}")
    
    # Also save as CSV for easy viewing
    csv_file = reports_dir / f"bidirectional_validation_{symbol}_{timestamp}.csv"
    summary_data = []
    for mode in ["long_only", "short_only", "combined"]:
        row = {"mode": mode}
        row.update(results[mode]["metrics"])
        summary_data.append(row)
    
    pd.DataFrame(summary_data).to_csv(csv_file, index=False)
    print(f"✅ Saved CSV to: {csv_file}")


if __name__ == "__main__":
    main()
