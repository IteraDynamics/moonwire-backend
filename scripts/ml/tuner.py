# scripts/ml/tuner.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from scripts.ml.backtest import run_backtest


MODELS_DIR = Path("models")
LOGS_DIR = Path("logs")
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

BACKTEST_SUMMARY_PATH = MODELS_DIR / "backtest_summary.json"

# Extended grid to ensure signals fire
CONF_GRID = [0.40, 0.45, 0.50, 0.55, 0.58, 0.60]
DEBOUNCE_GRID = [5, 10, 15, 30]
HORIZON_GRID = [1, 2, 4]

# Utility ----------------------------------------------------------------------

def _extract_metrics(bt: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize metrics from backtest output."""
    m = bt.get("metrics", {})
    if not isinstance(m, dict):
        m = {}
    n_trades = int(m.get("n_trades", len(bt.get("trades", [])) if isinstance(bt.get("trades", []), list) else 0))
    win_rate = float(m.get("win_rate", 0.0))
    profit_factor = float(m.get("profit_factor", 0.0))
    max_drawdown = float(m.get("max_drawdown", 0.0))
    signals_per_day = float(m.get("signals_per_day", 0.0))
    return dict(
        n_trades=n_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown,
        signals_per_day=signals_per_day,
    )


def _score_metrics(m: Dict[str, Any]) -> float:
    """Score the metrics into a single numeric objective (higher is better)."""
    pf = m.get("profit_factor", 0.0)
    wr = m.get("win_rate", 0.0)
    dd = abs(m.get("max_drawdown", 0.0))
    spd = m.get("signals_per_day", 0.0)
    trades = m.get("n_trades", 0)
    if trades == 0 or pf <= 0:
        return -999
    return (wr * 100.0) + (pf * 10.0) - (dd * 100.0) + (spd * 2.0)


def _aggregate_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics across symbols."""
    if not metrics_list:
        return dict(win_rate=0.0, profit_factor=0.0, max_drawdown=0.0, signals_per_day=0.0, n_trades=0)
    df = pd.DataFrame(metrics_list)
    return {
        "win_rate": float(df["win_rate"].mean()),
        "profit_factor": float(df["profit_factor"].mean()),
        "max_drawdown": float(df["max_drawdown"].mean()),
        "signals_per_day": float(df["signals_per_day"].mean()),
        "n_trades": int(df["n_trades"].sum()),
    }


# Main tuner -------------------------------------------------------------------

def tune_thresholds(
    pred_dfs: Dict[str, pd.DataFrame],
    prices: Dict[str, pd.DataFrame],
    write_summary: bool = True,
) -> Dict[str, Any]:
    """
    Grid search over (conf_min, debounce_min, horizon_h).
    Returns dict with best params and aggregate metrics.
    """
    best_score = -np.inf
    best_params: Dict[str, Any] = {}
    best_result: Dict[str, Any] = {}
    all_results: List[Dict[str, Any]] = []

    for conf_min in CONF_GRID:
        for debounce in DEBOUNCE_GRID:
            for horizon in HORIZON_GRID:
                per_sym_metrics: List[Dict[str, Any]] = []
                sym_details = {}

                for sym, df in pred_dfs.items():
                    px = prices.get(sym)
                    bt = run_backtest(
                        pred_df=df,
                        prices_df=px,
                        conf_min=conf_min,
                        debounce_min=debounce,
                        horizon_h=horizon,
                        symbol=sym,
                    )
                    m = _extract_metrics(bt)
                    per_sym_metrics.append(m)
                    sym_details[sym] = m

                agg = _aggregate_metrics(per_sym_metrics)
                score = _score_metrics(agg)
                all_results.append(
                    dict(conf_min=conf_min, debounce=debounce, horizon=horizon, score=score, agg=agg)
                )

                if score > best_score:
                    best_score = score
                    best_params = dict(conf_min=conf_min, debounce_min=debounce, horizon_h=horizon)
                    best_result = dict(aggregate=agg, per_symbol=sym_details)

    # Fallback: if no trades across all grid points, force lowest threshold
    if best_result.get("aggregate", {}).get("n_trades", 0) == 0:
        conf_min = min(CONF_GRID)
        fallback_conf = dict(conf_min=conf_min, debounce_min=10, horizon_h=1)
        per_sym_metrics: List[Dict[str, Any]] = []
        sym_details = {}
        for sym, df in pred_dfs.items():
            px = prices.get(sym)
            bt = run_backtest(
                pred_df=df,
                prices_df=px,
                conf_min=conf_min,
                debounce_min=10,
                horizon_h=1,
                symbol=sym,
            )
            m = _extract_metrics(bt)
            per_sym_metrics.append(m)
            sym_details[sym] = m
        agg = _aggregate_metrics(per_sym_metrics)
        best_result = dict(aggregate=agg, per_symbol=sym_details)
        best_params = fallback_conf

    result = {
        **best_result,
        **best_params,
        "params": best_params,
        "agg": best_result.get("aggregate", {}),
    }

    if write_summary:
        with open(BACKTEST_SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    # Optional: write trades if enabled
    if os.getenv("MW_WRITE_BT_LOGS", "1") == "1":
        trades_path = LOGS_DIR / "trades.jsonl"
        with open(trades_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"note": "Trades are written by run_backtest when MW_WRITE_BT_LOGS=1"}) + "\n")

    return result


# CLI entry point --------------------------------------------------------------

if __name__ == "__main__":
    print("Running tuner demo...")
    dummy_preds = {
        "BTC": pd.DataFrame({
            "ts": pd.date_range("2024-01-01", periods=50, freq="H", tz="UTC"),
            "p_long": np.linspace(0.3, 0.7, 50)
        }),
        "ETH": pd.DataFrame({
            "ts": pd.date_range("2024-01-01", periods=50, freq="H", tz="UTC"),
            "p_long": np.linspace(0.4, 0.8, 50)
        }),
    }
    dummy_prices = {
        "BTC": pd.DataFrame({
            "ts": pd.date_range("2024-01-01", periods=50, freq="H", tz="UTC"),
            "close": np.linspace(10000, 11000, 50)
        }),
        "ETH": pd.DataFrame({
            "ts": pd.date_range("2024-01-01", periods=50, freq="H", tz="UTC"),
            "close": np.linspace(1500, 1600, 50)
        }),
    }
    res = tune_thresholds(dummy_preds, dummy_prices)
    print(json.dumps(res, indent=2))