# scripts/ml/tuner.py
from __future__ import annotations
import itertools
from typing import Dict
import pandas as pd
from .backtest import run_backtest
from .utils import save_json

CONF_GRID = [0.55, 0.58, 0.60, 0.62]
DEBOUNCE_GRID = [15, 30, 45, 60]
HORIZON_GRID = [1, 2]

def _score(m):
    # Higher is better: prioritize profit factor, then (−maxDD), then win rate, then more trades
    return (m["profit_factor"], -m["max_drawdown"], m["win_rate"], m["trades"])

def tune_thresholds(pred_dfs: Dict[str, pd.DataFrame], prices: Dict[str, pd.DataFrame]) -> Dict:
    best = None
    best_params = None
    for conf, deb, hor in itertools.product(CONF_GRID, DEBOUNCE_GRID, HORIZON_GRID):
        agg = {"trades":0, "win_rate":0.0, "profit_factor":0.0, "max_drawdown":0.0, "signals_per_day":0.0}
        per_symbol = {}
        for sym, pdf in pred_dfs.items():
            m = run_backtest(pdf, prices[sym], conf, deb, hor)
            per_symbol[sym] = m
            # aggregate simple averages (except trades is sum)
            agg["trades"] += m["trades"]
            for k in ("win_rate","profit_factor","max_drawdown","signals_per_day"):
                agg[k] += m[k]
        n = max(1, len(pred_dfs))
        for k in ("win_rate","profit_factor","max_drawdown","signals_per_day"):
            agg[k] /= n

        # constraints
        if agg["win_rate"] < 0.60: 
            continue
        if not (5.0 <= agg["signals_per_day"] <= 10.0):
            continue

        candidate = {"params":{"conf_min":conf,"debounce_min":deb,"horizon_h":hor}, "agg":agg, "per_symbol":per_symbol}
        if (best is None) or (_score(agg) > _score(best["agg"])):
            best = candidate
            best_params = candidate["params"]

    if best_params is None:
        # fallback sane defaults
        best_params = {"conf_min":0.60, "debounce_min":30, "horizon_h":1}
        best = {"params": best_params, "agg": {"win_rate":0.5,"profit_factor":1.0,"max_drawdown":-0.1,"signals_per_day":5.0}, "per_symbol":{}}

    save_json("models/signal_thresholds.json", best_params)
    return best