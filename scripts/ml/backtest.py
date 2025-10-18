# scripts/ml/backtest.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict
from .utils import append_jsonl

def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min()) if len(dd) else 0.0

def run_backtest(pred_df: pd.DataFrame, prices_df: pd.DataFrame,
                 conf_min: float, debounce_min: int, horizon_h: int,
                 fees_bps: float = 1.0, slippage_bps: float = 2.0) -> Dict:
    """
    pred_df: columns ['ts','p_long'] aligned to prices bars (hourly).
    Entry at next bar open; exit after horizon_h bars at close.
    """
    df = pred_df.merge(prices_df[["ts","open","close"]], on="ts", how="inner").copy()
    df = df.sort_values("ts").reset_index(drop=True)

    # signal with debounce (minutes on hourly bars -> minimum bar spacing)
    min_bars_gap = max(1, int(round(debounce_min / 60.0)))
    last_sig_idx = -10**9

    trades = []
    for i in range(len(df) - horizon_h):
        if df.loc[i, "p_long"] >= conf_min and (i - last_sig_idx) >= min_bars_gap:
            entry_ts = df.loc[i+1, "ts"] if i + 1 < len(df) else df.loc[i, "ts"]
            entry_px = float(df.loc[i+1, "open"]) if i + 1 < len(df) else float(df.loc[i, "close"])
            exit_ts = df.loc[i + horizon_h, "ts"]
            exit_px = float(df.loc[i + horizon_h, "close"])
            gross = (exit_px - entry_px) / entry_px
            cost = (fees_bps + slippage_bps) / 10000.0
            pnl = gross - cost
            trades.append({
                "ts": str(entry_ts),
                "exit_ts": str(exit_ts),
                "side": "long",
                "entry": entry_px,
                "exit": exit_px,
                "pnl": pnl,
                "pnl_pct": pnl * 100.0,
            })
            last_sig_idx = i

    # equity curve
    equity = np.cumprod([1 + t["pnl"] for t in trades]) if trades else np.array([1.0])
    mdd = _max_drawdown(equity)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    wr = wins / max(len(trades), 1)
    pf = sum(max(0.0, t["pnl"]) for t in trades) / max(1e-9, sum(-min(0.0, t["pnl"]) for t in trades))
    signals_per_day = len(trades) / max(1.0, (len(df) / 24.0))

    # logs
    for t in trades:
        append_jsonl("logs/trades.jsonl", t)
    # equity log
    cum = 1.0
    for i, t in enumerate(trades, start=1):
        cum *= (1 + t["pnl"])
        append_jsonl("logs/equity_curve.jsonl", {"i": i, "equity": cum})

    return {
        "trades": len(trades),
        "win_rate": wr,
        "profit_factor": pf,
        "max_drawdown": mdd,
        "signals_per_day": signals_per_day,
    }