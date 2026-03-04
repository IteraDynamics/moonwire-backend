# scripts/ml/backtest_bidirectional.py
"""
Bidirectional backtest supporting LONG, SHORT, and COMBINED strategies.
Extends the original long-only backtest with parameterized thresholds.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd


TradeMode = Literal["long_only", "short_only", "combined"]


@dataclass
class Trade:
    ts_entry: pd.Timestamp
    ts_exit: pd.Timestamp
    symbol: str
    side: str          # "long" or "short"
    entry_px: float
    exit_px: float
    pnl: float
    pnl_pct: float


def _to_utc_hourly(df: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    """Ensure ts is UTC, truncate to hour, drop duplicates, sort."""
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], utc=True)
    out[ts_col] = out[ts_col].dt.floor("h")
    out = out.drop_duplicates(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    return out


def _ensure_price_cols(px: pd.DataFrame) -> pd.DataFrame:
    """Ensure 'close' column exists."""
    cols = set([c.lower() for c in px.columns])
    if "close" not in cols and "Close" not in px.columns:
        if {"open", "high", "low"}.issubset(cols):
            px = px.copy()
            px["close"] = px[["open", "high", "low"]].mean(axis=1)
    return px


def _max_drawdown(equity: np.ndarray) -> float:
    """Return max drawdown as a **negative** percentage."""
    if equity.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    dd = (equity - peaks) / peaks
    return float(np.min(dd)) if dd.size else 0.0


def _signals_per_day(n_trades: int, ts_start: Optional[pd.Timestamp], ts_end: Optional[pd.Timestamp]) -> float:
    if n_trades <= 0 or ts_start is None or ts_end is None or ts_end <= ts_start:
        return 0.0
    days = (ts_end - ts_start).total_seconds() / 86400.0
    if days <= 0:
        return 0.0
    return float(n_trades / days)


def _compute_cagr(final_equity: float, days: float) -> float:
    """Annualized return."""
    if days <= 0 or final_equity <= 0:
        return 0.0
    years = days / 365.25
    if years <= 0:
        return 0.0
    return float((final_equity ** (1.0 / years)) - 1.0)


def run_backtest_bidirectional(
    pred_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    mode: TradeMode = "combined",
    long_thresh: float = 0.65,
    short_thresh: float = 0.35,
    debounce_hours: int = 10,
    horizon_hours: int = 3,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    symbol: str = "",
) -> Dict[str, Any]:
    """
    Bidirectional threshold backtest.
    
    Args:
        pred_df: DataFrame with columns [ts, p_long]
        prices_df: DataFrame with columns [ts, close]
        mode: "long_only", "short_only", or "combined"
        long_thresh: Enter LONG when p_long >= this (default 0.65)
        short_thresh: Enter SHORT when p_long <= this (default 0.35)
        debounce_hours: Minimum hours between trades
        horizon_hours: Exit after this many hours
        fee_bps: Trading fee in basis points
        slippage_bps: Slippage in basis points
        symbol: Asset symbol for logging
        
    Returns:
        {
            "metrics": {win_rate, profit_factor, max_drawdown, cagr, signals_per_day, n_trades, avg_hold_hours},
            "trades": [ ... ],
            "equity": [ {"ts":..., "equity":...}, ... ]
        }
    """
    if pred_df is None or prices_df is None or pred_df.empty or prices_df.empty:
        return {
            "metrics": {"win_rate": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0, "cagr": 0.0, "signals_per_day": 0.0, "n_trades": 0, "avg_hold_hours": 0.0},
            "trades": [],
            "equity": [],
        }

    # Normalize inputs
    px = _to_utc_hourly(prices_df, ts_col="ts")
    px = _ensure_price_cols(px)
    px = px[["ts", "close"]].dropna()
    
    preds = _to_utc_hourly(pred_df, ts_col="ts")
    if "p_long" not in preds.columns:
        cand = [c for c in preds.columns if c.lower() in {"prob_long", "p", "prob"}]
        if not cand:
            return {
                "metrics": {"win_rate": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0, "cagr": 0.0, "signals_per_day": 0.0, "n_trades": 0, "avg_hold_hours": 0.0},
                "trades": [],
                "equity": [],
            }
        preds = preds.rename(columns={cand[0]: "p_long"})
    preds = preds[["ts", "p_long"]].dropna()

    # Merge predictions with prices
    merged = pd.merge(preds, px, on="ts", how="inner")
    if merged.empty:
        merged = pd.merge_asof(preds.sort_values("ts"), px.sort_values("ts"), on="ts", direction="forward")
        merged = merged.dropna()
        merged = merged[["ts", "p_long", "close"]]

    if merged.empty:
        return {
            "metrics": {"win_rate": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0, "cagr": 0.0, "signals_per_day": 0.0, "n_trades": 0, "avg_hold_hours": 0.0},
            "trades": [],
            "equity": [],
        }

    merged = merged.sort_values("ts").reset_index(drop=True)
    deb_bars = max(1, debounce_hours)

    # Simulation state
    trades: List[Trade] = []
    equity_curve: List[Tuple[pd.Timestamp, float]] = []
    
    in_pos = False
    current_side = None
    enter_idx = None
    enter_px = None
    last_entry_bar = -10_000
    
    eq = 1.0

    for i in range(len(merged) - 1):
        ts_i = merged.loc[i, "ts"]
        p_long = float(merged.loc[i, "p_long"])
        px_i = float(merged.loc[i, "close"])
        equity_curve.append((ts_i, eq))

        # Check exit condition (horizon reached)
        if in_pos and enter_idx is not None:
            if i - enter_idx >= horizon_hours:
                exit_px = px_i
                
                # Calculate P&L based on side
                if current_side == "long":
                    gross = (exit_px / enter_px) - 1.0
                else:  # short
                    gross = (enter_px / exit_px) - 1.0
                
                fees = (fee_bps + slippage_bps) / 10000.0
                net = gross - 2 * fees
                eq *= (1.0 + net)
                
                trades.append(
                    Trade(
                        ts_entry=merged.loc[enter_idx, "ts"],
                        ts_exit=ts_i,
                        symbol=symbol or "UNK",
                        side=current_side,
                        entry_px=float(enter_px),
                        exit_px=float(exit_px),
                        pnl=float(net),
                        pnl_pct=float(net),
                    )
                )
                
                in_pos = False
                current_side = None
                enter_idx = None
                enter_px = None
                last_entry_bar = i

        # Entry logic based on mode
        if not in_pos and (i - last_entry_bar) >= deb_bars:
            should_enter_long = (mode in ["long_only", "combined"]) and (p_long >= long_thresh)
            should_enter_short = (mode in ["short_only", "combined"]) and (p_long <= short_thresh)
            
            if should_enter_long or should_enter_short:
                j = i + 1
                if j < len(merged):
                    enter_idx = j
                    enter_px = float(merged.loc[j, "close"])
                    current_side = "long" if should_enter_long else "short"
                    in_pos = True
                    last_entry_bar = i

    # Close any open position at end
    if in_pos and enter_idx is not None:
        ts_last = merged.loc[len(merged) - 1, "ts"]
        exit_px = float(merged.loc[len(merged) - 1, "close"])
        
        if current_side == "long":
            gross = (exit_px / enter_px) - 1.0
        else:
            gross = (enter_px / exit_px) - 1.0
        
        fees = (fee_bps + slippage_bps) / 10000.0
        net = gross - 2 * fees
        eq *= (1.0 + net)
        
        trades.append(
            Trade(
                ts_entry=merged.loc[enter_idx, "ts"],
                ts_exit=ts_last,
                symbol=symbol or "UNK",
                side=current_side,
                entry_px=float(enter_px),
                exit_px=float(exit_px),
                pnl=float(net),
                pnl_pct=float(net),
            )
        )

    # Append final equity
    if not merged.empty:
        equity_curve.append((merged.loc[len(merged) - 1, "ts"], eq))

    # Compute metrics
    if not trades:
        return {
            "metrics": {"win_rate": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0, "cagr": 0.0, "signals_per_day": 0.0, "n_trades": 0, "avg_hold_hours": 0.0},
            "trades": [],
            "equity": equity_curve,
        }

    n_trades = len(trades)
    winners = [t for t in trades if t.pnl > 0]
    losers = [t for t in trades if t.pnl <= 0]
    win_rate = len(winners) / n_trades if n_trades > 0 else 0.0

    total_gain = sum(t.pnl for t in winners)
    total_loss = abs(sum(t.pnl for t in losers))
    profit_factor = (total_gain / total_loss) if total_loss > 0 else 0.0

    eq_array = np.array([e[1] for e in equity_curve])
    max_dd = _max_drawdown(eq_array)

    ts_start = merged.loc[0, "ts"]
    ts_end = merged.loc[len(merged) - 1, "ts"]
    days = (ts_end - ts_start).total_seconds() / 86400.0
    cagr = _compute_cagr(eq, days)
    spd = _signals_per_day(n_trades, ts_start, ts_end)

    hold_hours = [(t.ts_exit - t.ts_entry).total_seconds() / 3600.0 for t in trades]
    avg_hold = float(np.mean(hold_hours)) if hold_hours else 0.0

    return {
        "metrics": {
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "max_drawdown": float(max_dd),
            "cagr": float(cagr),
            "signals_per_day": float(spd),
            "n_trades": int(n_trades),
            "avg_hold_hours": float(avg_hold),
        },
        "trades": [asdict(t) for t in trades],
        "equity": [{"ts": str(ts), "equity": float(eq)} for ts, eq in equity_curve],
    }


def run_validation_report(
    pred_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    long_thresh: float = 0.65,
    short_thresh: float = 0.35,
    debounce_hours: int = 10,
    horizon_hours: int = 3,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    symbol: str = "",
) -> Dict[str, Any]:
    """
    Run all three modes and return comparison report.
    
    Returns dict with keys: long_only, short_only, combined, summary
    """
    modes = ["long_only", "short_only", "combined"]
    results = {}
    
    for mode in modes:
        results[mode] = run_backtest_bidirectional(
            pred_df=pred_df,
            prices_df=prices_df,
            mode=mode,
            long_thresh=long_thresh,
            short_thresh=short_thresh,
            debounce_hours=debounce_hours,
            horizon_hours=horizon_hours,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            symbol=symbol,
        )
    
    # Summary analysis
    summary = {
        "best_win_rate": max(results[m]["metrics"]["win_rate"] for m in modes),
        "best_profit_factor": max(results[m]["metrics"]["profit_factor"] for m in modes),
        "best_mode_by_winrate": max(modes, key=lambda m: results[m]["metrics"]["win_rate"]),
        "best_mode_by_pf": max(modes, key=lambda m: results[m]["metrics"]["profit_factor"]),
        "note": "Combined should ideally beat or match best single-side mode.",
    }
    
    results["summary"] = summary
    return results
