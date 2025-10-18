# scripts/ml/tuner.py
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

# MoonWire paths (repo root assumed two levels up from this file)
ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"
LOGS_DIR = ROOT / "logs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

BACKTEST_SUMMARY_PATH = MODELS_DIR / "backtest_summary.json"
THRESHOLDS_PATH = MODELS_DIR / "signal_thresholds.json"
TRADES_LOG_PATH = LOGS_DIR / "trades.jsonl"

# Import the repo's backtest function
# Expected signature:
# run_backtest(pred_df, prices_df, conf_min, debounce_min, horizon_h, fees_bps=..., slippage_bps=...)
from scripts.ml.backtest import run_backtest  # type: ignore


@dataclass
class BTMetrics:
    n_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    signals_per_day: float = 0.0
    wins: int = 0
    losses: int = 0


# -----------------------
# Helpers (robust parsing)
# -----------------------

def _as_int(x: Any, default: int = 0) -> int:
    try:
        if isinstance(x, (list, tuple)):
            return len(x)
        return int(x)
    except Exception:
        return default


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _extract_metrics(bt: Dict[str, Any]) -> BTMetrics:
    """
    Accepts flexible shapes from run_backtest and extracts consistent metrics.
    Supports:
      - at root or under 'metrics'
      - 'trades' can be count or list
    """
    m = bt.get("metrics", {})

    n_trades = _as_int(m.get("n_trades", bt.get("trades", 0)))
    if n_trades == 0:
        # If trades are a list at root, try it
        n_trades = _as_int(bt.get("trades", 0), 0)

    win_rate = _as_float(bt.get("win_rate", m.get("win_rate", 0.0)), 0.0)
    profit_factor = _as_float(bt.get("profit_factor", m.get("profit_factor", 0.0)), 0.0)
    max_drawdown = _as_float(bt.get("max_drawdown", m.get("max_drawdown", 0.0)), 0.0)
    spd = _as_float(bt.get("signals_per_day", m.get("signals_per_day", 0.0)), 0.0)

    wins = _as_int(m.get("wins", bt.get("wins", 0)), 0)
    losses = _as_int(m.get("losses", bt.get("losses", 0)), 0)

    # If we have trades as a list with pnl, recompute wins/losses if needed
    trades_obj = bt.get("trades", None)
    if isinstance(trades_obj, (list, tuple)) and len(trades_obj) > 0:
        if wins == 0 and losses == 0:
            w = 0
            l = 0
            for t in trades_obj:
                pnl = t.get("pnl", 0.0)
                if pnl > 0:
                    w += 1
                elif pnl < 0:
                    l += 1
            wins, losses = w, l
        if n_trades == 0:
            n_trades = len(trades_obj)
        if n_trades > 0 and win_rate == 0.0 and (wins + losses) > 0:
            win_rate = wins / max(1, wins + losses)

    return BTMetrics(
        n_trades=n_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown,
        signals_per_day=spd,
        wins=wins,
        losses=losses,
    )


def _weighted_average(values: List[float], weights: List[int]) -> float:
    if not values or not weights or sum(weights) == 0:
        return 0.0
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    w[w < 0] = 0
    denom = w.sum()
    if denom <= 0:
        return 0.0
    return float((v * w).sum() / denom)


def _aggregate_metrics(per_symbol: Dict[str, BTMetrics]) -> Dict[str, float]:
    if not per_symbol:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "signals_per_day": 0.0,
        }
    n_trades = [m.n_trades for m in per_symbol.values()]
    wrs = [m.win_rate for m in per_symbol.values()]
    pfs = [m.profit_factor for m in per_symbol.values()]
    mdds = [m.max_drawdown for m in per_symbol.values()]
    spd = [m.signals_per_day for m in per_symbol.values()]

    agg = {
        "n_trades": int(sum(n_trades)),
        "win_rate": _weighted_average(wrs, n_trades),  # weight by # trades
        "profit_factor": _weighted_average(pfs, n_trades) if sum(n_trades) > 0 else float(np.mean(pfs)),
        "max_drawdown": float(np.mean(mdds)) if mdds else 0.0,
        "signals_per_day": float(np.sum(spd)),  # aggregate across symbols (sum)
    }
    return agg


def _penalty(agg: Dict[str, float], target_wr: float = 0.60, min_spd: float = 5.0, max_spd: float = 10.0) -> float:
    """Lower is better. Penalize distance from constraints, prefer higher PF and lower MDD."""
    wr = agg.get("win_rate", 0.0)
    spd = agg.get("signals_per_day", 0.0)
    pf = agg.get("profit_factor", 0.0)
    mdd = abs(agg.get("max_drawdown", 0.0))

    # Distance from constraints
    wr_gap = max(0.0, target_wr - wr)
    spd_gap = 0.0
    if spd < min_spd:
        spd_gap = (min_spd - spd)
    elif spd > max_spd:
        spd_gap = (spd - max_spd)

    # Prefer higher PF, lower MDD
    score = 0.0
    score += 10.0 * wr_gap
    score += 2.0 * spd_gap
    score += 1.0 * mdd
    score += 1.0 * max(0.0, 1.0 - pf)  # if PF<1, penalize towards 1

    return score


def _mk_params_grid() -> Iterable[Tuple[float, int, int]]:
    conf_vals = [0.55, 0.58, 0.60, 0.62]
    debounce_vals = [15, 30, 45, 60]
    horizon_vals = [1, 2]
    for c in conf_vals:
        for d in debounce_vals:
            for h in horizon_vals:
                yield (c, d, h)


def _env_or_default(key: str, default: Any) -> Any:
    v = os.getenv(key)
    if v is None:
        return default
    # Try to cast numerics if default is numeric
    try:
        if isinstance(default, int):
            return int(v)
        if isinstance(default, float):
            return float(v)
        return v
    except Exception:
        return default


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# -----------------------
# Public API
# -----------------------

def tune_thresholds(
    pred_dfs: Dict[str, pd.DataFrame],
    prices: Dict[str, pd.DataFrame],
    write_summary: bool = True,
) -> Dict[str, Any]:
    """
    Grid-search over thresholds with constraints:
      - win_rate >= 0.60
      - 5 <= signals/day <= 10 (aggregate across symbols)
    Tie-break by profit factor (higher), then max drawdown (lower).
    Writes:
      - models/backtest_summary.json
      - models/signal_thresholds.json
      - logs/trades.jsonl (best-run trades if available and MW_WRITE_BT_LOGS=1)
    Returns dict with keys: 'params', 'agg', 'per_symbol', 'grid'
    """
    fees_bps = _env_or_default("MW_FEES_BPS", 1.0)
    slippage_bps = _env_or_default("MW_SLIPPAGE_BPS", 2.0)

    tried: List[Dict[str, Any]] = []
    best_params: Dict[str, Any] | None = None
    best_agg: Dict[str, float] | None = None
    best_per_symbol: Dict[str, Dict[str, float]] | None = None
    best_pen = float("inf")
    best_trades_by_symbol: Dict[str, List[Dict[str, Any]]] = {}

    symbols = sorted(pred_dfs.keys())

    for conf_min, debounce_min, horizon_h in _mk_params_grid():
        per_symbol_metrics: Dict[str, BTMetrics] = {}
        trades_by_symbol: Dict[str, List[Dict[str, Any]]] = {}

        for sym in symbols:
            pred_df = pred_dfs[sym]
            price_df = prices.get(sym)
            if price_df is None or pred_df is None or pred_df.empty:
                per_symbol_metrics[sym] = BTMetrics()
                trades_by_symbol[sym] = []
                continue

            bt = run_backtest(
                pred_df=pred_df,
                prices_df=price_df,
                conf_min=conf_min,
                debounce_min=debounce_min,
                horizon_h=horizon_h,
                fees_bps=fees_bps,
                slippage_bps=slippage_bps,
            )

            # Collect trades if present for the best-run log
            trades_obj = bt.get("trades", [])
            if isinstance(trades_obj, list):
                trades_by_symbol[sym] = trades_obj
            else:
                trades_by_symbol[sym] = []

            per_symbol_metrics[sym] = _extract_metrics(bt)

        # Aggregate
        agg = _aggregate_metrics(per_symbol_metrics)

        # Penalty (lower is better)
        pen = _penalty(agg, target_wr=0.60, min_spd=5.0, max_spd=10.0)

        tried.append(
            {
                "conf_min": conf_min,
                "debounce_min": debounce_min,
                "horizon_h": horizon_h,
                "aggregate": agg,
                "per_symbol": {k: vars(v) for k, v in per_symbol_metrics.items()},
                "penalty": pen,
            }
        )

        # Selection rule:
        # 1) prefer feasible (win_rate >= .60 and 5<=spd<=10)
        feasible = (
            agg.get("win_rate", 0.0) >= 0.60
            and 5.0 <= agg.get("signals_per_day", 0.0) <= 10.0
        )

        def _tie_break_key(a: Dict[str, float]) -> Tuple[float, float, float]:
            # Higher PF, lower |MDD|, higher WR
            return (a.get("profit_factor", 0.0), -abs(a.get("max_drawdown", 0.0)), a.get("win_rate", 0.0))

        choose = False
        if best_params is None:
            choose = True
        else:
            if feasible and best_agg is not None:
                best_feasible = (
                    best_agg.get("win_rate", 0.0) >= 0.60
                    and 5.0 <= best_agg.get("signals_per_day", 0.0) <= 10.0
                )
                if feasible and not best_feasible:
                    choose = True
                elif feasible and best_feasible:
                    # tie-break within feasible set
                    choose = _tie_break_key(agg) > _tie_break_key(best_agg)  # type: ignore
            else:
                # choose by lower penalty if still infeasible
                choose = pen < best_pen

        if choose:
            best_params = {
                "conf_min": conf_min,
                "debounce_min": debounce_min,
                "horizon_h": horizon_h,
            }
            best_agg = agg
            best_per_symbol = {k: vars(v) for k, v in per_symbol_metrics.items()}
            best_pen = pen
            best_trades_by_symbol = trades_by_symbol

    # Safety fallback if nothing was set (empty inputs)
    if best_params is None:
        best_params = {"conf_min": 0.55, "debounce_min": 15, "horizon_h": 1}
        best_agg = {"n_trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0, "signals_per_day": 0.0}
        best_per_symbol = {sym: vars(BTMetrics()) for sym in symbols}

    # Compose final result
    result = {
        "params": best_params,
        "agg": best_agg,
        "per_symbol": best_per_symbol,
        "grid": tried,  # optional introspection
    }

    # Write summaries
    if write_summary:
        _write_json(BACKTEST_SUMMARY_PATH, result)

    # Always write thresholds.json to satisfy tests & CI expectations
    _write_json(
        THRESHOLDS_PATH,
        {
            "conf_min": best_params["conf_min"],
            "debounce_min": best_params["debounce_min"],
            "horizon_h": best_params["horizon_h"],
            "source": "tuner",
            "status": "finalized",
        },
    )

    # Optionally write trades.jsonl for the best run
    if os.getenv("MW_WRITE_BT_LOGS", "1") == "1":
        try:
            with open(TRADES_LOG_PATH, "w", encoding="utf-8") as f:
                # Flatten trades per symbol; if none, write a helpful note
                total_written = 0
                for sym, trades in best_trades_by_symbol.items():
                    if isinstance(trades, list) and trades:
                        for t in trades:
                            rec = dict(t)
                            rec["symbol"] = sym
                            f.write(json.dumps(rec) + "\n")
                            total_written += 1
                if total_written == 0:
                    f.write(json.dumps({"note": "No trades met thresholds in best configuration."}) + "\n")
        except Exception:
            # Never break CI on logging
            pass

    return result