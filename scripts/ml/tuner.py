# scripts/ml/tuner.py
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from .backtest import run_backtest


# ------------------------------- utils ----------------------------------------


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _parse_list_env(name: str, cast, default: List[Any]) -> List[Any]:
    raw = os.getenv(name, "")
    if not raw.strip():
        return default
    out: List[Any] = []
    for tok in raw.replace(" ", "").split(","):
        if tok == "":
            continue
        try:
            out.append(cast(tok))
        except Exception:
            # ignore bad tokens; keep going
            continue
    return out or default


def _round(x: float | None, ndigits: int = 4) -> float | None:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x), ndigits)


@dataclass
class BTMetrics:
    win_rate: float
    profit_factor: float
    max_drawdown: float
    signals_per_day: float
    n_trades: int
    # Optional richer payload for future use
    gross_profit: float | None = None
    gross_loss: float | None = None
    
def _extract_metrics(bt: Dict[str, Any]) -> BTMetrics:
    """
    Accepts either:
      A) {"metrics": {...}, "trades": int|list, ...}
      B) Flat: {"win_rate": .., "profit_factor": .., "max_drawdown": ..,
                "signals_per_day": .., "trades": int|list, ...}
    """
    m = bt.get("metrics")
    if isinstance(m, dict):
        # Nested metrics dict
        wr = float(m.get("win_rate", 0.0))
        pf = float(m.get("profit_factor", 0.0))
        dd = float(m.get("max_drawdown", 0.0))
        sig = float(m.get("signals_per_day", 0.0))
        # n_trades can be in metrics, or infer from top-level trades
        n_trades = m.get("n_trades", None)
        if n_trades is None:
            trades_obj = bt.get("trades", [])
            if isinstance(trades_obj, int):
                n_trades = trades_obj
            elif isinstance(trades_obj, (list, tuple)):
                n_trades = len(trades_obj)
            else:
                n_trades = 0
        n_trades = int(n_trades)
        gp = m.get("gross_profit")
        gl = m.get("gross_loss")
    else:
        # Flat dict
        wr = float(bt.get("win_rate", 0.0))
        pf = float(bt.get("profit_factor", 0.0))
        dd = float(bt.get("max_drawdown", 0.0))
        sig = float(bt.get("signals_per_day", 0.0))
        trades_obj = bt.get("trades", [])
        if isinstance(trades_obj, int):
            n_trades = trades_obj
        elif isinstance(trades_obj, (list, tuple)):
            n_trades = len(trades_obj)
        else:
            n_trades = int(bt.get("n_trades", 0))
        gp = bt.get("gross_profit")
        gl = bt.get("gross_loss")

    return BTMetrics(
        win_rate=wr,
        profit_factor=pf,
        max_drawdown=dd,
        signals_per_day=sig,
        n_trades=int(n_trades),
        gross_profit=float(gp) if gp is not None and gp == gp else None,  # keep None if NaN
        gross_loss=float(gl) if gl is not None and gl == gl else None,
    )



# ----------------------------- aggregation ------------------------------------


def _aggregate_from_per_symbol(ps: List[BTMetrics]) -> Dict[str, float]:
    """
    Aggregate across symbols:
      - win_rate: trade-weighted average
      - profit_factor: if we have gross P/L, compute exactly; else a robust approximation
      - max_drawdown: worst (most negative)
      - signals/day: average of per-symbol signals/day
    """
    if not ps:
        return {
            "win_rate": 0.50,
            "profit_factor": 1.0,
            "max_drawdown": -0.10,
            "signals_per_day": 5.0,
        }

    total_trades = sum(max(0, p.n_trades) for p in ps)
    total_trades = max(1, total_trades)

    # Win rate (trade-weighted)
    wins_weighted = sum(p.win_rate * max(1, p.n_trades) for p in ps)
    agg_wr = wins_weighted / total_trades

    # Profit factor
    have_gp = all(p.gross_profit is not None for p in ps)
    have_gl = all(p.gross_loss is not None for p in ps)
    if have_gp and have_gl:
        gross_profit = sum((p.gross_profit or 0.0) for p in ps)
        gross_loss = sum((p.gross_loss or 0.0) for p in ps)
        if gross_loss < 0:
            gross_loss = -gross_loss
        agg_pf = (gross_profit / gross_loss) if gross_loss > 0 else (1.0 if gross_profit == 0 else 9.99)
    else:
        # Approximate PF with a monotonic surrogate when GP/GL are unavailable
        # Treat PF>=1 as “excess profit”; PF<1 as “excess loss”
        excess_profit = 0.0
        excess_loss = 0.0
        for p in ps:
            pf = max(0.0, float(p.profit_factor))
            n = max(1, p.n_trades)
            if pf >= 1.0:
                excess_profit += n * (pf - 1.0)
            else:
                excess_loss += n * (1.0 - pf)
        agg_pf = (excess_profit / max(1e-9, excess_loss)) if excess_loss > 0 else (1.0 if excess_profit == 0 else 9.99)

    # Worst drawdown (most negative)
    agg_dd = min((p.max_drawdown for p in ps), default=-0.10)

    # Average signals/day
    agg_sig = float(np.mean([p.signals_per_day for p in ps])) if ps else 0.0

    return {
        "win_rate": _round(agg_wr, 4),
        "profit_factor": _round(agg_pf, 2),
        "max_drawdown": _round(agg_dd, 4),
        "signals_per_day": _round(agg_sig, 2),
    }


# ------------------------------- search ---------------------------------------


def _score_candidate(agg: Dict[str, float]) -> Tuple[float, float, float]:
    """
    Higher is better for ranking:
      1) profit_factor
      2) win_rate
     And *lower* (less negative) max_drawdown is better → we invert sign for tie-breaking.
    """
    pf = float(agg.get("profit_factor", 0.0) or 0.0)
    wr = float(agg.get("win_rate", 0.0) or 0.0)
    mdd = float(agg.get("max_drawdown", 0.0) or 0.0)
    return (pf, wr, -mdd)


def _within_objectives(agg: Dict[str, float], wr_min: float, sig_min: float, sig_max: float) -> bool:
    wr = float(agg.get("win_rate", 0.0) or 0.0)
    sig = float(agg.get("signals_per_day", 0.0) or 0.0)
    return (wr >= wr_min) and (sig_min <= sig <= sig_max)


# ------------------------------- public API -----------------------------------


def tune_thresholds(pred_dfs: Dict[str, pd.DataFrame], prices: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Grid-search tuning over confidence and debounce/horizon.
    Returns a dict of chosen parameters and writes artifacts.
    """

    # Search grids (with env overrides)
    conf_grid = _parse_list_env("MW_CONF_GRID", float, [0.55, 0.58, 0.60, 0.62])
    debounce_grid = _parse_list_env("MW_DEBOUNCE_GRID_MIN", int, [15, 30, 45, 60])
    horizon_grid = _parse_list_env("MW_HORIZON_GRID_H", int, [1, 2])

    # Objectives (with env overrides)
    wr_target = float(os.getenv("MW_TUNE_WINRATE_MIN", "0.60"))
    sig_min = float(os.getenv("MW_TUNE_SIGNALS_PER_DAY_MIN", "5"))
    sig_max = float(os.getenv("MW_TUNE_SIGNALS_PER_DAY_MAX", "10"))

    # Trading frictions (used by backtest)
    fees_bps = float(os.getenv("MW_FEES_BPS", "1.0"))
    slippage_bps = float(os.getenv("MW_SLIPPAGE_BPS", "2.0"))

    # Common outputs
    models_dir = Path("models")
    logs_dir = Path("logs")
    artifacts_dir = Path("artifacts")
    _ensure_dir(models_dir / "_.tmp")
    _ensure_dir(logs_dir / "_.tmp")
    _ensure_dir(artifacts_dir / "_.tmp")

    best_ok: Tuple[Tuple[float, int, int], Dict[str, Any]] | None = None
    best_any: Tuple[Tuple[float, int, int], Dict[str, Any]] | None = None

    # Evaluate grid
    for conf_min in conf_grid:
        for debounce_min in debounce_grid:
            for horizon_h in horizon_grid:
                per_symbol: Dict[str, Dict[str, Any]] = {}
                per_symbol_metrics: List[BTMetrics] = []

                # Run per-symbol backtests
                for sym, pred_df in pred_dfs.items():
                    if sym not in prices:
                        # Skip if we don't have matching prices
                        continue
                    try:
                        bt = run_backtest(
                            pred_df=pred_df,
                            prices_df=prices[sym],
                            conf_min=float(conf_min),
                            debounce_min=int(debounce_min),
                            horizon_h=int(horizon_h),
                            fees_bps=fees_bps,
                            slippage_bps=slippage_bps,
                        )
                    except Exception as e:
                        # Be robust in CI
                        bt = {"metrics": {"win_rate": 0.0, "profit_factor": 0.0, "max_drawdown": -0.15, "signals_per_day": 0.0, "n_trades": 0}}

                    m = _extract_metrics(bt)
                    per_symbol_metrics.append(m)
                    per_symbol[sym] = {
                        "win_rate": _round(m.win_rate, 4),
                        "profit_factor": _round(m.profit_factor, 2),
                        "max_drawdown": _round(m.max_drawdown, 4),
                        "signals_per_day": _round(m.signals_per_day, 2),
                        "n_trades": int(m.n_trades),
                    }

                # Aggregate
                agg = _aggregate_from_per_symbol(per_symbol_metrics)
                summary = {"aggregate": agg, "per_symbol": per_symbol}

                # Track bests
                if _within_objectives(agg, wr_target, sig_min, sig_max):
                    score = _score_candidate(agg)
                    if (best_ok is None) or (score > _score_candidate(best_ok[1]["aggregate"])):
                        best_ok = ((conf_min, debounce_min, horizon_h), summary)

                # Even if not within objectives, keep the best overall as a fallback
                score_any = _score_candidate(agg)
                if (best_any is None) or (score_any > _score_candidate(best_any[1]["aggregate"])):
                    best_any = ((conf_min, debounce_min, horizon_h), summary)

    # Decide final
    chosen_params: Tuple[float, int, int]
    chosen_summary: Dict[str, Any]
    if best_ok is not None:
        chosen_params, chosen_summary = best_ok
    else:
        # Fallback to best observed if nothing hit constraints
        chosen_params, chosen_summary = best_any  # type: ignore

    conf_min, debounce_min, horizon_h = chosen_params

    # Write artifacts
    backtest_summary_path = models_dir / "backtest_summary.json"
    _ensure_dir(backtest_summary_path)
    with backtest_summary_path.open("w", encoding="utf-8") as f:
        json.dump(chosen_summary, f, indent=2)

    thresholds = {
        "conf_min": float(conf_min),
        "debounce_min": int(debounce_min),
        "horizon_h": int(horizon_h),
        "objective": {
            "win_rate_min": wr_target,
            "signals_per_day_min": sig_min,
            "signals_per_day_max": sig_max,
        },
        "aggregate": chosen_summary.get("aggregate", {}),
    }
    thresholds_path = models_dir / "signal_thresholds.json"
    _ensure_dir(thresholds_path)
    with thresholds_path.open("w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    return thresholds