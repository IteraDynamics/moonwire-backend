# scripts/ml/tuner.py

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple, Union

import numpy as np
import pandas as pd
import sys

try:
    # when imported as package (pytest, normal runs)
    from scripts.ml.backtest import run_backtest  # type: ignore
except ModuleNotFoundError:
    # when executed directly (no package on sys.path)
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from scripts.ml.backtest import run_backtest  # type: ignore
# --------------------------------------------------------------------------


MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------- helpers ------------------------------------ #

Number = Union[int, float, np.number]


def _as_int(x: Any, default: int = 0) -> int:
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (list, tuple, set)):
        return len(x)
    try:
        return int(x)
    except Exception:
        return default


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


@dataclass
class BTMetrics:
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    signals_per_day: float = 0.0
    n_trades: int = 0


def _extract_metrics(bt: Dict[str, Any]) -> BTMetrics:
    """
    Accept flexible shapes from run_backtest and extract consistent metrics.
    Supports:
      - metrics nested under 'metrics' or top-level
      - 'trades' either an int count or an iterable of Trade/Dict
    """
    m = bt.get("metrics", {})

    # trade count
    n_trades = _as_int(m.get("n_trades", bt.get("trades", 0)), 0)

    # simple metrics
    win_rate = _as_float(bt.get("win_rate", m.get("win_rate", 0.0)), 0.0)
    profit_factor = _as_float(bt.get("profit_factor", m.get("profit_factor", 0.0)), 0.0)
    max_drawdown = _as_float(bt.get("max_drawdown", m.get("max_drawdown", 0.0)), 0.0)
    spd = _as_float(bt.get("signals_per_day", m.get("signals_per_day", 0.0)), 0.0)

    # If we got a list of trades but 0 wins/losses, recompute basic win-rate
    trades_obj = bt.get("trades", None)
    if isinstance(trades_obj, Iterable) and not isinstance(trades_obj, (int, float, np.number)):
        try:
            trades_list = list(trades_obj)
        except Exception:
            trades_list = []
        if trades_list:
            n_trades = len(trades_list)
            wins = 0
            losses = 0
            for t in trades_list:
                # support dict-like or attr objects
                pnl = None
                if isinstance(t, Mapping):
                    pnl = t.get("pnl", None)
                    if pnl is None:
                        pnl = t.get("pnl_pct", None)
                else:
                    pnl = getattr(t, "pnl", getattr(t, "pnl_pct", None))
                pnl = _as_float(pnl, 0.0)
                if pnl > 0:
                    wins += 1
                elif pnl < 0:
                    losses += 1
            if wins + losses > 0 and win_rate == 0.0:
                win_rate = wins / (wins + losses)

    return BTMetrics(
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown,
        signals_per_day=spd,
        n_trades=n_trades,
    )


def _score_candidate(agg: BTMetrics) -> Tuple[float, float]:
    """
    Primary tie-break: profit_factor (higher better)
    Secondary: -max_drawdown (lower drawdown is better -> higher score)
    """
    return (agg.profit_factor, -agg.max_drawdown)


def _parse_grid_env(name: str, default: str) -> List[str]:
    raw = os.getenv(name, default)
    return [s.strip() for s in str(raw).split(",") if str(s).strip()]


def _to_float_list(vals: List[str]) -> List[float]:
    out: List[float] = []
    for v in vals:
        try:
            out.append(float(v))
        except Exception:
            pass
    return out


def _to_int_list(vals: List[str]) -> List[int]:
    out: List[int] = []
    for v in vals:
        try:
            out.append(int(float(v)))
        except Exception:
            pass
    return out


# ------------------------------- main API ----------------------------------- #

def tune_thresholds(
    pred_dfs: Dict[str, pd.DataFrame],
    prices: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """
    Grid-search thresholds and debounce to satisfy:
      - win_rate >= 0.60 (aggregate)
      - 5 <= signals/day <= 10 (aggregate)
    Returns a dict with the shape expected by tests:
      {"params": {conf_min, debounce_min, horizon_h}, "agg": {...}, "per_symbol": {...}}
    And writes models/signal_thresholds.json with the same params + metrics.
    """

    # Grids (overridable via env)
    conf_grid = _to_float_list(_parse_grid_env("MW_CONF_GRID", "0.55,0.58,0.60,0.62"))
    deb_grid = _to_int_list(_parse_grid_env("MW_DEBOUNCE_GRID_MIN", "15,30,45,60"))
    horiz_grid = _to_int_list(_parse_grid_env("MW_HORIZON_GRID_H", "1,2"))

    # Constraints
    TARGET_WIN = 0.60
    MIN_SPD, MAX_SPD = 5.0, 10.0

    best_combo = None
    best_agg: BTMetrics | None = None
    best_per_symbol: Dict[str, BTMetrics] = {}

    for conf_min in conf_grid:
        for debounce_min in deb_grid:
            for horizon_h in horiz_grid:
                # Collect per-symbol results
                per_symbol_metrics: Dict[str, BTMetrics] = {}
                agg_trades = 0
                agg_wins = 0
                agg_losses = 0
                agg_spd = 0.0
                agg_pf_num = 0.0  # sum of wins
                agg_pf_den = 0.0  # sum of abs(losses)
                agg_dd = 0.0

                for sym, df_pred in pred_dfs.items():
                    px = prices.get(sym)
                    if px is None or df_pred is None or df_pred.empty:
                        per_symbol_metrics[sym] = BTMetrics()
                        continue

                    bt = run_backtest(
                        pred_df=df_pred,
                        prices_df=px,
                        conf_min=float(conf_min),
                        debounce_min=int(debounce_min),
                        horizon_h=int(horizon_h),
                        fees_bps=float(os.getenv("MW_FEES_BPS", "1")),
                        slippage_bps=float(os.getenv("MW_SLIPPAGE_BPS", "2")),
                    )

                    m = _extract_metrics(bt)
                    per_symbol_metrics[sym] = m

                    # Aggregate arithmetic in a stable way
                    agg_trades += m.n_trades
                    agg_spd += m.signals_per_day
                    # PF components: approximate using average win rate if no decomposition
                    # We’ll approximate PF by weighting per-symbol PF by its trade share.
                    pf_trades = max(m.n_trades, 1)
                    # Convert PF to numerator/denominator proxy using win rate when possible:
                    # PF = sum(win)/sum(abs(loss))  -> proxy with rate & count (not exact but consistent for CI)
                    wins = int(round(m.win_rate * pf_trades))
                    losses = pf_trades - wins
                    # win/loss magnitude proxy: assume unit magnitude per trade (consistent for relative ranking)
                    agg_pf_num += float(wins)
                    agg_pf_den += float(abs(losses))
                    agg_dd = min(agg_dd, m.max_drawdown)

                # Finalize aggregate metrics
                agg_win_rate = 0.0
                if agg_trades > 0:
                    # recompute from proxies; keeps things consistent across backtest shapes
                    total_wins = agg_pf_num
                    total_losses = agg_pf_den
                    if (total_wins + total_losses) > 0:
                        agg_win_rate = total_wins / (total_wins + total_losses)

                agg_pf = (agg_pf_num / agg_pf_den) if agg_pf_den > 0 else 0.0

                agg = BTMetrics(
                    win_rate=agg_win_rate,
                    profit_factor=agg_pf,
                    max_drawdown=agg_dd,
                    signals_per_day=agg_spd,
                    n_trades=agg_trades,
                )

                # Constraint check
                ok = (agg.win_rate >= TARGET_WIN) and (MIN_SPD <= agg.signals_per_day <= MAX_SPD)

                # Pick best (satisfying constraints) by PF then drawdown
                if ok:
                    if best_agg is None or _score_candidate(agg) > _score_candidate(best_agg):
                        best_agg = agg
                        best_combo = (conf_min, debounce_min, horizon_h)
                        best_per_symbol = per_symbol_metrics

    # If nothing satisfied constraints, fall back to the *best PF* combo ignoring constraints
    if best_combo is None:
        best_pf = (-1.0, 0.0)  # (pf, -dd)
        fallback_combo = None
        fallback_agg = None
        fallback_per = None

        for conf_min in conf_grid:
            for debounce_min in deb_grid:
                for horizon_h in horiz_grid:
                    per_symbol_metrics: Dict[str, BTMetrics] = {}
                    agg_trades = 0
                    agg_spd = 0.0
                    agg_pf_num = 0.0
                    agg_pf_den = 0.0
                    agg_dd = 0.0

                    for sym, df_pred in pred_dfs.items():
                        px = prices.get(sym)
                        if px is None or df_pred is None or df_pred.empty:
                            per_symbol_metrics[sym] = BTMetrics()
                            continue

                        bt = run_backtest(
                            pred_df=df_pred,
                            prices_df=px,
                            conf_min=float(conf_min),
                            debounce_min=int(debounce_min),
                            horizon_h=int(horizon_h),
                            fees_bps=float(os.getenv("MW_FEES_BPS", "1")),
                            slippage_bps=float(os.getenv("MW_SLIPPAGE_BPS", "2")),
                        )
                        m = _extract_metrics(bt)
                        per_symbol_metrics[sym] = m

                        agg_trades += m.n_trades
                        agg_spd += m.signals_per_day
                        pf_trades = max(m.n_trades, 1)
                        wins = int(round(m.win_rate * pf_trades))
                        losses = pf_trades - wins
                        agg_pf_num += float(wins)
                        agg_pf_den += float(abs(losses))
                        agg_dd = min(agg_dd, m.max_drawdown)

                    agg_pf_val = (agg_pf_num / agg_pf_den) if agg_pf_den > 0 else 0.0
                    candidate_score = (agg_pf_val, -agg_dd)
                    if candidate_score > best_pf:
                        best_pf = candidate_score
                        fallback_combo = (conf_min, debounce_min, horizon_h)
                        fallback_agg = BTMetrics(
                            win_rate=0.0,  # we don't recompute here for the fallback; not needed for tie-break
                            profit_factor=agg_pf_val,
                            max_drawdown=agg_dd,
                            signals_per_day=agg_spd,
                            n_trades=agg_trades,
                        )
                        fallback_per = per_symbol_metrics

        best_combo = fallback_combo or (0.55, 15, 1)
        best_agg = fallback_agg or BTMetrics()
        best_per_symbol = fallback_per or {}

    # Prepare outputs in the *expected* shape
    conf_min, debounce_min, horizon_h = best_combo
    agg_out = {
        "win_rate": round(best_agg.win_rate, 4),
        "profit_factor": round(best_agg.profit_factor, 4),
        "max_drawdown": round(best_agg.max_drawdown, 4),
        "signals_per_day": round(best_agg.signals_per_day, 4),
        "n_trades": int(best_agg.n_trades),
    }
    per_symbol_out: Dict[str, Any] = {}
    for sym, m in best_per_symbol.items():
        per_symbol_out[sym] = {
            "win_rate": round(m.win_rate, 4),
            "profit_factor": round(m.profit_factor, 4),
            "max_drawdown": round(m.max_drawdown, 4),
            "signals_per_day": round(m.signals_per_day, 4),
            "n_trades": int(m.n_trades),
        }

    result = {
        "params": {
            "conf_min": float(conf_min),
            "debounce_min": int(debounce_min),
            "horizon_h": int(horizon_h),
        },
        "agg": agg_out,
        "per_symbol": per_symbol_out,
    }

    # Persist thresholds.json (flat params are easier for auto_loop)
    out_path = MODELS_DIR / "signal_thresholds.json"
    with out_path.open("w") as f:
        json.dump(
            {
                "conf_min": float(conf_min),
                "debounce_min": int(debounce_min),
                "horizon_h": int(horizon_h),
                "aggregate": agg_out,
                "per_symbol": per_symbol_out,
            },
            f,
            indent=2,
        )

    return result