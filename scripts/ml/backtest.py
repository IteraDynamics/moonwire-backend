# scripts/ml/backtest.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Trade:
    ts_entry: pd.Timestamp
    ts_exit: pd.Timestamp
    symbol: str
    side: str          # "long" (short not used in v0.9.1)
    entry_px: float
    exit_px: float
    pnl: float
    pnl_pct: float


def _to_utc_hourly(df: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    """Ensure ts is UTC, truncate to hour, drop duplicates, sort."""
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], utc=True)
    # normalize to exact hour (no minutes/seconds) so joins are clean
    out[ts_col] = out[ts_col].dt.floor("H")
    out = out.drop_duplicates(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    return out


def _ensure_price_cols(px: pd.DataFrame) -> pd.DataFrame:
    """Allow either OHLCV or just 'close'. If only 'close' is present, keep it."""
    cols = set([c.lower() for c in px.columns])
    if "close" not in cols and "Close" not in px.columns:
        # if open/high/low exist, synthesize close from them (fallback)
        if {"open", "high", "low"}.issubset(cols):
            px = px.copy()
            # prefer explicit 'close' if present, otherwise average (gentle fallback)
            px["close"] = px[["open", "high", "low"]].mean(axis=1)
        else:
            # last resort: if there is exactly one price-like column, use it as close
            cand = [c for c in px.columns if c.lower() in {"price", "p", "value"}]
            if cand:
                px = px.copy()
                px["close"] = px[cand[0]]
    return px


def _max_drawdown(equity: np.ndarray) -> float:
    """Return max drawdown as a **negative** percentage (e.g., -0.123 for -12.3%)."""
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


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def run_backtest(
    pred_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    conf_min: float,
    debounce_min: int,
    horizon_h: int,
    fees_bps: float = 1.0,
    slippage_bps: float = 2.0,
    symbol: str = "",
) -> Dict[str, Any]:
    """
    Simple long-only threshold backtest:
      - merge hourly predictions with prices on ts
      - entry when p_long >= conf_min (debounced)
      - exit after horizon_h bars, or flip immediately if a new qualifying signal appears
      - entry at next bar's close; exit at close of exit bar
    Returns:
      {
        "metrics": {...},
        "trades": [ ... ],
        "equity": [ {"ts":..., "equity":...}, ... ]
      }
    """
    if pred_df is None or prices_df is None or pred_df.empty or prices_df.empty:
        return {
            "metrics": {"win_rate": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0, "signals_per_day": 0.0, "n_trades": 0},
            "trades": [],
            "equity": [],
        }

    # Normalize inputs
    px = _to_utc_hourly(prices_df, ts_col="ts")
    px = _ensure_price_cols(px)
    px = px[["ts", "close"]].dropna()
    preds = _to_utc_hourly(pred_df, ts_col="ts")
    # Keep only the prediction column we need
    if "p_long" not in preds.columns:
        # Try common fallbacks
        cand = [c for c in preds.columns if c.lower() in {"prob_long", "p", "prob"}]
        if not cand:
            return {
                "metrics": {"win_rate": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0, "signals_per_day": 0.0, "n_trades": 0},
                "trades": [],
                "equity": [],
            }
        preds = preds.rename(columns={cand[0]: "p_long"})
    preds = preds[["ts", "p_long"]].dropna()

    # Merge inner on hour; if empty, try nearest-asof (graceful alignment)
    merged = pd.merge(preds, px, on="ts", how="inner")
    if merged.empty:
        # asof join: align predictions to the next price bar (entry on next bar)
        merged = pd.merge_asof(preds.sort_values("ts"), px.sort_values("ts"), on="ts", direction="forward")
        merged = merged.dropna()
        merged = merged[["ts", "p_long", "close"]]

    if merged.empty:
        return {
            "metrics": {"win_rate": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0, "signals_per_day": 0.0, "n_trades": 0},
            "trades": [],
            "equity": [],
        }

    merged = merged.sort_values("ts").reset_index(drop=True)

    # Convert debounce from minutes to bars (hourly bars); minimum 1 bar between entries
    deb_bars = max(1, int(np.floor(debounce_min / 60.0)))

    # Iterate bars and simulate trades
    trades: List[Trade] = []
    equity_curve: List[Tuple[pd.Timestamp, float]] = []

    in_pos = False
    enter_idx = None
    enter_px = None
    last_entry_bar = -10_000  # very negative so first eligible signal is allowed

    # start equity at 1.0
    eq = 1.0

    for i in range(len(merged) - 1):  # -1 because we enter at next bar
        ts_i = merged.loc[i, "ts"]
        p_long = float(merged.loc[i, "p_long"])
        px_i = float(merged.loc[i, "close"])
        equity_curve.append((ts_i, eq))

        # check if we should exit due to horizon
        if in_pos and enter_idx is not None:
            if i - enter_idx >= horizon_h:
                # exit at this bar's close
                exit_px = px_i
                gross = (exit_px / enter_px) - 1.0
                fees = (fees_bps + slippage_bps) / 10000.0
                net = gross - 2 * fees  # entry + exit costs
                eq *= (1.0 + net)
                trades.append(
                    Trade(
                        ts_entry=merged.loc[enter_idx, "ts"],
                        ts_exit=ts_i,
                        symbol=symbol or "UNK",
                        side="long",
                        entry_px=float(enter_px),
                        exit_px=float(exit_px),
                        pnl=float(net),
                        pnl_pct=float(net),
                    )
                )
                in_pos = False
                enter_idx = None
                enter_px = None
                last_entry_bar = i  # debounce after a round trip

        # potential (flip) entry if not in position
        if not in_pos and p_long >= conf_min and (i - last_entry_bar) >= deb_bars:
            # enter at next bar close (if exists)
            j = i + 1
            if j < len(merged):
                enter_idx = j
                enter_px = float(merged.loc[j, "close"])
                in_pos = True
                last_entry_bar = i

    # close any open position at the last bar (graceful liquidation)
    if in_pos and enter_idx is not None:
        ts_last = merged.loc[len(merged) - 1, "ts"]
        exit_px = float(merged.loc[len(merged) - 1, "close"])
        gross = (exit_px / enter_px) - 1.0
        fees = (fees_bps + slippage_bps) / 10000.0
        net = gross - 2 * fees
        eq *= (1.0 + net)
        trades.append(
            Trade(
                ts_entry=merged.loc[enter_idx, "ts"],
                ts_exit=ts_last,
                symbol=symbol or "UNK",
                side="long",
                entry_px=float(enter_px),
                exit_px=float(exit_px),
                pnl=float(net),
                pnl_pct=float(net),
            )
        )

    # Metrics
    n_trades = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    losses = sum(1 for t in trades if t.pnl < 0)
    win_rate = float(wins / n_trades) if n_trades > 0 else 0.0

    prof = sum(t.pnl for t in trades if t.pnl > 0)
    loss = -sum(t.pnl for t in trades if t.pnl < 0)  # positive magnitude
    profit_factor = float(prof / loss) if loss > 1e-12 else (prof if prof > 0 else 0.0)

    eq_series = np.array([e for _, e in equity_curve], dtype=float)
    max_dd = _max_drawdown(eq_series)

    spd = _signals_per_day(n_trades, merged["ts"].min(), merged["ts"].max())

    metrics = {
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "max_drawdown": round(float(max_dd), 4),
        "signals_per_day": round(spd, 4),
        "n_trades": int(n_trades),
    }

    # Optional logs (guarded by env for grid-search performance)
    if os.getenv("MW_WRITE_BT_LOGS", "0") == "1":
        trades_records = [
            {
                "ts_entry": t.ts_entry.isoformat(),
                "ts_exit": t.ts_exit.isoformat(),
                "symbol": t.symbol,
                "side": t.side,
                "entry": t.entry_px,
                "exit": t.exit_px,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
            }
            for t in trades
        ]
        _write_jsonl(LOGS_DIR / "trades.jsonl", trades_records)
        _write_jsonl(
            LOGS_DIR / "equity_curve.jsonl",
            [{"ts": ts.isoformat(), "equity": float(e)} for ts, e in equity_curve],
        )

    return {
        "metrics": metrics,
        "trades": trades,  # list of dataclasses (tuner extracts n_trades defensively)
        "equity": [{"ts": ts, "equity": float(e)} for ts, e in equity_curve],
    }