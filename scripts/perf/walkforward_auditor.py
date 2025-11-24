from __future__ import annotations

"""
Purged Walk-Forward Auditor

Implements a class-based, regime-agnostic walk-forward analysis over
historical inference logs (JSONL). Each line in the input file is expected
to be a JSON object with at least:

    {
        "ts": "... ISO8601 ...",
        "symbol": "BTC",
        "ret": 0.01   # per-trade or per-bar return (optional but needed for metrics)
        ...
    }

The auditor:

1. Loads and filters rows from a shadow inference log.
2. Restricts to a trailing window (window_hours) ending at the last timestamp.
3. Partitions that window into K contiguous test folds in time.
4. For each fold, defines a *context* window that ends embargo_hours
   before the test window starts (purged walk-forward).
5. Computes per-fold and per-symbol Sharpe and max drawdown over the
   *test* window only (context is there to mirror training, but not scored).

This follows the spirit of Marcos López de Prado's purged walk-forward /
K-fold scheme by inserting a temporal gap (embargo) between the context
training window and the test audit window to minimize leakage.
"""

import json
import logging
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShadowRecord:
    ts: datetime
    symbol: str
    ret: Optional[float]
    raw: Dict[str, object]


@dataclass(frozen=True)
class FoldDefinition:
    index: int
    context_start: datetime
    context_end: datetime
    test_start: datetime
    test_end: datetime


class WalkForwardAuditor:
    """
    Class-based purged walk-forward auditor over a shadow inference JSONL log.

    Typical usage:
        auditor = WalkForwardAuditor(
            shadow_log_path=Path("logs/signal_inference_shadow.jsonl"),
            k_folds=5,
            window_hours=720,
            embargo_hours=6,
        )
        auditor.load_data(symbols=["BTC", "ETH"])
        folds = auditor.generate_folds()
        results = auditor.audit_performance(folds)
    """

    def __init__(
        self,
        shadow_log_path: Path,
        k_folds: int,
        window_hours: int,
        embargo_hours: int,
        return_key: str = "ret",
    ) -> None:
        self.shadow_log_path: Path = shadow_log_path
        self.k_folds: int = k_folds
        self.window_hours: int = window_hours
        self.embargo_hours: int = embargo_hours
        self.return_key: str = return_key

        self._records: List[ShadowRecord] = []
        self._timestamps: List[datetime] = []

    # -------------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------------

    def load_data(self, symbols: Optional[List[str]] = None) -> None:
        """
        Load and cache records from the JSONL shadow log.

        Parameters
        ----------
        symbols : Optional[List[str]]
            If provided, only include records whose 'symbol' is in this list.
            Symbols are uppercased before comparison.
        """
        if not self.shadow_log_path.exists():
            raise FileNotFoundError(f"Shadow log not found at: {self.shadow_log_path}")

        symbol_filter = {s.upper() for s in symbols} if symbols else None

        logger.info(
            "Loading shadow records from %s (symbols=%s)",
            self.shadow_log_path,
            ",".join(symbol_filter) if symbol_filter else "<ALL>",
        )

        records: List[ShadowRecord] = []

        with self.shadow_log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload: Dict[str, object] = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSON line")
                    continue

                ts_raw = str(payload.get("ts", ""))
                ts = self._parse_ts(ts_raw)
                if ts is None:
                    logger.debug("Skipping record with invalid timestamp: %s", ts_raw)
                    continue

                symbol = str(payload.get("symbol", "")).upper()
                if symbol_filter and symbol not in symbol_filter:
                    continue

                ret_val: Optional[float] = None
                raw_ret = payload.get(self.return_key)
                if isinstance(raw_ret, (int, float)):
                    ret_val = float(raw_ret)

                records.append(
                    ShadowRecord(
                        ts=ts,
                        symbol=symbol,
                        ret=ret_val,
                        raw=payload,
                    )
                )

        records.sort(key=lambda r: r.ts)
        if not records:
            raise ValueError("No usable shadow records after filtering.")

        # Restrict to trailing window_hours (if applicable)
        last_ts: datetime = records[-1].ts
        window_start: datetime = last_ts - timedelta(hours=self.window_hours)

        filtered: List[ShadowRecord] = [
            r for r in records if r.ts >= window_start
        ]

        if not filtered:
            raise ValueError(
                f"No records within the last {self.window_hours} hours "
                f"(window_start={window_start.isoformat()})."
            )

        self._records = filtered
        self._timestamps = [r.ts for r in filtered]

        logger.info(
            "Loaded %d records in window [%s .. %s]",
            len(self._records),
            self._records[0].ts.isoformat(),
            self._records[-1].ts.isoformat(),
        )

    # -------------------------------------------------------------------------
    # Fold generation (purged walk-forward)
    # -------------------------------------------------------------------------

    def generate_folds(self) -> List[FoldDefinition]:
        """
        Partition the trailing window into k contiguous *test* folds in time,
        and derive context windows for each fold, with an embargo between
        context_end and test_start.

        Returns
        -------
        List[FoldDefinition]
            One entry per fold, with context and test time bounds.
        """
        if not self._records:
            raise RuntimeError("No records loaded. Call load_data() first.")

        start: datetime = self._records[0].ts
        end: datetime = self._records[-1].ts

        total_seconds: float = (end - start).total_seconds()
        if total_seconds <= 0:
            raise ValueError("Non-positive time span in data window.")

        fold_seconds: float = total_seconds / float(self.k_folds)
        embargo_delta: timedelta = timedelta(hours=self.embargo_hours)

        folds: List[FoldDefinition] = []

        logger.info(
            "Generating %d folds over [%s .. %s] with embargo=%dh",
            self.k_folds,
            start.isoformat(),
            end.isoformat(),
            self.embargo_hours,
        )

        for idx in range(self.k_folds):
            # Define test window as contiguous slices of [start, end]
            test_start: datetime = start + timedelta(seconds=fold_seconds * idx)
            if idx == self.k_folds - 1:
                test_end: datetime = end
            else:
                test_end = start + timedelta(seconds=fold_seconds * (idx + 1))

            # Context is everything up to (test_start - embargo)
            context_end: datetime = test_start - embargo_delta
            if context_end < start:
                context_end = start

            context_start: datetime = start

            folds.append(
                FoldDefinition(
                    index=idx + 1,
                    context_start=context_start,
                    context_end=context_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )

            logger.debug(
                "Fold %d: context=[%s .. %s], test=[%s .. %s]",
                idx + 1,
                context_start.isoformat(),
                context_end.isoformat(),
                test_start.isoformat(),
                test_end.isoformat(),
            )

        return folds

    # -------------------------------------------------------------------------
    # Performance audit
    # -------------------------------------------------------------------------

    def audit_performance(
        self,
        folds: List[FoldDefinition],
    ) -> Dict[str, object]:
        """
        Compute Sharpe and max drawdown per fold (and per symbol) over
        the *test* windows implied by FoldDefinition entries.

        Parameters
        ----------
        folds : List[FoldDefinition]
            Fold definitions as produced by generate_folds().

        Returns
        -------
        Dict[str, object]
            JSON-serializable structure with:
                {
                    "meta": {...},
                    "folds": [
                        {
                            "fold": int,
                            "context": {...},
                            "test": {...},
                            "by_symbol": {
                                "BTC": {...},
                                ...
                            },
                            "aggregate": {...}
                        },
                        ...
                    ]
                }
        """
        if not self._records or not self._timestamps:
            raise RuntimeError("No records loaded. Call load_data() first.")

        logger.info("Auditing performance across %d folds", len(folds))

        fold_results: List[Dict[str, object]] = []

        for fd in folds:
            logger.info(
                "Evaluating fold %d: test=[%s .. %s]",
                fd.index,
                fd.test_start.isoformat(),
                fd.test_end.isoformat(),
            )

            test_records: List[ShadowRecord] = self._slice_by_time(
                fd.test_start,
                fd.test_end,
            )

            # Group by symbol
            by_symbol_returns: Dict[str, List[float]] = {}
            for rec in test_records:
                if rec.ret is None:
                    continue
                by_symbol_returns.setdefault(rec.symbol, []).append(rec.ret)

            fold_entry: Dict[str, object] = {
                "fold": fd.index,
                "context": {
                    "start": fd.context_start.isoformat(),
                    "end": fd.context_end.isoformat(),
                },
                "test": {
                    "start": fd.test_start.isoformat(),
                    "end": fd.test_end.isoformat(),
                    "n_records": len(test_records),
                },
                "by_symbol": {},
                "aggregate": {},
            }

            # Per-symbol metrics
            all_returns: List[float] = []
            for symbol, rets in by_symbol_returns.items():
                metrics = self._compute_metrics(rets)
                fold_entry["by_symbol"][symbol] = metrics  # type: ignore[index]
                all_returns.extend(rets)

            # Aggregate metrics across all symbols
            fold_entry["aggregate"] = self._compute_metrics(all_returns)

            fold_results.append(fold_entry)

        # Simple aggregate summary: counts only (you can extend as needed)
        aggregate_counts: Dict[str, object] = {
            "folds": len(folds),
            "total_records": len(self._records),
        }

        result: Dict[str, object] = {
            "meta": {
                "shadow_log_path": str(self.shadow_log_path),
                "k_folds": self.k_folds,
                "window_hours": self.window_hours,
                "embargo_hours": self.embargo_hours,
                "return_key": self.return_key,
            },
            "folds": fold_results,
            "aggregate": aggregate_counts,
        }

        logger.info("Walk-forward audit completed")
        return result

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _parse_ts(s: str) -> Optional[datetime]:
        if not s:
            return None
        try:
            # Handle trailing 'Z'
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    def _slice_by_time(
        self,
        start: datetime,
        end: datetime,
    ) -> List[ShadowRecord]:
        """
        Efficiently slice self._records between [start, end] using bisect,
        assuming self._timestamps is sorted.
        """
        ts = self._timestamps
        left: int = bisect_left(ts, start)
        right: int = bisect_right(ts, end)
        return self._records[left:right]

    @staticmethod
    def _compute_metrics(returns: List[float]) -> Dict[str, Optional[float]]:
        """
        Compute simple Sharpe (unscaled) and max drawdown over a vector of returns.

        Parameters
        ----------
        returns : List[float]
            Series of per-step returns (e.g. pct PnL).

        Returns
        -------
        Dict[str, Optional[float]]
            {
                "trades": int,
                "sharpe": float | None,
                "max_drawdown": float | None,
            }
        """
        n = len(returns)
        if n == 0:
            return {
                "trades": 0,
                "sharpe": None,
                "max_drawdown": None,
            }

        # Simple Sharpe: mean / std (no annualization)
        mean_ret: float = sum(returns) / float(n)
        var: float = sum((r - mean_ret) ** 2 for r in returns) / float(n)
        std: float = var ** 0.5
        sharpe: Optional[float] = mean_ret / std if std > 0.0 else None

        # Max drawdown over cumulative equity
        equity: float = 0.0
        peak: float = 0.0
        max_dd: float = 0.0
        for r in returns:
            equity += r
            if equity > peak:
                peak = equity
            dd = equity - peak
            if dd < max_dd:
                max_dd = dd

        # max_dd is negative (drop from peak); report as a negative value
        return {
            "trades": n,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
        }

if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Purged Walk-Forward Validation Auditor"
    )

    parser.add_argument(
        "--log-path",
        type=Path,
        required=True,
        help="Path to shadow inference JSONL log",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTC,ETH",
        help="Comma-separated list of symbols to include",
    )
    parser.add_argument(
        "--k-folds",
        type=int,
        default=5,
        help="Number of walk-forward folds",
    )
    parser.add_argument(
        "--window-hours",
        type=int,
        default=720,
        help="Trailing window size (hours)",
    )
    parser.add_argument(
        "--embargo-hours",
        type=int,
        default=6,
        help="Embargo gap between train and test windows (hours)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON file; if omitted prints to stdout",
    )

    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    auditor = WalkForwardAuditor(
        shadow_log_path=args.log_path,
        k_folds=args.k_folds,
        window_hours=args.window_hours,
        embargo_hours=args.embargo_hours,
    )

    auditor.load_data(symbols=symbols)
    folds = auditor.generate_folds()
    results = auditor.audit_performance(folds)

    if args.output:
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {args.output}")
    else:
        json.dump(results, sys.stdout, indent=2)
        print()