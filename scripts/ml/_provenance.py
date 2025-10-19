# scripts/ml/_provenance.py
from __future__ import annotations
import os
from typing import Dict, Any, Iterable
import pandas as pd

def _fmt_ts(x):
    """Best-effort timestamp pretty-printer: supports int, float, str, pandas types."""
    try:
        # ints/floats as epoch seconds → ISO
        if isinstance(x, (int, float)):
            return pd.to_datetime(x, unit="s", utc=True).isoformat()
        # pandas/np datetime-like → ISO
        try:
            return pd.to_datetime(x, utc=True).isoformat()
        except Exception:
            return str(x)
    except Exception:
        return str(x)

def _series_info(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {"rows": int(len(df))}
    if "ts" in df.columns and len(df):
        ts = df["ts"]
        out["ts_min"] = _fmt_ts(ts.iloc[0])
        out["ts_max"] = _fmt_ts(ts.iloc[-1])
        out["ts_monotonic"] = bool(ts.is_monotonic_increasing)
    if "close" in df.columns and len(df):
        # light-weight sanity about the data being "real-ish"
        out["close_min"] = float(pd.to_numeric(df["close"], errors="coerce").min())
        out["close_max"] = float(pd.to_numeric(df["close"], errors="coerce").max())
        out["close_var"] = float(pd.to_numeric(df["close"], errors="coerce").var(ddof=0) or 0.0)
    return out

def detect_provenance(
    prices: Dict[str, pd.DataFrame],
    symbols: Iterable[str],
    lookback_days: int | None = None,
    env: Dict[str, Any] | None = None,
    **_
) -> Dict[str, Any]:
    """
    Return a small, robust provenance payload. We intentionally avoid throwing;
    anything that fails is recorded under 'error' but we still return a result.
    """
    env = dict(env or {})
    payload: Dict[str, Any] = {
        "symbols": list(symbols),
        "lookback_days": int(lookback_days) if lookback_days is not None else None,
        "env": {
            "MW_OFFLINE_DEMO": os.getenv("MW_OFFLINE_DEMO", env.get("MW_OFFLINE_DEMO")),
            "MW_DEMO": os.getenv("MW_DEMO", env.get("MW_DEMO")),
            "DEMO_MODE": os.getenv("DEMO_MODE", env.get("DEMO_MODE")),
            "MW_BRANCH": os.getenv("MW_BRANCH", env.get("MW_BRANCH")),
        },
        "series": {},
    }

    # Decide "source" primarily from env flags; fall back to a quick heuristic.
    demo_flags = {
        str(payload["env"].get("MW_OFFLINE_DEMO", "0")).strip().lower() in {"1", "true"},
        str(payload["env"].get("MW_DEMO", "0")).strip().lower() in {"1", "true"},
        str(payload["env"].get("DEMO_MODE", "0")).strip().lower() in {"1", "true"},
    }
    source = "demo" if any(demo_flags) else "real"

    try:
        # Attach light stats per symbol (never fail build if a symbol is odd)
        for sym in symbols:
            df = prices.get(sym)
            if df is None or not hasattr(df, "empty") or df.empty:
                payload["series"][sym] = {"rows": 0}
                continue
            payload["series"][sym] = _series_info(df)
    except Exception as e:
        payload["error"] = f"{type(e).__name__}: {e}"

    payload["source"] = source
    return payload