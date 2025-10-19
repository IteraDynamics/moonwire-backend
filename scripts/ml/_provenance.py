# scripts/ml/_provenance.py
from pathlib import Path
import json
import pandas as pd
import time


def write_data_provenance(prices: dict, *, source_tag: str, lookback_days: int, out_dir: Path):
    """
    Write a provenance record for the loaded market data.

    Args:
        prices (dict): mapping of symbol -> DataFrame
        source_tag (str): 'real', 'demo', 'api', etc.
        lookback_days (int): number of days of data requested
        out_dir (Path): directory to save the provenance file

    Returns:
        dict: provenance info
    """
    rows = 0
    ts_min = None
    ts_max = None

    for df in prices.values():
        if df is None or len(df) == 0:
            continue
        rows += len(df)
        lo, hi = df["ts"].min(), df["ts"].max()
        ts_min = lo if ts_min is None else min(ts_min, lo)
        ts_max = hi if ts_max is None else max(ts_max, hi)

    prov = {
        "source": source_tag,
        "lookback_days": lookback_days,
        "symbols": sorted(prices.keys()),
        "rows_total": int(rows),
        "ts_min": None if ts_min is None else pd.Timestamp(ts_min).isoformat(),
        "ts_max": None if ts_max is None else pd.Timestamp(ts_max).isoformat(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data_provenance.json").write_text(json.dumps(prov, indent=2))
    return prov