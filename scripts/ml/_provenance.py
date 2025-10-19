from __future__ import annotations
from typing import Dict, List, Any
from datetime import datetime, timezone

def detect_provenance(prices: Dict[str, Any], symbols: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbols": symbols,
        "per_symbol": {},
        "overall_source": "unknown",
    }
    sources = set()
    for s in symbols:
        df = prices.get(s)
        if df is None or getattr(df, "empty", True):
            out["per_symbol"][s] = {"rows": 0, "ts_start": None, "ts_end": None, "source": "missing"}
            sources.add("missing")
            continue

        src = None
        if hasattr(df, "attrs") and isinstance(df.attrs, dict):
            src = df.attrs.get("source")  # e.g. "api" or "synthetic"

        info = {
            "rows": int(getattr(df, "shape", (0,))[0]),
            "ts_start": df.index.min().isoformat() if len(df) else None,
            "ts_end": df.index.max().isoformat() if len(df) else None,
            "source": src or "unknown",
        }
        out["per_symbol"][s] = info
        sources.add(info["source"])

    if "synthetic" in sources or "demo" in sources:
        out["overall_source"] = "synthetic_or_demo"
    elif "api" in sources:
        out["overall_source"] = "api"
    elif "missing" in sources:
        out["overall_source"] = "missing"
    else:
        out["overall_source"] = "unknown"

    return out