#!/usr/bin/env python3
from __future__ import annotations 

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Any, Dict

# Local import (repo layout)
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

from src.signal_log import write_signal  # noqa: E402


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Emit a single signal JSONL row to MoonWire logs (dual-write by default)."
    )
    p.add_argument("--symbol", required=True, help="Symbol (e.g., BTC, ETH, SOL)")
    p.add_argument("--dir", required=True, choices=["long", "short"], help="Direction")
    p.add_argument("--conf", type=float, default=0.70, help="Confidence (float)")
    p.add_argument("--price", type=float, required=True, help="Entry price")
    p.add_argument("--ts", default=None, help="ISO8601 UTC timestamp (default: now)")
    p.add_argument("--model", default="v0.9.0", help="Model version (default: v0.9.0)")
    p.add_argument("--source", default="manual", help="Source tag (default: manual)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    ts = (args.ts or _now_utc_iso()).strip()
    row: Dict[str, Any] = {
        # id is optional — write_signal will auto-generate if missing
        "id": None,
        "ts": ts,
        "symbol": (args.symbol or "").strip(),
        "direction": (args.dir or "").strip(),
        "confidence": float(args.conf),
        "price": float(args.price),
        "source": (args.source or "").strip(),
        "model_version": (args.model or "").strip(),
        "outcome": None,
    }

    normalized = write_signal(row)
    print(json.dumps(normalized, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
