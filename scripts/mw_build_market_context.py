#!/usr/bin/env python3
"""
mw_build_market_context.py

Tiny CLI wrapper to (re)build Market Context artifacts using CoinGecko ingest.

- Calls scripts.market.ingest_market.run_ingest(logs_dir, models_dir, artifacts_dir)
- Respects --demo flag (by setting MW_DEMO)
- Lets you override directories via flags; defaults to ./logs, ./models, ./artifacts
- Exits non-zero on failure so CI can catch it
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

def _bool_env(val: str | None) -> bool:
    if val is None:
        return False
    return val.strip().lower() in ("1", "true", "yes", "y", "on")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Market Context (CoinGecko) artifacts.")
    p.add_argument("--logs", dest="logs_dir", type=Path, default=Path("logs"),
                   help="Directory for append-only logs (default: ./logs)")
    p.add_argument("--models", dest="models_dir", type=Path, default=Path("models"),
                   help="Directory for JSON models (default: ./models)")
    p.add_argument("--artifacts", dest="artifacts_dir", type=Path, default=Path("artifacts"),
                   help="Directory for PNG artifacts (default: ./artifacts)")
    p.add_argument("--demo", dest="demo", action="store_true",
                   help="Force demo mode (sets MW_DEMO=true for this run)")
    p.add_argument("--no-demo", dest="no_demo", action="store_true",
                   help="Force live mode (sets MW_DEMO=false for this run)")
    # passthrough convenience (optional; ingest reads envs)
    p.add_argument("--coins", dest="coins", type=str, default=None,
                   help="Override MW_CG_COINS (e.g., 'bitcoin,ethereum,solana')")
    p.add_argument("--vs", dest="vs", type=str, default=None,
                   help="Override MW_CG_VS_CURRENCY (default via env)")
    p.add_argument("--lookback-h", dest="lookback_h", type=str, default=None,
                   help="Override MW_CG_LOOKBACK_H (hours)")
    p.add_argument("--base-url", dest="base_url", type=str, default=None,
                   help="Override MW_CG_BASE_URL")
    return p.parse_args()

def main() -> int:
    args = parse_args()

    # Ensure directories exist
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Respect --demo / --no-demo (explicit flag beats env)
    if args.demo and args.no_demo:
        print("[mw_build_market_context] --demo and --no-demo are mutually exclusive.", file=sys.stderr)
        return 2
    if args.demo:
        os.environ["MW_DEMO"] = "true"
    elif args.no_demo:
        os.environ["MW_DEMO"] = "false"
    else:
        # leave MW_DEMO as-is; allow existing env to flow through
        pass

    # Optional passthrough envs
    if args.coins is not None:
        os.environ["MW_CG_COINS"] = args.coins
    if args.vs is not None:
        os.environ["MW_CG_VS_CURRENCY"] = args.vs
    if args.lookback_h is not None:
        os.environ["MW_CG_LOOKBACK_H"] = args.lookback_h
    if args.base_url is not None:
        os.environ["MW_CG_BASE_URL"] = args.base_url

    try:
        from scripts.market.ingest_market import run_ingest
    except Exception as e:
        print(f"[mw_build_market_context] Import failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    try:
        # IMPORTANT: your run_ingest expects *positional* arguments
        out = run_ingest(args.logs_dir, args.models_dir, args.artifacts_dir)
        # Be chatty on success
        print("[mw_build_market_context] Done.")
        if out is not None:
            print(out)
        # Sanity: show the key outputs if present
        mc = args.models_dir / "market_context.json"
        if mc.exists():
            print(f"[mw_build_market_context] wrote {mc}")
        return 0
    except Exception as e:
        print(f"[mw_build_market_context] Ingest failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())