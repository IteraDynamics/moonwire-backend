#!/usr/bin/env python3
"""
Convenience wrapper to (re)build Market Context artifacts.

Usage examples:

# Demo (no network)
python scripts/mw_build_market_context.py --demo

# Live CoinGecko (requires key in env or flag)
python scripts/mw_build_market_context.py \
  --no-demo \
  --cg-api-key "$MW_CG_API_KEY" \
  --cg-base-url "https://api.coingecko.com/api/v3" \
  --coins "bitcoin,ethereum,solana" \
  --lookback-h 72

You can also set environment variables instead of flags:
  MW_DEMO=true|false
  MW_CG_API_KEY=...
  MW_CG_BASE_URL=...
  MW_CG_COINS=bitcoin,ethereum,solana
  MW_CG_VS_CURRENCY=usd
  MW_CG_LOOKBACK_H=72
  MW_CG_RATE_LIMIT_PER_MIN=25
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Market Context artifacts via CoinGecko ingest.")
    # Paths
    p.add_argument("--logs-dir", default="logs", help="Directory for append-only logs (default: logs)")
    p.add_argument("--models-dir", default="models", help="Directory for JSON artifacts (default: models)")
    p.add_argument("--artifacts-dir", default="artifacts", help="Directory for PNG artifacts (default: artifacts)")

    # Demo / live
    demo_default = os.getenv("MW_DEMO", "true").lower() in ("1", "true", "yes", "y")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--demo", dest="demo", action="store_true", default=demo_default, help="Force demo mode")
    g.add_argument("--no-demo", dest="demo", action="store_false", help="Disable demo mode (use live data)")

    # CoinGecko knobs
    p.add_argument("--cg-api-key", default=os.getenv("MW_CG_API_KEY"), help="CoinGecko API key (Pro key if applicable)")
    p.add_argument("--cg-base-url", default=os.getenv("MW_CG_BASE_URL", "https://api.coingecko.com/api/v3"),
                   help="CoinGecko base URL")
    p.add_argument("--coins", default=os.getenv("MW_CG_COINS", "bitcoin,ethereum,solana"),
                   help="Comma-separated list of coins (ids)")
    p.add_argument("--vs", default=os.getenv("MW_CG_VS_CURRENCY", "usd"), help="Quote currency (default: usd)")
    p.add_argument("--lookback-h", type=int, default=int(os.getenv("MW_CG_LOOKBACK_H", "72")),
                   help="Lookback window in hours (default: 72)")
    p.add_argument("--rate-limit-per-min", type=int, default=int(os.getenv("MW_CG_RATE_LIMIT_PER_MIN", "25")),
                   help="Client-side pacing for requests per minute")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    # Ensure dirs
    logs_dir = Path(args.logs_dir); logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir); models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(args.artifacts_dir); artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Reflect flags into env that the underlying ingest already reads
    os.environ["MW_DEMO"] = "true" if args.demo else "false"
    if args.cg_api_key:
        os.environ["MW_CG_API_KEY"] = args.cg_api_key
    if args.cg_base_url:
        os.environ["MW_CG_BASE_URL"] = args.cg_base_url
    if args.coins:
        os.environ["MW_CG_COINS"] = args.coins
    if args.vs:
        os.environ["MW_CG_VS_CURRENCY"] = args.vs
    os.environ["MW_CG_LOOKBACK_H"] = str(args.lookback_h)
    os.environ["MW_CG_RATE_LIMIT_PER_MIN"] = str(args.rate_limit_per_min)

    try:
        from scripts.market.ingest_market import run_ingest
    except Exception as e:
        print(f"[mw_build_market_context] Failed to import ingest: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    try:
        run_ingest(logs_dir=logs_dir, models_dir=models_dir, artifacts_dir=artifacts_dir)
    except Exception as e:
        print(f"[mw_build_market_context] Ingest failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print("[mw_build_market_context] Done.")
    print(f"  models:    {models_dir / 'market_context.json'}")
    print(f"  artifacts: {(artifacts_dir / 'market_trend_price_bitcoin.png').parent}")
    print(f"  logs:      {logs_dir / 'market_prices.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())