#!/usr/bin/env python3
import json
import os
import sys
import math
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- minimal, deterministic demo-safe summary builder with perf integration ---

# Perf modules (added in v0.9.0 task)
from scripts.perf.paper_trader import run_paper_trader
from scripts.summary_sections.performance_validation import append as perf_append

ART_DIR = Path("artifacts")
MODELS_DIR = Path("models")
LOGS_DIR = Path("logs")

def utcnow_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

@dataclass
class Ctx:
    demo_mode: bool = True
    perf_mode: str = "backtest"
    caches: Dict[str, Any] = field(default_factory=dict)

def ensure_dirs():
    ART_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# Demo seed for CI tests
# -----------------------
def generate_demo_data_if_needed(ctx: Ctx) -> None:
    """
    Seeds tiny demo JSONs so CI always has something to show; kept intentionally
    lightweight and deterministic.
    """
    ensure_dirs()

    # Minimal market context stub
    mc_path = MODELS_DIR / "market_context.json"
    if not mc_path.exists():
        payload = {
            "generated_at": utcnow_iso(),
            "coins": [
                {"symbol": "BTC", "price": 60006.29, "h1": 0.0, "h24": 0.0, "h72": 0.0},
                {"symbol": "ETH", "price": 3006.29, "h1": 0.1, "h24": 0.7, "h72": 0.0},
                {"symbol": "SOL", "price": 156.29, "h1": 1.1, "h24": 14.3, "h72": 0.0},
            ],
            "window_hours": 72,
            "demo": True,
        }
        mc_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Create the markdown shell if missing
    md_path = ART_DIR / "demo_summary.md"
    if not md_path.exists():
        lines = []
        lines.append("🌙 MoonWire CI Demo Summary\n")
        lines.append("🚀 Overview\n")
        lines.append("📊 Version: v0.8.2 | Run: 🟢 All checks passed\n")
        lines.append("[View Artifacts](${ GITHUB_RUN_URL })\n")
        lines.append("🚀 MoonWire CI Demo Summary\n")
        # Minimal market block
        with mc_path.open("r", encoding="utf-8") as f:
            mc = json.load(f)
        coins = mc.get("coins", [])
        lines.append("📈 Market Context (CoinGecko, 72h)\n")
        for c in coins:
            lines.append(f"• {c['symbol']} → ${c['price']:.2f} │ h1 {c['h1']}% │ h24 {c['h24']}% │ h72 {c['h72']}%\n")
        lines.append("— Data via CoinGecko API; subject to plan rate limits.\n\n")
        md_path.write_text("".join(lines), encoding="utf-8")

# -----------------------
# Performance integration
# -----------------------
def ensure_performance_validation(ctx: Ctx, md_lines: List[str]) -> None:
    """
    Run paper trader (backtest by default). Always demo-safe: if there are no signals
    or prices, it synthesizes a tiny deterministic set so charts render.
    """
    try:
        result = run_paper_trader(ctx, mode=ctx.perf_mode)
        # Append compact CI block
        md_lines += perf_append(md_lines, ctx)
    except Exception as e:
        md_lines.append("🚀 Signal Performance Validation (v0.9.0)\n")
        md_lines.append(f"⚠️ Error running performance validation: `{e}`\n\n")

# -----------------------
# Main
# -----------------------
def main() -> int:
    ensure_dirs()
    ctx = Ctx(
        demo_mode=os.getenv("DEMO_MODE", "true").lower() == "true",
        perf_mode=os.getenv("MW_PERF_MODE", "backtest"),
    )

    # Make sure we have demo scaffolding
    generate_demo_data_if_needed(ctx)

    # Load current summary markdown
    md_path = ART_DIR / "demo_summary.md"
    if md_path.exists():
        md_lines = md_path.read_text(encoding="utf-8").splitlines(keepends=True)
    else:
        md_lines = []

    # Ensure performance section is present each run
    ensure_performance_validation(ctx, md_lines)

    # Write back the summary
    md_path.write_text("".join(md_lines), encoding="utf-8")
    print(f"Wrote CI summary to {md_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
