# scripts/perf/replay_shadow_to_paper.py
from __future__ import annotations

from pathlib import Path
import json
import sys

# Robust import path: prefer local scripts, fallback to src if you move it later
try:
    from scripts.perf.paper_trader import PaperTrader
except Exception:
    try:
        from src.perf.paper_trader import PaperTrader  # if you relocate later
    except Exception as e:
        print(f"[FATAL] Could not import PaperTrader: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    shadow = Path("logs/signal_inference_shadow.jsonl")
    trades_out = Path("logs/paper_trades.jsonl")
    summary_out = Path("models/paper_summary.json")

    # For now: accept 'shadow' + 'shadow_only_live_ml_candidate' as trade-like,
    # require at least some confidence (tune this later / read from governance if desired).
    trader = PaperTrader(accept_reasons={"shadow", "shadow_only_live_ml_candidate", "live_ml_executed"},
                         min_conf=0.50)

    summary = trader.replay_shadow(shadow, trades_out, summary_out)
    print("[paper] summary:", json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()