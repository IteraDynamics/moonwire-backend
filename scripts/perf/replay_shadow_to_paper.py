# scripts/perf/replay_shadow_to_paper.py
from __future__ import annotations
import json, time
from pathlib import Path
from scripts.perf.paper_trader import PaperTrader  # adjust import if path differs

LOG = Path("logs/signal_inference_shadow.jsonl")

def iter_shadow():
    if not LOG.exists():
        return
    for line in LOG.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if not row.get("ml_ok"):
            continue
        yield {
            "ts": row.get("ts"),
            "symbol": row.get("symbol"),
            "dir": row.get("ml_dir"),
            "conf": row.get("ml_conf"),
        }

def main():
    pt = PaperTrader(
        fees_bps=1, slippage_bps=2,
        conf_min_by_symbol=None,   # or load models/governance_params.json
    )
    for s in iter_shadow():
        pt.on_signal(s["symbol"], s["dir"], s["conf"], s["ts"])
    # persist PnL/equity to artifacts
    Path("artifacts").mkdir(exist_ok=True)
    pt.write_reports("artifacts/paper_")
    print("Paper trader replay complete.")

if __name__ == "__main__":
    main()
