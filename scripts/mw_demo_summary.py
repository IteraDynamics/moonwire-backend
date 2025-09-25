# scripts/mw_demo_summary.py
from __future__ import annotations

import os
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Tuple


# ------------------------------
# Demo data seeding (test-backed)
# ------------------------------

def _ziso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def generate_demo_data_if_needed(origins_rows: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Return (reviewers, events).

    Contract required by tests:
      - If DEMO_MODE != "true": return ([], []).
      - If DEMO_MODE == "true" and `origins_rows` is non-empty (real reviewers): return (origins_rows, []).
      - If DEMO_MODE == "true" and no reviewers provided: synthesize 3 reviewers and
        emit one event per reviewer (len(events) == len(reviewers)).
    """
    demo_mode = str(os.getenv("DEMO_MODE", "")).lower() == "true"
    if not demo_mode:
        return [], []

    if origins_rows:
        # In demo mode but real reviewers were passed in -> do not synthesize events.
        return list(origins_rows), []

    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    reviewers = [
        {
            "id": "rev_demo_1",
            "origin": "reddit",
            "score": 0.82,
            "timestamp": _ziso(now - timedelta(hours=2)),
        },
        {
            "id": "rev_demo_2",
            "origin": "rss_news",
            "score": 0.31,
            "timestamp": _ziso(now - timedelta(hours=1)),
        },
        {
            "id": "rev_demo_3",
            "origin": "twitter",
            "score": 0.67,
            "timestamp": _ziso(now),
        },
    ]
    events = [
        {
            "type": "demo_review_created",
            "at": r["timestamp"],
            "meta": {"version": "v0.6.6", "note": "seeded in demo mode"},
            "review_id": r["id"],
        }
        for r in reviewers
    ]
    return reviewers, events


# --------------------------------
# Minimal CI summary file generator
# --------------------------------

def _read_json_safe(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def build_demo_summary(models_dir: Path = Path("models"), artifacts_dir: Path = Path("artifacts")) -> Path:
    """
    Create a lightweight CI markdown summary at artifacts/demo_summary.md.
    This never raises on missing files; it simply reports what exists.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    out_md = artifacts_dir / "demo_summary.md"

    # Collect a few optional model artifacts for display
    sections: List[str] = []
    sections.append("### MoonWire CI Demo Summary\n")

    # Market context (optional)
    mc_path = models_dir / "market_context.json"
    mc = _read_json_safe(mc_path)
    if mc:
        returns = mc.get("returns", {})
        coins = mc.get("coins", [])
        lines = []
        for sym in coins:
            r = returns.get(sym, {})
            if isinstance(r, dict):
                lines.append(f"- {sym}: h1 {r.get('h1','n/a')} | h24 {r.get('h24','n/a')} | h72 {r.get('h72','n/a')}")
        if lines:
            sections.append("#### 📈 Market Context (CoinGecko)\n" + "\n".join(lines) + "\n")

    # Calibration trend (optional)
    crt_path = models_dir / "calibration_reliability_trend.json"
    crt = _read_json_safe(crt_path)
    if crt and isinstance(crt, dict):
        dim = crt.get("meta", {}).get("dim", "origin")
        series = crt.get("series", [])
        bullets = []
        for s in series[:3]:
            key = s.get("key", "unknown")
            pts = s.get("points", [])
            if pts:
                last = pts[-1]
                bullets.append(f"- {key}: ECE {last.get('ece','n/a')} (n={last.get('n','?')})")
        if bullets:
            sections.append("#### 🧮 Calibration Trend (latest)\n" + "\n".join(bullets) + "\n")

    # List available PNG plots in artifacts
    pngs = sorted([p for p in artifacts_dir.glob("*.png")])
    if pngs:
        sections.append("#### 🖼️ Plots available\n" + "\n".join(f"- {p.name}" for p in pngs) + "\n")

    # Fallback if nothing else
    if len(sections) == 1:
        sections.append("_No artifacts found yet; this is a minimal summary stub._\n")

    out_md.write_text("\n".join(sections))
    return out_md


def _main() -> None:
    # Keep demo seeding behavior (used by tests) and also ensure the CI summary exists.
    reviewers, events = generate_demo_data_if_needed([])
    _ = build_demo_summary(Path("models"), Path("artifacts"))
    demo = "true" if reviewers or events else "false"
    print(
        f"[mw_demo_summary] DEMO_MODE derived output present: {demo} "
        f"(reviewers={len(reviewers)}, events={len(events)})"
    )


if __name__ == "__main__":
    _main()