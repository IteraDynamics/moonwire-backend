"""
Calibration & Reliability Trend vs Market + Social Bursts

Extends calibration trend analysis with:
- Market regimes (BTC returns/volatility).
- Social bursts (from Reddit context).

Artifacts:
- models/calibration_reliability_trend.json (enriched with market + bursts).
- artifacts/calibration_trend_ece.png
- artifacts/calibration_trend_brier.png
- Markdown CI summary.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime
from dateutil import parser as dateparser

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None

def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))

def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat() + "Z"

# ---------------------------------------------------------------------
# Main append
# ---------------------------------------------------------------------

def append(md: List[str], ctx) -> None:
    """
    Enrich calibration trend with market + social bursts,
    emit plots + markdown.
    """
    models_dir: Path = Path(ctx.models_dir)
    artifacts_dir: Path = Path(ctx.artifacts_dir)

    trend_path = models_dir / "calibration_reliability_trend.json"
    social_path = models_dir / "social_reddit_context.json"

    trend = _load_json(trend_path)
    if not trend:
        md.append("\n> ⚠️ Calibration trend missing.\n")
        return

    bursts: List[Dict[str, Any]] = []
    social = _load_json(social_path)
    if social and "bursts" in social:
        bursts = social["bursts"]

    # Enrich buckets
    series = trend.get("series", [])
    for s in series:
        for pt in s.get("points", []):
            pt.setdefault("alerts", [])
            pt["social_bursts"] = []

            b_start = pt.get("bucket_start")
            if not b_start:
                continue

            # find bursts in same interval
            matched = [b for b in bursts if b.get("bucket_start") == b_start]
            if matched:
                pt["social_bursts"].extend(matched)
                # add alert if both high_ece and burst
                if "high_ece" in pt.get("alerts", []):
                    if "social_burst_overlap" not in pt["alerts"]:
                        pt["alerts"].append("social_burst_overlap")

    _save_json(trend_path, trend)

    # Plots
    for metric in ("ece", "brier"):
        fig, ax = plt.subplots(figsize=(8, 3))
        for s in series:
            xs = [p.get("bucket_start") for p in s.get("points", [])]
            ys = [p.get(metric) for p in s.get("points", [])]
            ax.plot(xs, ys, marker="o", label=s.get("key", "series"))

            # overlay bursts
            for p in s.get("points", []):
                if p.get("social_bursts"):
                    x = p.get("bucket_start")
                    y = p.get(metric)
                    ax.scatter(x, y, marker="^", color="red", zorder=5)

        ax.set_title(f"Calibration trend ({metric}) with social bursts")
        ax.legend()
        fig.autofmt_xdate()
        out = artifacts_dir / f"calibration_trend_{metric}.png"
        fig.savefig(out)
        plt.close(fig)

    # Markdown
    md.append("### 🧮 Calibration & Reliability Trend vs Market + Social (72h)")
    for s in series:
        key = s.get("key")
        last = s.get("points", [])[-1] if s.get("points") else {}
        ece = last.get("ece")
        alerts = ", ".join(last.get("alerts", []))
        if last.get("social_bursts"):
            terms = [b.get("term") or "" for b in last["social_bursts"]]
            terms_str = ", ".join([t for t in terms if t])
            md.append(f"{key} → ECE {ece:.2f} [{alerts}] + bursts [{terms_str}]")
        else:
            md.append(f"{key} → ECE {ece:.2f} [{alerts}]")
