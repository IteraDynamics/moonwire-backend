# scripts/summary_sections/trigger_explainability.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")  # ensure headless
import matplotlib.pyplot as plt  # noqa: E402

from .common import SummaryContext, ensure_dir, parse_ts, _iso

SAMPLE_JSON = "models/explainability_sample.json"
PLOT_PNG = "artifacts/explainability_top_features.png"

def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None

def _write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2))

def _seed_demo_sample(now: datetime) -> Dict[str, Any]:
    # Deterministic demo payload with a few fired triggers and top features per origin
    rnd = [
        {
            "ts": _iso(now),
            "origin": "reddit",
            "model_version": "v18",
            "adjusted_score": 0.87,
            "decision": "fire",
            "explanation": [
                {"feature": "btc_return_1h", "contribution": +0.21},
                {"feature": "reddit_burst_etf", "contribution": +0.17},
                {"feature": "volatility_6h", "contribution": +0.11},
            ],
        },
        {
            "ts": _iso(now),
            "origin": "twitter",
            "model_version": "v12",
            "adjusted_score": 0.81,
            "decision": "fire",
            "explanation": [
                {"feature": "solana_sentiment", "contribution": +0.23},
                {"feature": "volatility_6h", "contribution": +0.12},
                {"feature": "btc_return_1h", "contribution": +0.07},
            ],
        },
        {
            "ts": _iso(now),
            "origin": "rss_news",
            "model_version": "v09",
            "adjusted_score": 0.76,
            "decision": "fire",
            "explanation": [
                {"feature": "sec_approval_term", "contribution": +0.19},
                {"feature": "btc_price_jump", "contribution": +0.15},
                {"feature": "volatility_6h", "contribution": +0.08},
            ],
        },
    ]
    return {
        "generated_at": _iso(now),
        "window_hours": 72,
        "sample_size": len(rnd),
        "rows": rnd,
        "demo": True,
    }

def _summarize_top_features(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, List[str]], Counter]:
    """
    Returns:
      - per_origin_top: origin -> list of most frequent top features (up to 3)
      - global_counts: Counter over all features
    """
    per_origin_counts: Dict[str, Counter] = defaultdict(Counter)
    global_counts: Counter = Counter()
    for r in rows:
        if (r.get("decision") or "").lower() != "fire":
            continue
        origin = r.get("origin") or "unknown"
        expl = r.get("explanation") or []
        # take top-K from explanation order if available
        feats = []
        for item in expl:
            f = (item.get("feature") or "").strip()
            if f:
                feats.append(f)
        if not feats:
            continue
        # only count top 3 per row to reduce noise
        for f in feats[:3]:
            per_origin_counts[origin][f] += 1
            global_counts[f] += 1

    per_origin_top: Dict[str, List[str]] = {}
    for origin, cnt in per_origin_counts.items():
        per_origin_top[origin] = [f for f, _c in cnt.most_common(3)]
    return per_origin_top, global_counts

def _plot_global_top(global_counts: Counter, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    if not global_counts:
        # create an empty placeholder so CI has a file
        plt.figure(figsize=(6, 3))
        plt.title("Top features (no data)")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        return

    feats, counts = zip(*global_counts.most_common(10))
    plt.figure(figsize=(8, 4.5))
    plt.barh(list(feats)[::-1], list(counts)[::-1])  # no custom colors per guidelines
    plt.title("Most common top features (last 72h)")
    plt.xlabel("count in top-3 (across fired triggers)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Render 'Trigger Explainability' section.
    Reads models/explainability_sample.json if present; in demo, seeds a deterministic sample.
    Emits a global top-features plot and per-origin top lists.
    """
    models_dir = Path(getattr(ctx, "models_dir", "models"))
    artifacts_dir = Path(getattr(ctx, "artifacts_dir", "artifacts"))
    sample_path = models_dir / SAMPLE_JSON.split("/", 1)[-1]
    plot_path = artifacts_dir / PLOT_PNG.split("/", 1)[-1]

    # Try to load sample JSON
    data = _load_json(sample_path)

    # If missing and DEMO, seed deterministic sample
    if (not data or not isinstance(data, dict) or "rows" not in data) and (
        str(os.getenv("MW_DEMO") or os.getenv("DEMO_MODE") or "").lower() == "true"
        or getattr(ctx, "is_demo", False)
    ):
        now = datetime.now(timezone.utc).replace(microsecond=0)
        data = _seed_demo_sample(now)
        _write_json(sample_path, data)

    # If still nothing, render an informative skip
    if not data or not isinstance(data, dict) or "rows" not in data:
        md.append("\n> ⚠️ Trigger Explainability: no sample available (missing models/explainability_sample.json).")
        return

    rows = list(data.get("rows") or [])
    per_origin_top, global_counts = _summarize_top_features(rows)
    _plot_global_top(global_counts, plot_path)

    md.append("\n### 🔍 Trigger Explainability (last 72h)")
    if not rows:
        md.append("_no fired triggers in window_")
    else:
        # Print per-origin top features
        # keep a stable order: reddit, twitter, rss_news, then others
        preferred = ["reddit", "twitter", "rss_news"]
        seen = set()
        for o in preferred + sorted(set(per_origin_top.keys()) - set(preferred)):
            if o in seen or o not in per_origin_top:
                continue
            seen.add(o)
            tops = per_origin_top[o]
            if tops:
                md.append(f"{o}  → top: {', '.join(tops)}")
        # Footer
        demo_flag = " (demo)" if data.get("demo") else ""
        md.append(f"\n_Footer: Feature contributions estimated via model coefficients/importances{demo_flag}._")