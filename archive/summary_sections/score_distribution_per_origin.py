# scripts/summary_sections/score_distribution_per_origin.py
"""
Score Distribution by Origin (48h) with drift overlay.

- Reads recent scores from models/trigger_history.jsonl
- Splits by origin, then by drifted vs non-drifted
- Renders overlaid histograms per origin
- Writes images to artifacts/score_hist_<origin>_overlay.png
- Embeds image **basenames** in markdown (because demo_summary.md lives in artifacts/)
- Seeds plausible demo data when logs are sparse

Env knobs:
- MW_SCORE_WINDOW_H (default 48)
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import math
import random
import statistics as stats
from typing import Dict, List, Tuple

import matplotlib
# CI sets MPLBACKEND=Agg, but be defensive:
matplotlib.use("Agg")  # type: ignore
import matplotlib.pyplot as plt  # noqa: E402

from .common import SummaryContext

# ---------- constants ----------
ART_DIR = Path("artifacts")
ART_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_WINDOW_H = 48  # default 48h


# ---------- helpers ----------
def _parse_ts(val) -> datetime | None:
    if val is None:
        return None
    try:
        return datetime.fromtimestamp(float(val), tz=timezone.utc)
    except Exception:
        pass
    try:
        s = str(val)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None


def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out


def _score_of_row(r: dict) -> float | None:
    # Prefer adjusted probability-like fields
    for k in ("adjusted_score", "prob", "prob_trigger_next_6h", "score"):
        if k in r:
            try:
                v = float(r[k])
                if math.isfinite(v):
                    return v
            except Exception:
                pass
    return None


def _is_drifted(r: dict) -> bool:
    # Heuristics used elsewhere in the repo
    if isinstance(r.get("drifted_features"), list) and len(r["drifted_features"]) > 0:
        return True
    for k in ("drift", "drifted"):
        if k in r:
            try:
                return bool(r[k])
            except Exception:
                pass
    return False


def _quantile(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs2 = sorted(xs)
    i = q * (len(xs2) - 1)
    lo = int(math.floor(i))
    hi = int(math.ceil(i))
    if lo == hi:
        return float(xs2[lo])
    w = i - lo
    return float(xs2[lo] * (1 - w) + xs2[hi] * w)


def _safe_name(s: str) -> str:
    s = s or "unknown"
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in s)


# TEST VISIBILITY: some tests import `_slug` from this module
def _slug(origin: str) -> str:
    """Return a filesystem/URL-safe slug for an origin."""
    return _safe_name(origin)


def _seed_demo_for_origin(origin: str, n: int = 40) -> Tuple[List[float], List[float]]:
    """
    Create plausible demo distributions:
      - non-drifted centered ~0.25–0.35
      - drifted slightly lower mean and broader spread
    """
    random.seed(hash(origin) % (2**32 - 1))
    base_mu = random.uniform(0.20, 0.32)
    base_sigma = random.uniform(0.05, 0.08)
    drift_mu = max(0.02, base_mu - random.uniform(0.02, 0.08))
    drift_sigma = base_sigma + random.uniform(0.01, 0.05)

    n_drift = max(8, n // 3)
    n_nodrift = max(10, n - n_drift)

    def _clip01(v: float) -> float:
        return max(0.0, min(1.0, v))

    nodrift = [_clip01(random.gauss(base_mu, base_sigma)) for _ in range(n_nodrift)]
    drifted = [_clip01(random.gauss(drift_mu, drift_sigma)) for _ in range(n_drift)]
    return nodrift, drifted


# ---------- core ----------
def _load_recent_by_origin(ctx: SummaryContext, window_h: int) -> Dict[str, Dict[str, List[float]]]:
    """
    Returns: {origin: {"drifted": [scores], "non_drifted": [scores]}}
    """
    cache_key = f"triggers_by_origin_{window_h}h"
    if cache_key in ctx.caches:
        return ctx.caches[cache_key]

    hist_path = ctx.models_dir / "trigger_history.jsonl"
    rows = _load_jsonl(hist_path)

    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_h)
    out: Dict[str, Dict[str, List[float]]] = {}

    for r in rows:
        ts = _parse_ts(r.get("timestamp"))
        if not ts or ts < cutoff:
            continue
        origin = r.get("origin") or "unknown"
        score = _score_of_row(r)
        if score is None:
            continue
        bucket = "drifted" if _is_drifted(r) else "non_drifted"
        d = out.setdefault(origin, {"drifted": [], "non_drifted": []})
        d[bucket].append(float(score))

    ctx.caches[cache_key] = out
    return out


def _summarize(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {
        "n": len(xs),
        "mean": float(stats.fmean(xs)) if xs else float("nan"),
        "median": float(stats.median(xs)) if xs else float("nan"),
        "p90": float(_quantile(xs, 0.90)) if xs else float("nan"),
    }


def _render_hist(origin: str, nodrift: List[float], drifted: List[float]) -> str:
    """
    Save figure to artifacts/score_hist_<origin>_overlay.png
    Return the **basename** to be embedded in markdown.
    """
    safe = _slug(origin)
    fname = f"score_hist_{safe}_overlay.png"
    out_path = ART_DIR / fname

    plt.figure(figsize=(6.0, 2.6))
    bins = 12
    if nodrift:
        plt.hist(nodrift, bins=bins, alpha=0.7, label="non-drifted")
    if drifted:
        plt.hist(drifted, bins=bins, alpha=0.7, label="drifted")
    plt.title(f"{origin} scores (48h)")
    plt.xlabel("score")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return fname  # basename so it renders from inside artifacts/demo_summary.md


def append(md: List[str], ctx: SummaryContext):
    """
    Public entry point used by the orchestrator.
    """
    try:
        window_h = int(float((__import__("os").getenv("MW_SCORE_WINDOW_H") or DEFAULT_WINDOW_H)))
    except Exception:
        window_h = DEFAULT_WINDOW_H

    by_origin = _load_recent_by_origin(ctx, window_h)

    seeded_demo = False
    if not by_origin and ctx.is_demo:
        # Seed 2–3 origins with plausible shapes
        seeded_demo = True
        for origin in ("twitter", "reddit", "rss_news"):
            nd, dr = _seed_demo_for_origin(origin, n=random.randint(28, 52))
            by_origin[origin] = {"non_drifted": nd, "drifted": dr}

    # Header
    hdr = f"\n### 📐 Score Distribution by Origin ({window_h}h)"
    if seeded_demo:
        hdr += "\n_(demo)_"
    md.append(hdr)

    if not by_origin:
        md.append("_No recent scores available._")
        return

    # Stable order: sort by origin name
    for origin in sorted(by_origin.keys()):
        nodrift = list(by_origin[origin].get("non_drifted", []))
        drifted = list(by_origin[origin].get("drifted", []))
        all_scores = nodrift + drifted

        # Stats
        s_all = _summarize(all_scores)
        s_nd = _summarize(nodrift)
        s_dr = _summarize(drifted)

        # delta between group means (non-drifted - drifted)
        delta = float("nan")
        if s_nd["n"] > 0 and s_dr["n"] > 0 and math.isfinite(s_nd["mean"]) and math.isfinite(s_dr["mean"]):
            delta = s_nd["mean"] - s_dr["mean"]

        # Plot
        img_name = _render_hist(origin, nodrift, drifted)

        # Markdown block for this origin
        md.append(f"\n- **{origin}**")
        md.append(
            f"  - n={s_all['n']}, mean={s_all['mean']:.3f}, median={s_all['median']:.3f}, p90={s_all['p90']:.3f}"
        )
        md.append(f"  - split: drifted={s_dr['n']} | non-drifted={s_nd['n']}")
        if math.isfinite(delta):
            md.append(
                f"  - group means: drifted={s_dr['mean']:.3f}, non-drifted={s_nd['mean']:.3f}, Δ={delta:.3f}"
            )
        else:
            if s_dr["n"] > 0:
                md.append(f"  - drifted mean={s_dr['mean']:.3f}")
            if s_nd["n"] > 0:
                md.append(f"  - non-drifted mean={s_nd['mean']:.3f}")

        # IMPORTANT: embed **basename** so it renders from within artifacts/demo_summary.md
        md.append(f"  \n![]({img_name})")