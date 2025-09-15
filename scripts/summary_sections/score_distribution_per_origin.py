# scripts/summary_sections/score_distribution_per_origin.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone, timedelta
import os, json, math, statistics as stats

import matplotlib
# CI uses MPLBACKEND=Agg; this is harmless locally too.
matplotlib.use(os.getenv("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt

from src.paths import MODELS_DIR
from scripts.summary_sections.common import is_demo_mode

ART = Path("artifacts"); ART.mkdir(exist_ok=True)

# ---------- tiny utils ----------
def _parse_ts_any(v):
    if v is None:
        return None
    try:
        return datetime.fromtimestamp(float(v), tz=timezone.utc)
    except Exception:
        pass
    try:
        s = str(v)
        s = s[:-1] + "+00:00" if s.endswith("Z") else s
        dt = datetime.fromisoformat(s)
        return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None

def _load_jsonl_safe(p: Path) -> list:
    if not p.exists():
        return []
    out = []
    try:
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
    except Exception:
        return []
    return out

def _score_from_row(r: dict) -> float | None:
    for key in ("adjusted_score", "prob_trigger_next_6h", "score", "p"):
        if key in r:
            try:
                v = float(r[key])
                if math.isnan(v) or math.isinf(v):
                    continue
                # clamp to [0,1] just to be safe for hist scaling
                return max(0.0, min(1.0, v))
            except Exception:
                continue
    return None

def _is_drifted_row(r: dict) -> bool:
    try:
        if r.get("drift") is True or r.get("drifted") is True:
            return True
        df = r.get("drifted_features")
        if isinstance(df, (list, tuple)) and len(df) > 0:
            return True
    except Exception:
        pass
    return False

def _slug(s: str) -> str:
    s = (s or "unknown").lower()
    return "".join(ch if ch.isalnum() else "_" for ch in s)

def _pctl(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    vals_sorted = sorted(vals)
    idx = max(0, min(len(vals_sorted)-1, int(round(q * (len(vals_sorted)-1)))))
    return float(vals_sorted[idx])

def _fmt(v: float | None) -> str:
    try:
        return f"{float(v):.3f}"
    except Exception:
        return "n/a"

# ---------- core collector ----------
def _collect_scores_by_origin(models_dir: Path, hours: int) -> dict[str, dict[str, list[float]]]:
    """
    Returns: { origin: {"drifted": [...], "non": [...]} }
    """
    th_path = models_dir / "trigger_history.jsonl"
    rows = _load_jsonl_safe(th_path)

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)

    out: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        ts = _parse_ts_any(r.get("timestamp"))
        if ts is None or ts < cutoff:
            continue
        origin = str(r.get("origin") or "unknown").lower()
        sc = _score_from_row(r)
        if sc is None:
            continue
        drifted = _is_drifted_row(r)
        bucket = "drifted" if drifted else "non"
        d = out.setdefault(origin, {"drifted": [], "non": []})
        d[bucket].append(sc)

    # drop origins with no samples
    out = {o: d for o, d in out.items() if (d["drifted"] or d["non"])}

    return out

# ---------- demo seeding ----------
def _demo_seed() -> dict[str, dict[str, list[float]]]:
    # 3 origins with plausible splits & shapes
    import random
    random.seed(7)
    def gen(n, mu, sigma):
        xs = []
        for _ in range(n):
            v = random.gauss(mu, sigma)
            xs.append(max(0.0, min(1.0, v)))
        return xs

    return {
        "twitter":  {"drifted": gen(16, 0.18, 0.07), "non": gen(32, 0.30, 0.08)},
        "reddit":   {"drifted": gen(12, 0.22, 0.06), "non": gen(28, 0.26, 0.07)},
        "rss_news": {"drifted": gen(20, 0.14, 0.05), "non": gen(14, 0.20, 0.06)},
    }

# ---------- plotting ----------
def _plot_overlay(hist_drifted: list[float], hist_non: list[float], title: str, out_path: Path):
    # Single figure per origin; default color cycle; legend via labels.
    plt.figure(figsize=(5.6, 3.0))
    bins = 20
    if hist_non:
        plt.hist(hist_non, bins=bins, alpha=0.6, label="non-drifted")  # alpha only, no color set
    if hist_drifted:
        plt.hist(hist_drifted, bins=bins, alpha=0.6, label="drifted")
    plt.title(title)
    plt.xlabel("score")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()

# ---------- main entry ----------
def append(md: list[str], ctx) -> None:
    """
    Append a per-origin score overlay section to the CI summary.
    """
    hours = 48
    try:
        hours = int(os.getenv("MW_SCORE_WINDOW_H", "48"))
    except Exception:
        pass

    # cache key so other sections could reuse if desired
    cache_key = f"score_by_origin_{hours}h"
    if cache_key in ctx.caches:
        by_origin = ctx.caches[cache_key]
    else:
        by_origin = _collect_scores_by_origin(ctx.models_dir or MODELS_DIR, hours=hours)
        if (not by_origin or list(by_origin.keys()) == ["unknown"]) and is_demo_mode():
            by_origin = _demo_seed()
            ctx.caches[cache_key] = by_origin
            seeded = True
        else:
            ctx.caches[cache_key] = by_origin
            seeded = False

    md.append(f"\n### 📐 Score Distribution by Origin ({hours}h)")
    if not by_origin:
        md.append("_No recent scores available._")
        return
    if seeded:
        md.append("_(demo)_")

    # Sort origins by total n desc
    items = []
    for origin, parts in by_origin.items():
        n_d = len(parts.get("drifted", []))
        n_n = len(parts.get("non", []))
        n   = n_d + n_n
        items.append((origin, n))
    items.sort(key=lambda kv: kv[1], reverse=True)

    for origin, _ in items:
        v_d = list(by_origin[origin].get("drifted", []))
        v_n = list(by_origin[origin].get("non", []))
        v_all = v_d + v_n
        if not v_all:
            continue

        n = len(v_all)
        mean = sum(v_all) / n
        med = stats.median(v_all)
        p90 = _pctl(v_all, 0.90)

        mean_d = (sum(v_d)/len(v_d)) if v_d else None
        mean_n = (sum(v_n)/len(v_n)) if v_n else None
        delta = None
        if (mean_d is not None) and (mean_n is not None):
            delta = mean_n - mean_d

        # Plot
        slug = _slug(origin)
        out_img = ART / f"score_hist_{slug}_overlay.png"
        _plot_overlay(v_d, v_n, f"{origin}: scores ({hours}h)", out_img)

        # Markdown block for this origin
        md.append(f"- **{origin}**")
        md.append(
            f"  - n={n}, mean={_fmt(mean)}, median={_fmt(med)}, p90={_fmt(p90)}"
        )
        md.append(
            f"  - split: drifted={len(v_d)} | non-drifted={len(v_n)}"
        )
        if (mean_d is not None) or (mean_n is not None):
            md.append(
                f"  - group means: drifted={_fmt(mean_d)}, non-drifted={_fmt(mean_n)}, Δ={_fmt(delta)}"
            )
        md.append(f"  - ![]({out_img.as_posix()})")