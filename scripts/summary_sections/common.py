# scripts/summary_sections/common.py
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional
import os, json, hashlib, random, uuid

# ---- Context passed to every section ----
@dataclass
class SummaryContext:
    logs_dir: Path
    models_dir: Path
    is_demo: bool
    origins_rows: list = field(default_factory=list)
    yield_data: Optional[dict] = None
    candidates: list[str] = field(default_factory=list)
    caches: dict = field(default_factory=dict)   # sections may reuse/store computed data

# ---- Generic helpers (moved out of mw_demo_summary) ----
def is_demo_mode() -> bool:
    return os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes")

def red(s: str) -> str:
    return "000000" if not s else hashlib.sha1(s.encode()).hexdigest()[:6]

def band_weight_from_score(score):
    if score is None: return 1.0
    if score >= 0.75: return 1.25
    if score >= 0.50: return 1.0
    return 0.75

def weight_to_label(w: float) -> str:
    if w >= 1.20: return "High"
    if w >= 0.90: return "Med"
    return "Low"

def parse_ts(val):
    if val is None: return None
    try:
        ts = float(val); return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        pass
    try:
        s = str(val);  s = s[:-1] + "+00:00" if s.endswith("Z") else s
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None

def pick_candidate_origins(origins_rows, yield_data=None, top=3, default=("twitter","reddit","rss_news")):
    seen, out = set(), []
    # Prefer yield plan ordering if available
    if yield_data:
        for item in yield_data.get("budget_plan", []) or []:
            o = item.get("origin")
            if o and o != "unknown" and o not in seen:
                out.append(o); seen.add(o)
                if len(out) >= top: return out
    # Then origin breakdown
    for row in origins_rows or []:
        o = row.get("origin")
        if o and o != "unknown" and o not in seen:
            out.append(o); seen.add(o)
            if len(out) >= top: return out
    # Then sane defaults
    for o in default:
        if o not in seen:
            out.append(o); seen.add(o)
            if len(out) >= top: break
    return out[:top]
    
def _build_summary_features_for_origin(
    origin: str,
    trends_by_origin: dict[str, list] | None = None,
    regimes_map: dict | None = None,   # can be str OR {"regime": ...}
    metrics_map: dict[str, dict] | None = None,
    bursts_by_origin: dict[str, list] | None = None,
) -> dict:
    """
    Build a compact feature vector for one origin from summary analytics:
      - rolling counts over last 1/6/24/72 buckets (flags_count / flags / count)
      - latest burst z-score
      - regime one-hot (calm/normal/turbulent)
      - precision_7d / recall_7d
      - leadership_max_r placeholder (sections may overwrite)
    """
    series = (trends_by_origin or {}).get(origin, []) or []

    # Determine chronological order; then take the last k (most recent) buckets.
    def _series_latest_k(k: int) -> list:
        if not series:
            return []
        try:
            first = series[0].get("timestamp_bucket")
            last  = series[-1].get("timestamp_bucket")
            asc = (str(first) <= str(last))
        except Exception:
            asc = True
        s = series if asc else list(reversed(series))
        return s[-k:] if k <= len(s) else s

    def _flags_from_bucket(b: dict) -> float:
        # accept multiple key names to be robust across sources/demos
        v = b.get("flags_count")
        if v is None: v = b.get("flags")
        if v is None: v = b.get("count")
        try:
            return float(v or 0.0)
        except Exception:
            return 0.0

    def sum_last(k: int) -> float:
        buckets = _series_latest_k(k)
        return float(sum(_flags_from_bucket(x) for x in buckets))

    feats = {
        "count_1h":  sum_last(1),
        "count_6h":  sum_last(6),
        "count_24h": sum_last(24),
        "count_72h": sum_last(72),
        "burst_z":   0.0,
        "regime_calm": 0.0,
        "regime_normal": 0.0,
        "regime_turbulent": 0.0,
        "precision_7d": 0.0,
        "recall_7d": 0.0,
        "leadership_max_r": 0.0,  # another section may set this
    }

    # Latest burst z
    bursts = (bursts_by_origin or {}).get(origin) or []
    if bursts:
        try:
            feats["burst_z"] = float((bursts[-1] or {}).get("z_score", 0.0) or 0.0)
        except Exception:
            pass

    # Regime (handles str or dict{"regime": ...})
    raw_reg = (regimes_map or {}).get(origin)
    if isinstance(raw_reg, dict):
        regime = (raw_reg.get("regime") or "").strip().lower()
    elif isinstance(raw_reg, str):
        regime = raw_reg.strip().lower()
    else:
        regime = ""
    if regime in ("calm", "normal", "turbulent"):
        feats[f"regime_{regime}"] = 1.0

    # Precision/recall (7d)
    m = (metrics_map or {}).get(origin) or {}
    try:
        feats["precision_7d"] = float(m.get("precision", 0.0) or 0.0)
        feats["recall_7d"]    = float(m.get("recall", 0.0) or 0.0)
    except Exception:
        pass

    return feats

# --- DEMO seeders used by a few sections ---
def generate_demo_data_if_needed(reviewers, flag_times=None):
    flag_times = flag_times or []
    if not is_demo_mode() or reviewers:
        return reviewers, []
    now = datetime.now(timezone.utc)
    n = random.randint(3, 5)
    choices = [0.75, 1.0, 1.25]
    seeded, display = [], []
    for _ in range(n):
        rid = f"demo-{uuid.uuid4().hex[:8]}"
        w = random.choice(choices)
        ts = (now - timedelta(minutes=random.randint(2, 55)))
        seeded.append({"id": rid, "weight": w, "timestamp": ts.isoformat()})
        display.append({"id": rid, "weight": w})
        flag_times.append(ts)
    return display, seeded

def generate_demo_origins_if_needed(origins_rows):
    """
    If DEMO_MODE is on and origins are empty/unknown, seed a simple breakdown
    so CI demo runs never render empty sections.
    """
    if not is_demo_mode():
        return origins_rows
    rows = origins_rows or []
    if not rows or all((r.get("origin") == "unknown") for r in rows):
        demo_sources = ["twitter", "reddit", "rss_news"]
        counts = [random.randint(1, 5) for _ in demo_sources]
        total = max(sum(counts), 1)
        return [
            {"origin": src, "count": c, "percent": round(c/total*100, 1)}
            for src, c in zip(demo_sources, counts)
        ]
    return rows

# Optional: tiny JSONL loader some sections may want
def load_jsonl_safe(path: Path):
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

__all__ = [
    "SummaryContext",
    "is_demo_mode",
    "red",
    "band_weight_from_score",
    "weight_to_label",
    "parse_ts",
    "pick_candidate_origins",
    "generate_demo_data_if_needed",
    "generate_demo_origins_if_needed",
    "load_jsonl_safe",
]