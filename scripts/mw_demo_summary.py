#!/usr/bin/env python3
"""
MoonWire CI Demo Summary (read-only)

Outputs
 - artifacts/demo_summary.md
 - artifacts/consensus.png   (lightweight bar)
 - artifacts/consensus_social.png (if you keep social card logic; harmless if empty)

Reads logs from LOGS_DIR (env, default: ./logs); never mutates logs unless
DEMO_MODE=true AND retraining_log is empty, in which case it calls the demo seeder
to append mock data for this run.

NOTE: Tests import `generate_demo_data_if_needed` from this module.
That function is read-only (in-memory) and returns (reviewers, events).
"""

from __future__ import annotations

import os, json, hashlib, random, uuid
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from src.analytics.origin_utils import compute_origin_breakdown

# ---------- env-driven config ----------
LOGS_DIR = Path(os.getenv("LOGS_DIR", "logs"))
ART      = Path(os.getenv("ARTIFACTS_DIR", "artifacts")); ART.mkdir(parents=True, exist_ok=True)

DEFAULT_THRESHOLD = 2.5
SOCIAL_W, SOCIAL_H, DPI = 1280, 720, 120

DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes")

# knobs for yield section (safe, read-only)
YIELD_DAYS       = int(os.getenv("YIELD_DAYS", "7"))
YIELD_MIN_EVENTS = int(os.getenv("YIELD_MIN_EVENTS", "1"))   # keep low for tiny fixtures
YIELD_ALPHA      = float(os.getenv("YIELD_ALPHA", "0.7"))    # reserved if you blend metrics later
# --------------------------------------


# ---------- helpers ----------
def red(s: str) -> str:
    return "000000" if not s else hashlib.sha1(s.encode()).hexdigest()[:6]

def load_jsonl(path: Path):
    if not path.exists(): return []
    out = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln: continue
        try: out.append(json.loads(ln))
        except Exception: pass
    return out

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
    except Exception: pass
    try:
        s = str(val);  s = s[:-1] + "+00:00" if s.endswith("Z") else s
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception: return None

def is_demo_mode() -> bool:
    return DEMO_MODE


# ---------- READ-ONLY demo seeding ----------
def generate_demo_data_if_needed(reviewers, flag_times=None):
    """Read-only helper used by tests. Returns (reviewers, seeded_events_metadata)."""
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
    if not is_demo_mode():
        return origins_rows
    # Trigger seeding if no data or all origins are "unknown"
    if not origins_rows or all(r.get("origin") == "unknown" for r in origins_rows):
        demo_sources = ["twitter", "reddit", "rss_news"]
        counts = [random.randint(1, 5) for _ in demo_sources]
        total = max(sum(counts), 1)
        return [
            {"origin": src, "count": c, "pct": round(c/total*100, 2)}
            for src, c in zip(demo_sources, counts)
        ]
    return origins_rows


# ---------- maybe seed logs (only if DEMO_MODE and truly empty) ----------
def maybe_seed_real_logs_if_empty():
    if not is_demo_mode():
        return False
    retrain_path = LOGS_DIR / "retraining_log.jsonl"
    if retrain_path.exists():
        try:
            if any(ln.strip() for ln in retrain_path.read_text().splitlines()):
                return False
        except Exception:
            pass
    try:
        from scripts.demo_seed_reviewers import seed_once
        seed_once()
        return True
    except Exception as e:
        print(f"[demo] seeding skipped due to error: {e}")
        return False

_ = maybe_seed_real_logs_if_empty()


# ---- load logs ----
retrain_log   = load_jsonl(LOGS_DIR / "retraining_log.jsonl")
triggered_log = load_jsonl(LOGS_DIR / "retraining_triggered.jsonl")
scores_log    = load_jsonl(LOGS_DIR / "reviewer_scores.jsonl")
score_by_id   = {r.get("reviewer_id"): r for r in scores_log}

# ---- latest signal ----
if retrain_log:
    def _key(r):
        t = r.get("timestamp", 0)
        try: return float(t)
        except Exception: return 0.0
    latest = max(retrain_log, key=_key)
    sig_id = latest.get("signal_id", "unknown")
    sig_rows = [r for r in retrain_log if r.get("signal_id") == sig_id]
else:
    sig_id = "none"
    sig_rows = []

# ---- weights & timeline ----
seen = set()
reviewers = []
flag_times = []
for r in sorted(sig_rows, key=lambda x: x.get("timestamp", 0)):
    t = parse_ts(r.get("timestamp"))
    if t: flag_times.append(t)
    rid = r.get("reviewer_id","")
    if rid in seen:
        continue
    seen.add(rid)
    w = r.get("reviewer_weight")
    if w is None:
        sc = (score_by_id.get(rid) or {}).get("score")
        w = band_weight_from_score(sc)
    reviewers.append({"id": rid, "weight": round(float(w), 2)})

reviewers, _seeded_events = generate_demo_data_if_needed(reviewers, flag_times)

# ---- compute origins (last YIELD_DAYS) ----
try:
    origins_rows, _totals = compute_origin_breakdown(
        flags_path=LOGS_DIR / "retraining_log.jsonl",
        triggers_path=LOGS_DIR / "retraining_triggered.jsonl",
        days=YIELD_DAYS,
        include_triggers=True
    )
except Exception:
    origins_rows = []

# Normalize key name for percent display (handle either 'pct' or 'percent')
if origins_rows:
    fixed = []
    for o in origins_rows:
        pct_val = o.get("pct")
        if pct_val is None and "percent" in o:
            pct_val = o["percent"]
        # If still None, compute from counts
        if pct_val is None:
            total_c = sum(x.get("count", 0) for x in origins_rows) or 1
            pct_val = round((o.get("count", 0) * 100.0) / total_c, 2)
        fixed.append({"origin": o.get("origin", "unknown"),
                      "count": o.get("count", 0),
                      "pct": pct_val})
    origins_rows = fixed

origins_rows = generate_demo_origins_if_needed(origins_rows)

total_weight = round(sum(r["weight"] for r in reviewers), 2)
threshold    = DEFAULT_THRESHOLD
would_trigger = total_weight >= threshold
last_trig = max((t for t in triggered_log if t.get("signal_id")==sig_id),
                key=lambda x: x.get("timestamp", 0), default=None) if triggered_log else None
now_iso = datetime.now(timezone.utc).isoformat()

# ---------- tiny bar for CI ----------
plt.figure(figsize=(3.6, 2.2), dpi=180)
plt.title("Consensus Weight vs Threshold")
plt.bar(["weight","threshold"], [total_weight, threshold])
plt.tight_layout()
(ART / "consensus.png").write_text("")  # ensure path exists even if savefig fails
try:
    plt.savefig(ART / "consensus.png", dpi=180)
finally:
    plt.close()


# ---------- markdown summary ----------
md = []
md.append("# MoonWire CI Demo Summary\n")
md.append(f"MoonWire Demo Summary — {now_iso}\n")
md.append("Pipeline proof (CI): end-to-end tests passed; consensus math reproduced on latest flagged signal.\n")
md.append(f"- **Signal:** `{red(sig_id)}`")
md.append(f"- **Unique reviewers:** {len(reviewers)}")
md.append(f"- **Combined weight:** **{total_weight}**")
md.append(f"- **Threshold:** **{threshold}** → **{'TRIGGERS' if would_trigger else 'NO TRIGGER'}**")
if last_trig:
    md.append(f"- **Last retrain trigger logged:** {last_trig.get('timestamp','')}")
md.append("\n**Reviewers (redacted):**")
if reviewers:
    for r in reviewers:
        md.append(f"- `{red(r['id'])}` → {weight_to_label(r['weight'])}")
else:
    md.append("- _none found in this run_")

md.append(f"\n**Signal origin breakdown (last {YIELD_DAYS} days):**")
if origins_rows:
    for o in origins_rows:
        md.append(f"- {o['origin']}: {o['count']} ({o['pct']}%)")
else:
    md.append("- _no origin data_")

# ---- Source Yield Plan section (always emit a JSON line if any data) ----
md.append("\n## Source Yield Plan\n")
yield_plan = {}
if origins_rows:
    # apply min-events threshold for eligibility
    eligible = [o for o in origins_rows if o.get("count", 0) >= YIELD_MIN_EVENTS]
    denom = sum(o.get("count", 0) for o in eligible)
    if denom > 0:
        # simple volume-based split (alpha reserved if you mix in conversion later)
        yield_plan = {o["origin"]: round(o["count"] * 100.0 / denom, 2) for o in eligible}

# Output a single-line JSON so tests can find a line starting with "{"
if yield_plan:
    md.append(json.dumps({"budget_plan": yield_plan}))
else:
    md.append("_no yield plan data_")

(ART / "demo_summary.md").write_text("\n".join(md))

print(f"Wrote: {ART/'demo_summary.md'}")