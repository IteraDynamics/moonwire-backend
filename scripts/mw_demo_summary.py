#!/usr/bin/env python3
"""
MoonWire CI Demo Summary (read-only)

Outputs:
 - artifacts/demo_summary.md
 - artifacts/consensus.png
 - artifacts/consensus_social.png
 - artifacts/origin_breakdown.png
 - (new) Source Yield Plan section

Reads ./logs/*.jsonl; never mutates logs unless DEMO_MODE=true AND retraining log is empty.
"""

import os, json, hashlib, random, uuid
from pathlib import Path
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt

# ---------- config ----------
LOGS_DIR = Path(os.getenv("LOGS_DIR", "logs"))
ART = Path("artifacts"); ART.mkdir(exist_ok=True)
DEFAULT_THRESHOLD = 2.5
# ----------------------------

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
    return os.getenv("DEMO_MODE", "false").lower() in ("1","true","yes")

# ---------- READ-ONLY demo seeding ----------
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

total_weight = round(sum(r["weight"] for r in reviewers), 2)
threshold    = DEFAULT_THRESHOLD
would_trigger = total_weight >= threshold
last_trig = max((t for t in triggered_log if t.get("signal_id")==sig_id),
                key=lambda x: x.get("timestamp", 0), default=None) if triggered_log else None
now_iso = datetime.now(timezone.utc).isoformat()

# ---------- small CI bar ----------
plt.figure(figsize=(3.8, 2.4), dpi=200)
plt.title("Consensus Weight vs Threshold")
plt.bar(["weight","threshold"], [total_weight, threshold])
plt.tight_layout()
small_png = ART / "consensus.png"
plt.savefig(small_png, dpi=200)
plt.close()

# ---------- origin breakdown graphic ----------
def draw_origin_breakdown():
    try:
        from src.analytics.origin_utils import compute_origin_breakdown
        from src.paths import RETRAINING_LOG_PATH, RETRAINING_TRIGGERED_LOG_PATH
        rows, totals = compute_origin_breakdown(RETRAINING_LOG_PATH, RETRAINING_TRIGGERED_LOG_PATH, days=7, include_triggers=False)
        if not rows:
            return None
        origins = [r["origin"] for r in rows]
        counts  = [r["count"] for r in rows]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(origins, counts, color="#3BA1FF")
        ax.set_title("Signal Origins (Last 7 Days)")
        ax.set_ylabel("Count")
        plt.xticks(rotation=30, ha="right")
        out = ART / "origin_breakdown.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close(fig)
        return out, totals, rows
    except Exception as e:
        print(f"[origin breakdown skipped: {e}]")
        return None

origin_result = draw_origin_breakdown()

# ---------- source yield plan section ----------
def compute_yield_plan_section():
    try:
        from src.analytics.source_yield import compute_source_yield
        from src.paths import RETRAINING_LOG_PATH, RETRAINING_TRIGGERED_LOG_PATH
        result = compute_source_yield(RETRAINING_LOG_PATH, RETRAINING_TRIGGERED_LOG_PATH, days=7, min_events=5, alpha=0.7)
        return result
    except Exception as e:
        print(f"[yield plan skipped: {e}]")
        return None

yield_plan_result = compute_yield_plan_section()

# ---------- markdown summary ----------
md = []
md.append("# MoonWire CI Demo Summary\n")
md.append(f"MoonWire Demo Summary — {now_iso}\n")
md.append(f"- **Signal:** `{red(sig_id)}`")
md.append(f"- **Unique reviewers:** {len(reviewers)}")
md.append(f"- **Combined weight:** **{total_weight}**")
md.append(f"- **Threshold:** **{threshold}**  → **{'TRIGGERS' if would_trigger else 'NO TRIGGER'}**")
if last_trig:
    md.append(f"- **Last retrain trigger logged:** {last_trig.get('timestamp','')}")
md.append("")

if origin_result:
    _, totals, rows = origin_result
    md.append("## Origin Breakdown (Last 7 Days)")
    md.append(f"- Flags: {totals['flags']}")
    md.append(f"- Triggers: {totals['triggers']}")
    md.append(f"- Total events: {totals['total_events']}")
    header = f"{'Origin':<15} {'Count':>5} {'Pct':>6}"
    md.append("```")
    md.append(header)
    md.append("-" * len(header))
    for r in rows:
        md.append(f"{r['origin']:<15} {r['count']:>5} {r['pct']:>5.1f}%")
    md.append("```")
    md.append("![Origin Breakdown](origin_breakdown.png)")
    md.append("")

if yield_plan_result:
    md.append("## Source Yield Plan")
    md.append(f"- Window: {yield_plan_result['window_days']} days")
    md.append(f"- Flags total: {yield_plan_result['totals']['flags']}")
    md.append(f"- Triggers total: {yield_plan_result['totals']['triggers']}")
    header = f"{'Origin':<15} {'Flags':>6} {'Triggers':>9} {'Rate':>8} {'Yield':>8} {'Eligible':>9}"
    md.append("```")
    md.append(header)
    md.append("-" * len(header))
    for r in yield_plan_result["origins"]:
        md.append(f"{r['origin']:<15} {r['flags']:>6} {r['triggers']:>9} {r['trigger_rate']:>8.3f} {r['yield_score']:>8.3f} {str(r['eligible']):>9}")
    md.append("```")
    if yield_plan_result["budget_plan"]:
        md.append("**Budget Plan (% allocation):**")
        for bp in yield_plan_result["budget_plan"]:
            md.append(f"- {bp['origin']}: {bp['pct']:.1f}%")
    md.append("")

(ART / "demo_summary.md").write_text("\n".join(md))

print(f"Wrote: {ART/'demo_summary.md'}")
print(f"Wrote: {small_png}")
if origin_result:
    print(f"Wrote: {origin_result[0]}")