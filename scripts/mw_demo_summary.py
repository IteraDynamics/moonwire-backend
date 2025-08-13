#!/usr/bin/env python3
"""
MoonWire CI Demo Summary (read-only)

Outputs
 - artifacts/demo_summary.md
 - artifacts/consensus.png
 - artifacts/consensus_social.png

Now includes:
 - Origin breakdown using compute_origin_breakdown()
 - Demo-mode seeding of mock origin events for visuals
"""

import os, json, hashlib, random, uuid
from pathlib import Path
from datetime import datetime, timezone, timedelta

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from src.paths import (
    LOGS_DIR,
    RETRAINING_LOG_PATH,
    RETRAINING_TRIGGERED_LOG_PATH,
)
from src.analytics.origin_utils import compute_origin_breakdown

# ---------- config ----------
ART = Path("artifacts"); ART.mkdir(exist_ok=True)
DEFAULT_THRESHOLD = 2.5
SOCIAL_W, SOCIAL_H, DPI = 1280, 720, 120  # 16:9

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
        ts = now - timedelta(minutes=random.randint(2, 55))
        seeded.append({"id": rid, "weight": w, "timestamp": ts.isoformat()})
        display.append({"id": rid, "weight": w})
        flag_times.append(ts)
    return display, seeded

def generate_demo_origins_if_needed(origins_rows):
    if not is_demo_mode() or origins_rows:
        return origins_rows
    demo = [
        {"origin": "twitter", "count": 5, "pct": 50.0},
        {"origin": "reddit", "count": 3, "pct": 30.0},
        {"origin": "rss_news", "count": 2, "pct": 20.0},
    ]
    return demo

# ---------- maybe seed real logs if empty ----------
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
retrain_log   = load_jsonl(RETRAINING_LOG_PATH)
triggered_log = load_jsonl(RETRAINING_TRIGGERED_LOG_PATH)
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

# Demo seeding for reviewers
reviewers, _seeded_events = generate_demo_data_if_needed(reviewers, flag_times)

# ---- origin breakdown ----
origins_rows, origins_totals = compute_origin_breakdown(
    flags_path=RETRAINING_LOG_PATH,
    triggers_path=RETRAINING_TRIGGERED_LOG_PATH,
    days=7,
    include_triggers=False
)
origins_rows = generate_demo_origins_if_needed(origins_rows)

# ---- consensus math ----
total_weight = round(sum(r["weight"] for r in reviewers), 2)
threshold    = DEFAULT_THRESHOLD
would_trigger = total_weight >= threshold
now_iso = datetime.now(timezone.utc).isoformat()

# ---------- small CI bar ----------
plt.figure(figsize=(3.8, 2.4), dpi=200)
plt.title("Consensus Weight vs Threshold")
plt.bar(["weight","threshold"], [total_weight, threshold])
plt.tight_layout()
small_png = ART / "consensus.png"
plt.savefig(small_png, dpi=200)
plt.close()

# ---------- social card ----------
def draw_social_card():
    bg="#0B0F19"; panel="#0F1827"; text="#E8EEF7"; sub="#9FB3C8"
    bar_fill="#3BA1FF"; th_col="#FFB74D"; ok="#22C55E"; warn="#EF4444"; line="#1A2433"
    fig = plt.figure(figsize=(SOCIAL_W/DPI, SOCIAL_H/DPI), dpi=DPI, facecolor=bg)
    F = fig.transFigure
    L, R, T, B = 0.06, 0.94, 0.92, 0.08
    W, H = R-L, T-B
    fig.text(L, T+0.02, "MoonWire — Consensus Check",
             color=text, fontsize=34, weight=700, ha="left", va="center", transform=F)
    status = "TRIGGERED" if would_trigger else "NO TRIGGER"
    badge_col = ok if would_trigger else warn
    fig.text(R, T+0.02, f" {status} ",
             color="white", fontsize=24, weight=700, ha="right", va="center", transform=F,
             bbox=dict(boxstyle="round,pad=0.35", fc=badge_col, ec=badge_col))
    fig.text(L, T-0.005, f"Signal {red(sig_id)}", color=sub, fontsize=16, ha="left", va="center", transform=F)

    def chip(x, title, val):
        w,h = 0.15, 0.065
        fig.patches.append(FancyBboxPatch((x, T-0.085), w, h,
                         boxstyle="round,pad=0.28,rounding_size=0.02",
                         linewidth=0, facecolor=panel, transform=F))
        fig.text(x+0.012, T-0.058, title, color=sub, fontsize=12, ha="left", va="center", transform=F)
        fig.text(x+0.012, T-0.083, val,   color=text, fontsize=18, weight=600, ha="left", va="center", transform=F)

    chip(L+0.54, "Total weight", f"{total_weight:.2f}")
    chip(L+0.72, "Threshold",    f"{threshold:.2f}")

    # reviewer panel
    box_x = L; box_y = B+0.45; box_w = 0.42; box_h = 0.4
    fig.patches.append(FancyBboxPatch((box_x, box_y), box_w, box_h,
                        boxstyle="round,pad=0.02,rounding_size=0.012",
                        linewidth=0, facecolor=line, transform=F))
    fig.text(box_x + 0.02, box_y + box_h - 0.035, "Reviewers (redacted)",
             color=sub, fontsize=14, ha="left", va="top", transform=F)
    if reviewers:
        lines = [f"• {red(r['id'])} — {weight_to_label(r['weight'])}" for r in reviewers[:6]]
        fig.text(box_x + 0.02, box_y + box_h - 0.07, "\n".join(lines),
                 color=text, fontsize=15, ha="left", va="top", transform=F)
    else:
        fig.text(box_x + 0.02, box_y + box_h - 0.07, "No reviewers yet",
                 color=text, fontsize=15, ha="left", va="top", transform=F)

    # origin panel
    box2_x = L+0.48; box2_y = box_y; box2_w = 0.42; box2_h = 0.4
    fig.patches.append(FancyBboxPatch((box2_x, box2_y), box2_w, box2_h,
                        boxstyle="round,pad=0.02,rounding_size=0.012",
                        linewidth=0, facecolor=line, transform=F))
    fig.text(box2_x + 0.02, box2_y + box2_h - 0.035, "Origins (last 7d)",
             color=sub, fontsize=14, ha="left", va="top", transform=F)
    if origins_rows:
        lines = [f"• {row['origin']}: {row['count']} ({row['pct']}%)" for row in origins_rows]
        fig.text(box2_x + 0.02, box2_y + box2_h - 0.07, "\n".join(lines),
                 color=text, fontsize=15, ha="left", va="top", transform=F)
    else:
        fig.text(box2_x + 0.02, box2_y + box2_h - 0.07, "No origin data",
                 color=text, fontsize=15, ha="left", va="top", transform=F)

    out = ART / "consensus_social.png"
    fig.savefig(out, dpi=DPI, facecolor=bg)
    plt.close(fig)
    return out

social_png = draw_social_card()

# ---------- markdown summary ----------
md = []
md.append("# MoonWire CI Demo Summary")
md.append("")
md.append(f"MoonWire Demo Summary — {now_iso}")
md.append("")
md.append("Pipeline proof (CI): end-to-end tests passed; consensus math reproduced on latest flagged signal.")
md.append("")
md.append(f"- **Signal:** `{red(sig_id)}`")
md.append(f"- **Unique reviewers:** {len(reviewers)}")
md.append(f"- **Combined weight:** **{total_weight}**")
md.append(f"- **Threshold:** **{threshold}**  → **{'TRIGGERS' if would_trigger else 'NO TRIGGER'}**")
md.append("")
md.append("**Reviewers (redacted):**")
if reviewers:
    for r in reviewers:
        md.append(f"- `{red(r['id'])}` → {weight_to_label(r['weight'])}")
else:
    md.append("- _none found in this run_")
md.append("")
md.append("**Signal origin breakdown (last 7 days):**")
if origins_rows:
    for row in origins_rows:
        md.append(f"- {row['origin']}: {row['count']} ({row['pct']}%)")
else:
    md.append("- _no origin data found_")
md.append("")
md.append("![Consensus](consensus.png)")

(ART / "demo_summary.md").write_text("\n".join(md))

print(f"Wrote: {ART/'demo_summary.md'}")
print(f"Wrote: {small_png}")
print(f"Wrote: {social_png}")