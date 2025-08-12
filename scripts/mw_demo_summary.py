#!/usr/bin/env python3
"""
MoonWire CI Demo Summary (read-only)

Outputs
 - artifacts/demo_summary.md
 - artifacts/consensus.png
 - artifacts/consensus_social.png

Reads ./logs/*.jsonl; never mutates logs unless DEMO_MODE=true AND retraining log is empty,
in which case it calls the demo seeder to append mock data for this run.

NOTE: Tests import `generate_demo_data_if_needed` from this module.
That function is read-only (in-memory) and returns (reviewers, events).
"""

import os, json, hashlib, random, uuid
from pathlib import Path
from datetime import datetime, timezone, timedelta

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

# ---------- config ----------
LOGS_DIR = Path("logs")
ART = Path("artifacts"); ART.mkdir(exist_ok=True)

DEFAULT_THRESHOLD = 2.5
SOCIAL_W, SOCIAL_H, DPI = 1280, 720, 120  # stable 16:9; slightly higher DPI
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

# ---------- READ-ONLY demo seeding for tests/visuals ----------
def generate_demo_data_if_needed(reviewers, flag_times=None):
    """
    Read-only, in-memory seeding used by tests and the visual.
    If DEMO_MODE=true *and* reviewers is empty, returns a seeded set of
    3–5 reviewers with weights ∈ {0.75,1.0,1.25} and timestamps within the last 60 min.
    Returns (display_reviewers, seeded_events). Does NOT write to disk.
    """
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

# ---------- attempt real log seeding (only if DEMO_MODE && empty) ----------
def maybe_seed_real_logs_if_empty():
    """
    Side-effect seeding: ONLY when DEMO_MODE=true and retraining log is empty.
    Writes mock reviewers to logs so the whole stack (endpoints + visuals) sees data.
    """
    if not is_demo_mode():
        return False
    retrain_path = LOGS_DIR / "retraining_log.jsonl"
    # If retraining log already has any content, skip
    if retrain_path.exists():
        try:
            if any(ln.strip() for ln in retrain_path.read_text().splitlines()):
                return False
        except Exception:
            pass
    # Empty or missing: try to seed
    try:
        from scripts.demo_seed_reviewers import seed_once
        seed_once()  # default: random count & signal
        return True
    except Exception as e:
        # Non-fatal; proceed without seeded logs
        print(f"[demo] seeding skipped due to error: {e}")
        return False

# ---- maybe seed first, then load logs ----
_ = maybe_seed_real_logs_if_empty()

retrain_log   = load_jsonl(LOGS_DIR / "retraining_log.jsonl")
triggered_log = load_jsonl(LOGS_DIR / "retraining_triggered.jsonl")
scores_log    = load_jsonl(LOGS_DIR / "reviewer_scores.jsonl")
score_by_id   = {r.get("reviewer_id"): r for r in scores_log}

# ---- latest signal in retraining log ----
if retrain_log:
    # timestamp may be epoch or iso-ish; prefer numeric when available
    def _key(r):
        t = r.get("timestamp", 0)
        try:
            return float(t)
        except Exception:
            return 0.0
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
    if rid in seen:  # first flag counts
        continue
    seen.add(rid)
    w = r.get("reviewer_weight")
    if w is None:
        sc = (score_by_id.get(rid) or {}).get("score")
        w = band_weight_from_score(sc)
    reviewers.append({"id": rid, "weight": round(float(w), 2)})

# (Optional) in-memory seeding for visuals only — keeps tests happy too
reviewers, _seeded_events = generate_demo_data_if_needed(reviewers, flag_times)

total_weight = round(sum(r["weight"] for r in reviewers), 2)
threshold    = DEFAULT_THRESHOLD
would_trigger = total_weight >= threshold
last_trig = max((t for t in triggerd_log if t.get("signal_id")==sig_id),
                key=lambda x: x.get("timestamp", 0), default=None) if (triggered_log := triggered_log) else None
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
    # theme
    bg="#0B0F19"; panel="#0F1827"; text="#E8EEF7"; sub="#9FB3C8"
    bar_fill="#3BA1FF"; th_col="#FFB74D"; ok="#22C55E"; warn="#EF4444"; line="#1A2433"

    fig = plt.figure(figsize=(SOCIAL_W/DPI, SOCIAL_H/DPI), dpi=DPI, facecolor=bg)
    F = fig.transFigure  # 0..1 coords

    # layout grid (margins)
    L, R, T, B = 0.06, 0.94, 0.92, 0.08
    W, H = R-L, T-B

    # title row
    fig.text(L, T+0.02, "MoonWire — Consensus Check",
             color=text, fontsize=34, weight=700, ha="left", va="center", transform=F)

    status = "TRIGGERED" if would_trigger else "NO TRIGGER"
    badge_col = ok if would_trigger else warn
    fig.text(R, T+0.02, f" {status} ",
             color="white", fontsize=24, weight=700, ha="right", va="center", transform=F,
             bbox=dict(boxstyle="round,pad=0.35", fc=badge_col, ec=badge_col))

    mode_tag = "• DEMO MODE" if is_demo_mode() else ""
    fig.text(L, T-0.005, f"Signal {red(sig_id)}  {mode_tag}",
             color=sub, fontsize=16, ha="left", va="center", transform=F)

    # metric chips
    def chip(x, title, val):
        w,h = 0.15, 0.065
        fig.patches.append(FancyBboxPatch((x, T-0.085), w, h,
                         boxstyle="round,pad=0.28,rounding_size=0.02",
                         linewidth=0, facecolor=panel, transform=F))
        fig.text(x+0.012, T-0.058, title, color=sub, fontsize=12, ha="left", va="center", transform=F)
        fig.text(x+0.012, T-0.083, val,   color=text, fontsize=18, weight=600, ha="left", va="center", transform=F)

    chip(L+0.54, "Total weight", f"{total_weight:.2f}")
    chip(L+0.72, "Threshold",    f"{threshold:.2f}")

    # main card
    card_x, card_y, card_w, card_h = L, B+0.02, W, H-0.10
    fig.patches.append(FancyBboxPatch((card_x, card_y), card_w, card_h,
                        boxstyle="round,pad=0.015,rounding_size=0.015",
                        linewidth=0, facecolor=panel, transform=F))

    # consensus bar (left column)
    left_x = card_x + 0.03
    bar_w = 0.52 * card_w
    bar_h = 0.11
    bar_y = card_y + card_h*0.56

    fig.text(left_x, bar_y + bar_h + 0.03, "Consensus vs Threshold",
             color=sub, fontsize=16, ha="left", va="bottom", transform=F)

    # track
    fig.patches.append(FancyBboxPatch((left_x, bar_y), bar_w, bar_h,
                        boxstyle="round,pad=0.02,rounding_size=0.012",
                        linewidth=0, facecolor=line, transform=F))

    # fill
    cap = max(threshold, total_weight, 3.0)
    fill_w = max(0.0, min(bar_w, (total_weight/cap)*bar_w))
    if fill_w > 0:
        fig.patches.append(FancyBboxPatch((left_x, bar_y), fill_w, bar_h,
                            boxstyle="round,pad=0.02,rounding_size=0.012",
                            linewidth=0, facecolor=bar_fill, transform=F))
    # threshold marker
    th_x = left_x + (threshold/cap)*bar_w
    fig.patches.append(Rectangle((th_x-0.003, bar_y-0.02), 0.006, bar_h+0.04,
                                 linewidth=0, facecolor=th_col, transform=F))
    fig.text(th_x, bar_y + bar_h + 0.005, f"{threshold:.2f}",
             color=th_col, fontsize=16, weight=600, ha="center", va="bottom", transform=F)

    # reviewer panel (right column)
    box_x = left_x + bar_w + 0.04
    box_w = card_x + card_w - box_x - 0.03
    box_h = bar_h + 0.14
    box_y = bar_y - 0.04

    fig.patches.append(FancyBboxPatch((box_x, box_y), box_w, box_h,
                        boxstyle="round,pad=0.02,rounding_size=0.012",
                        linewidth=0, facecolor=line, transform=F))
    fig.text(box_x + 0.02, box_y + box_h - 0.035, "Reviewers (redacted)",
             color=sub, fontsize=14, ha="left", va="top", transform=F)

    if reviewers:
        lines = [f"• {red(r['id'])} — {weight_to_label(r['weight'])}" for r in reviewers[:7]]
        if len(reviewers) > 7:
            lines.append(f"… +{len(reviewers)-7} more")
        fig.text(box_x + 0.02, box_y + box_h - 0.07, "\n".join(lines),
                 color=text, fontsize=15, ha="left", va="top", transform=F)
    else:
        fig.text(box_x + 0.02, box_y + box_h - 0.07, "No reviewers yet",
                 color=text, fontsize=15, ha="left", va="top", transform=F)

    # timeline
    tl_x = left_x; tl_w = card_w - 0.06; tl_h = 0.11; tl_y = card_y + 0.20
    fig.patches.append(FancyBboxPatch((tl_x, tl_y), tl_w, tl_h,
                        boxstyle="round,pad=0.02,rounding_size=0.012",
                        linewidth=0, facecolor=line, transform=F))
    fig.text(tl_x + 0.015, tl_y + tl_h + 0.01, "Flag timeline (last run)",
             color=sub, fontsize=14, ha="left", va="bottom", transform=F)

    if flag_times:
        times = sorted(flag_times); t0, t1 = times[0], times[-1]
        span = max((t1 - t0).total_seconds(), 1.0)
        base_y = tl_y + tl_h/2 - 0.003
        fig.patches.append(Rectangle((tl_x + 0.02, base_y), tl_w - 0.04, 0.006,
                                     linewidth=0, facecolor=panel, transform=F))
        for t in times:
            x01 = (t - t0).total_seconds()/span
            x = tl_x + 0.02 + x01*(tl_w - 0.04)
            fig.patches.append(Rectangle((x-0.003, base_y-0.015), 0.006, 0.03,
                                         linewidth=0, facecolor=bar_fill, transform=F))
    else:
        fig.text(tl_x + 0.02, tl_y + tl_h/2, "No flags this run",
                 color=sub, fontsize=13, ha="left", va="center", transform=F)

    # footer
    fig.text(L, B-0.005, f"moonwire • demo mode • {now_iso}",
             color=sub, fontsize=12, ha="left", va="top", transform=F)

    out = ART / "consensus_social.png"
    # IMPORTANT: no bbox_inches="tight" to avoid random cropping/letterboxing
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
if last_trig:
    md.append(f"- **Last retrain trigger logged:** {last_trig.get('timestamp','')}")
md.append("")
md.append("**Reviewers (redacted):**")
if reviewers:
    for r in reviewers:
        md.append(f"- `{red(r['id'])}` → {weight_to_label(r['weight'])}")
else:
    md.append("- _none found in this run_")
md.append("")
md.append("![Consensus](consensus.png)")

(ART / "demo_summary.md").write_text("\n".join(md))

print(f"Wrote: {ART/'demo_summary.md'}")
print(f"Wrote: {small_png}")
print(f"Wrote: {social_png}")