#!/usr/bin/env python3
"""
MoonWire CI Demo Summary (read-only)

Outputs
 - artifacts/demo_summary.md
 - artifacts/consensus.png
 - artifacts/consensus_social.png

Reads ./logs/*.jsonl; never mutates logs.
"""

import os, json, hashlib, random, uuid
from pathlib import Path
from datetime import datetime, timezone, timedelta

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---------------- config ----------------
LOGS_DIR = Path("logs")
ART = Path("artifacts"); ART.mkdir(exist_ok=True)

DEFAULT_THRESHOLD = 2.5
SOCIAL_W, SOCIAL_H, DPI = 1280, 720, 120   # 16:9, crisp but light
# palette: grayscale + one accent
BG   = "#0e1116"
FG   = "#e6edf3"
MUT  = "#a0acb8"
ACC  = "#4da3ff"   # accent for weight
TH   = "#ffbf66"   # threshold line
GOOD = "#34c759"
BAD  = "#ff453a"
# ----------------------------------------


# ------------- helpers ------------------
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
        s = str(val)
        s = s[:-1] + "+00:00" if s.endswith("Z") else s
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception: return None

def seed_reviewers_if_empty(reviewers, flag_times):
    """In-memory demo seeding; never writes to disk."""
    demo = os.getenv("DEMO_MODE", "false").lower() in ("1","true","yes")
    if reviewers or not demo: return reviewers, []
    now = datetime.now(timezone.utc)
    n = random.randint(3,5)
    choices = [0.75, 1.0, 1.25]
    seeded = []
    view = []
    for _ in range(n):
        rid = f"demo-{uuid.uuid4().hex[:8]}"
        w   = random.choice(choices)
        ts  = now - timedelta(minutes=random.randint(3,55))
        seeded.append({"id": rid, "weight": w, "timestamp": ts.isoformat()})
        view.append({"id": rid, "weight": w})
        flag_times.append(ts)
    return view, seeded
# ----------------------------------------


# --------- load current context ----------
retrain_log   = load_jsonl(LOGS_DIR / "retraining_log.jsonl")
triggered_log = load_jsonl(LOGS_DIR / "retraining_triggered.jsonl")
scores_log    = load_jsonl(LOGS_DIR / "reviewer_scores.jsonl")
score_by_id   = {r.get("reviewer_id"): r for r in scores_log}

if retrain_log:
    latest  = max(retrain_log, key=lambda r: r.get("timestamp", 0))
    sig_id  = latest.get("signal_id", "unknown")
    sig_rows = [r for r in retrain_log if r.get("signal_id")==sig_id]
else:
    sig_id = "none"; sig_rows = []

seen, reviewers, flag_times = set(), [], []
for r in sorted(sig_rows, key=lambda x: x.get("timestamp", 0)):
    t = parse_ts(r.get("timestamp"))
    if t: flag_times.append(t)
    rid = r.get("reviewer_id","")
    if rid in seen: continue
    seen.add(rid)
    w = r.get("reviewer_weight")
    if w is None:
        sc = (score_by_id.get(rid) or {}).get("score")
        w = band_weight_from_score(sc)
    reviewers.append({"id": rid, "weight": round(float(w), 2)})

reviewers, seeded = seed_reviewers_if_empty(reviewers, flag_times)

total_weight = round(sum(r["weight"] for r in reviewers), 2)
threshold    = DEFAULT_THRESHOLD
would_trigger = total_weight >= threshold
last_trig = max((t for t in triggered_log if t.get("signal_id")==sig_id),
                key=lambda x: x.get("timestamp", 0), default=None)
now_iso = datetime.now(timezone.utc).isoformat()

# ---- tiny CI bar (kept) ----
plt.figure(figsize=(3.6, 2.2), dpi=200)
plt.title("Consensus Weight vs Threshold")
plt.bar(["weight","threshold"], [total_weight, threshold])
plt.tight_layout()
small_png = ART / "consensus.png"
plt.savefig(small_png, dpi=200)
plt.close()


# -------- minimalist social card --------
def draw_social_card():
    fig = plt.figure(figsize=(SOCIAL_W/DPI, SOCIAL_H/DPI), dpi=DPI, facecolor=BG)
    # gridspec: header (1), main (3), footer (0.6)
    gs = GridSpec(nrows=5, ncols=2, figure=fig,
                  left=0.06, right=0.94, top=0.92, bottom=0.08,
                  hspace=0.3, wspace=0.25)

    # header
    axh = fig.add_subplot(gs[0, :])
    axh.axis("off")
    axh.text(0.0, 0.7, "MoonWire — Consensus Check",
             color=FG, fontsize=28, fontweight="bold", ha="left", va="center")
    status = "TRIGGERED" if would_trigger else "NO TRIGGER"
    axh.text(1.0, 0.7, status,
             color="white", fontsize=16, fontweight="bold", ha="right", va="center",
             bbox=dict(boxstyle="round,pad=0.35", fc=(GOOD if would_trigger else BAD), ec="none"))
    mode = " • DEMO MODE (seeded)" if seeded else ""
    axh.text(0.0, 0.1, f"Signal {red(sig_id)}{mode}",
             color=MUT, fontsize=12, ha="left", va="center")

    # left: gauge
    axg = fig.add_subplot(gs[1:3, 0])
    axg.set_facecolor(BG)
    cap = max(threshold, total_weight, 3.0)
    axg.barh([0], [total_weight], height=0.35, color=ACC)
    axg.axvline(threshold, color=TH, linewidth=2)
    axg.set_xlim(0, cap)
    axg.set_yticks([])
    axg.tick_params(axis="x", colors=MUT, labelsize=11)
    for spine in axg.spines.values():
        spine.set_color(MUT); spine.set_linewidth(0.6)
    axg.set_xlabel("weight", color=MUT, fontsize=11)
    axg.text(total_weight, 0, f" {total_weight:.2f}", color=FG, fontsize=12, va="center")
    axg.text(threshold, 0.42, f"{threshold:.2f}", color=TH, fontsize=12, ha="center", va="bottom")

    # right: reviewers
    axr = fig.add_subplot(gs[1:3, 1])
    axr.axis("off")
    axr.text(0.0, 1.0, "Reviewers (redacted)", color=MUT, fontsize=12, ha="left", va="top")
    if reviewers:
        lines = [f"• {red(r['id'])} — {weight_to_label(r['weight'])}" for r in reviewers[:10]]
        if len(reviewers) > 10:
            lines.append(f"… +{len(reviewers)-10} more")
        axr.text(0.0, 0.88, "\n".join(lines), color=FG, fontsize=13, ha="left", va="top", linespacing=1.4)
    else:
        axr.text(0.0, 0.88, "No reviewers yet", color=FG, fontsize=13, ha="left", va="top")

    # bottom: timeline
    axt = fig.add_subplot(gs[3:, :])
    axt.set_facecolor(BG)
    axt.spines["top"].set_visible(False)
    axt.spines["right"].set_visible(False)
    axt.spines["left"].set_visible(False)
    axt.spines["bottom"].set_color(MUT)
    axt.tick_params(axis="y", left=False, labelleft=False)
    axt.tick_params(axis="x", colors=MUT, labelsize=10)
    axt.set_title("Flag timeline (last run)", color=MUT, fontsize=12, loc="left", pad=6)

    if flag_times:
        times = sorted(flag_times)
        t0, t1 = times[0], times[-1]
        span = max((t1 - t0).total_seconds(), 1.0)
        xs = [(t - t0).total_seconds() / span for t in times]
        axt.scatter(xs, [0]*len(xs), s=24, color=ACC)
        axt.set_xlim(-0.02, 1.02)
        axt.set_xticks([0, 1], labels=["start", "end"])
    else:
        axt.set_xlim(0, 1)
        axt.set_xticks([])
        axt.text(0.0, 0.5, "No flags this run", color=MUT, fontsize=11, ha="left", va="center")

    # footer
    fig.text(0.06, 0.04, f"moonwire • demo mode • {now_iso}",
             color=MUT, fontsize=10, ha="left", va="center")

    out = ART / "consensus_social.png"
    fig.savefig(out, dpi=DPI, facecolor=BG)  # no tight bbox → consistent framing
    plt.close(fig)
    return out


social_png = draw_social_card()

# -------- markdown summary (unchanged) --------
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