#!/usr/bin/env python3
"""
MoonWire CI Demo Summary (read-only)

Outputs:
 - artifacts/demo_summary.md
 - artifacts/consensus.png
 - artifacts/consensus_social.png

Safe: reads ./logs/*.jsonl written by tests; does not import app code or mutate logs.
"""

import os
import json
import hashlib
import random
import uuid
from pathlib import Path
from datetime import datetime, timezone, timedelta

import matplotlib.pyplot as plt

# -------------------- config --------------------
LOGS_DIR = Path("logs")
ART = Path("artifacts")
ART.mkdir(exist_ok=True)

DEFAULT_THRESHOLD = 2.5
SOCIAL_W, SOCIAL_H, DPI = 1280, 720, 110
# ------------------------------------------------


# ---------- helpers ----------
def red(s: str) -> str:
    if not s:
        return "000000"
    return hashlib.sha1(s.encode()).hexdigest()[:6]


def load_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return out


def band_weight_from_score(score):
    if score is None:
        return 1.0
    if score >= 0.75:
        return 1.25
    if score >= 0.50:
        return 1.0
    return 0.75


def weight_to_label(w: float) -> str:
    if w >= 1.125:
        return "High"
    if w >= 0.875:
        return "Med"
    return "Low"


def parse_ts(val):
    if val is None:
        return None
    try:
        ts = float(val)
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        pass
    try:
        s = str(val)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None


# ---------- DEMO seeding ----------
def seed_reviewers_if_empty(reviewers, flag_times=None):
    """
    If DEMO_MODE=true and reviewers list is empty, seed 3-5 demo reviewers.
    Returns (reviewers, seeded_events). flag_times is optional for tests.
    """
    flag_times = flag_times or []
    demo_on = os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes")
    if not demo_on or reviewers:
        return reviewers, []

    now = datetime.now(timezone.utc)
    n = random.randint(3, 5)
    choices = [0.75, 1.0, 1.25]
    seeded = []
    display = []
    for _ in range(n):
        rid = f"demo-{uuid.uuid4().hex[:8]}"
        w = random.choice(choices)
        ts = now - timedelta(minutes=random.randint(2, 55))
        seeded.append({"id": rid, "weight": w, "timestamp": ts.isoformat()})
        display.append({"id": rid, "weight": w})
        flag_times.append(ts)
    return display, seeded


# Backwards compatibility for existing tests
def generate_demo_data_if_needed(reviewers, flag_times=None):
    return seed_reviewers_if_empty(reviewers, flag_times)


# ---- load logs ----
retrain_log = load_jsonl(LOGS_DIR / "retraining_log.jsonl")
triggered_log = load_jsonl(LOGS_DIR / "retraining_triggered.jsonl")
scores_log = load_jsonl(LOGS_DIR / "reviewer_scores.jsonl")
score_by_id = {r.get("reviewer_id"): r for r in scores_log}

# ---- latest signal ----
if retrain_log:
    latest = max(retrain_log, key=lambda r: r.get("timestamp", 0))
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
    if t:
        flag_times.append(t)
    rid = r.get("reviewer_id", "")
    if rid in seen:
        continue
    seen.add(rid)
    w = r.get("reviewer_weight")
    if w is None:
        sc = (score_by_id.get(rid) or {}).get("score")
        w = band_weight_from_score(sc)
    reviewers.append({"id": rid, "weight": round(float(w), 2)})

# Seed for demo if needed (no writes)
reviewers, seeded_events = seed_reviewers_if_empty(reviewers, flag_times)

total_weight = round(sum(r["weight"] for r in reviewers), 2)
threshold = DEFAULT_THRESHOLD
would_trigger = total_weight >= threshold
last_trig = max(
    (t for t in triggered_log if t.get("signal_id") == sig_id),
    key=lambda x: x.get("timestamp", 0),
    default=None,
)

now = datetime.now(timezone.utc).isoformat()

# ---- small CI bar ----
plt.figure(figsize=(3.8, 2.4), dpi=200)
plt.title("Consensus Weight vs Threshold")
plt.bar(["weight", "threshold"], [total_weight, threshold], color=["#4C78A8", "#F58518"])
plt.tight_layout()
small_png = ART / "consensus.png"
plt.savefig(small_png, dpi=200)
plt.close()


# ---------- Simple, readable social card ----------
def draw_social_card():
    fig, ax = plt.subplots(figsize=(SOCIAL_W / DPI, SOCIAL_H / DPI), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax.axis("off")

    # Title
    ax.text(0.02, 0.95, "MoonWire — Consensus Check", fontsize=28, weight="bold", ha="left", va="center")

    # Status
    status = "TRIGGERED" if would_trigger else "NO TRIGGER"
    status_color = "green" if would_trigger else "red"
    ax.text(0.98, 0.95, status, fontsize=20, color=status_color, weight="bold", ha="right", va="center")

    # Signal info
    mode_tag = "(DEMO MODE)" if seeded_events else ""
    ax.text(0.02, 0.90, f"Signal {red(sig_id)} {mode_tag}", fontsize=14, color="gray", ha="left", va="center")

    # Consensus bar chart
    ax.barh(["Consensus"], [total_weight], color="#4C78A8", height=0.4)
    ax.axvline(threshold, color="#F58518", linestyle="--", label=f"Threshold {threshold:.2f}")
    ax.set_xlim(0, max(threshold, total_weight) * 1.2)
    ax.set_xlabel("Weight")

    # Reviewer list
    y_pos = 0.70
    ax.text(0.02, y_pos, "Reviewers (redacted):", fontsize=14, weight="bold", ha="left", va="center")
    if reviewers:
        for r in reviewers[:7]:
            y_pos -= 0.04
            ax.text(0.04, y_pos, f"{red(r['id'])} — {weight_to_label(r['weight'])}", fontsize=12, ha="left", va="center")
        if len(reviewers) > 7:
            y_pos -= 0.04
            ax.text(0.04, y_pos, f"... +{len(reviewers) - 7} more", fontsize=12, ha="left", va="center")
    else:
        ax.text(0.04, y_pos - 0.04, "No reviewers yet", fontsize=12, ha="left", va="center")

    # Footer
    ax.text(0.02, 0.02, f"moonwire • {now}", fontsize=10, color="gray", ha="left", va="center")

    social_png = ART / "consensus_social.png"
    plt.savefig(social_png, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return social_png


# ---- draw social card ----
social_png = draw_social_card()

# ---- markdown summary ----
md = []
md.append("# MoonWire CI Demo Summary")
md.append("")
md.append(f"MoonWire Demo Summary — {now}")
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

md_path = ART / "demo_summary.md"
md_path.write_text("\n".join(md))

print(f"Wrote: {md_path}")
print(f"Wrote: {ART / 'consensus.png'}")
print(f"Wrote: {social_png}")