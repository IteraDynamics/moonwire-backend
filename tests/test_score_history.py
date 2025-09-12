# tests/test_score_history.py
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

from src.ml.metrics import compute_score_distribution

ISO = "%Y-%m-%dT%H:%M:%SZ"

def _w(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

def test_score_history_stats_and_hist(tmp_path):
    p = tmp_path / "score_history.jsonl"
    t0 = datetime.now(timezone.utc)

    rows = []
    # make 10 scores: 5 near 0.2-0.3, 5 near 0.7-0.8
    for i in range(5):
        rows.append({"timestamp": (t0 - timedelta(hours=1, minutes=i)).strftime(ISO), "origin": "reddit",
                     "adjusted_score": 0.2 + 0.02*i, "model_version": "v0.5.2"})
    for i in range(5):
        rows.append({"timestamp": (t0 - timedelta(hours=2, minutes=i)).strftime(ISO), "origin": "twitter",
                     "adjusted_score": 0.7 + 0.02*i, "model_version": "v0.5.2"})

    _w(p, rows)

    snap = compute_score_distribution(p, window_hours=24, threshold=0.5)
    assert snap["count"] == 10
    assert 0.4 < snap["mean"] < 0.7
    assert snap["min"] >= 0.2 and snap["max"] <= 0.78
    # above 0.5 should be the 5 higher ones
    assert abs(snap["pct_above_threshold"] - 0.5) < 1e-9

    # histogram sanity: counts should sum to n
    total_bins = sum(b["count"] for b in snap["hist"])
    assert total_bins == snap["count"]
