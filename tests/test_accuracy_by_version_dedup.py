# tests/test_accuracy_by_version_dedup.py
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

from src.ml.metrics import compute_accuracy_by_version

ISO = "%Y-%m-%dT%H:%M:%SZ"

def _w(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

def test_dedup_and_overalls(tmp_path):
    trig = tmp_path / "trigger_history.jsonl"
    lab  = tmp_path / "label_feedback.jsonl"

    # Use relative "now" so the 24h window always contains our rows
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    t0 = now - timedelta(minutes=3)  # trigger 3 minutes ago

    # One trigger (v0.5.2, decision=True)
    _w(trig, [{
        "timestamp": t0.strftime(ISO),
        "origin": "reddit",
        "decision": True,
        "model_version": "v0.5.2",
    }])

    # Two labels mapping to same trigger within the ±5m window:
    # first True (TP), then False (would be FP) — but dedup keeps only the first.
    _w(lab, [
        {"timestamp": (t0 + timedelta(minutes=1)).strftime(ISO), "origin": "reddit",
         "label": True,  "adjusted_score": 0.7, "model_version": "v0.5.2"},
        {"timestamp": (t0 + timedelta(minutes=2)).strftime(ISO), "origin": "reddit",
         "label": False, "adjusted_score": 0.4, "model_version": "v0.5.2"},
    ])

    res = compute_accuracy_by_version(
        trig, lab, window_hours=24, match_window_minutes=5, dedup_one_label_per_trigger=True
    )

    v = res.get("v0.5.2")
    assert v is not None, f"expected version present, got res={res}"
    # Dedup → only the first label counts: TP=1, FP=0, FN=0
    assert v["tp"] == 1 and v["fp"] == 0 and v["fn"] == 0 and v["n"] == 1

    # Overalls exist
    assert "_micro" in res and "_macro" in res
    micro = res["_micro"]; macro = res["_macro"]
    assert micro["tp"] == 1 and micro["n"] == 1
    assert 0.99 <= micro["precision"] <= 1.01
    assert 0.99 <= micro["recall"]    <= 1.01
    assert 0.99 <= micro["f1"]        <= 1.01
    assert macro["versions"] >= 1
