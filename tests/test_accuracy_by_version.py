# tests/test_accuracy_by_version.py
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

from src.ml.metrics import compute_accuracy_by_version

ISO = "%Y-%m-%dT%H:%M:%SZ"

def _w(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

def test_compute_accuracy_by_version(tmp_path):
    trig = tmp_path / "trigger_history.jsonl"
    lab  = tmp_path / "label_feedback.jsonl"

    t0 = datetime(2025, 9, 11, 12, 0, 0, tzinfo=timezone.utc)

    # Triggers: v0.5.2 (3 rows), v0.5.1 (2 rows)
    trig_rows = [
        # v0.5.2
        {"timestamp": (t0 + timedelta(minutes=0)).strftime(ISO),  "origin": "reddit",  "decision": True,  "model_version": "v0.5.2"},
        {"timestamp": (t0 + timedelta(minutes=10)).strftime(ISO), "origin": "reddit",  "decision": True,  "model_version": "v0.5.2"},
        {"timestamp": (t0 + timedelta(minutes=20)).strftime(ISO), "origin": "reddit",  "decision": False, "model_version": "v0.5.2"},
        # v0.5.1
        {"timestamp": (t0 + timedelta(minutes=60)).strftime(ISO), "origin": "twitter", "decision": True,  "model_version": "v0.5.1"},
        {"timestamp": (t0 + timedelta(minutes=70)).strftime(ISO), "origin": "twitter", "decision": False, "model_version": "v0.5.1"},
    ]
    _w(trig, trig_rows)

    # Labels (v0.5.2+ style with model_version)
    lab_rows = [
        # v0.5.2
        {"timestamp": (t0 + timedelta(minutes=1)).strftime(ISO),  "origin": "reddit",  "label": True,  "adjusted_score": 0.7, "model_version": "v0.5.2"},  # TP
        {"timestamp": (t0 + timedelta(minutes=11)).strftime(ISO), "origin": "reddit",  "label": False, "adjusted_score": 0.4, "model_version": "v0.5.2"}, # FP
        {"timestamp": (t0 + timedelta(minutes=21)).strftime(ISO), "origin": "reddit",  "label": True,  "adjusted_score": 0.6, "model_version": "v0.5.2"},  # FN
        # v0.5.1
        {"timestamp": (t0 + timedelta(minutes=62)).strftime(ISO), "origin": "twitter", "label": True,  "adjusted_score": 0.8, "model_version": "v0.5.1"},  # TP
        {"timestamp": (t0 + timedelta(minutes=71)).strftime(ISO), "origin": "twitter", "label": True,  "adjusted_score": 0.8, "model_version": "v0.5.1"},  # FN
    ]
    _w(lab, lab_rows)

    res = compute_accuracy_by_version(trig, lab, window_hours=87600)  # large window for stability

    v52 = res.get("v0.5.2");  v51 = res.get("v0.5.1")
    assert v52 is not None and v51 is not None

    # v0.5.2: TP=1, FP=1, FN=1
    assert v52["tp"] == 1 and v52["fp"] == 1 and v52["fn"] == 1 and v52["n"] == 3
    assert abs(v52["precision"] - 0.5) < 1e-9
    assert abs(v52["recall"]    - 0.5) < 1e-9
    assert abs(v52["f1"]        - 0.5) < 1e-9

    # v0.5.1: TP=1, FP=0, FN=1 → P=1.0, R=0.5, F1≈0.6667
    assert v51["tp"] == 1 and v51["fp"] == 0 and v51["fn"] == 1 and v51["n"] == 2
    assert abs(v51["precision"] - 1.0) < 1e-9
    assert abs(v51["recall"]    - 0.5) < 1e-9
    assert abs(v51["f1"]        - (2*1.0*0.5/(1.0+0.5))) < 1e-9
