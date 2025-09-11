# tests/test_label_feedback_version_tagging.py
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

import src.trigger_likelihood_router as tlr  # <-- use the package import

ISO = "%Y-%m-%dT%H:%M:%SZ"

def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

def test_model_version_found_within_window(tmp_path):
    trig_path = tmp_path / "trigger_history.jsonl"
    # monkeypatch the module-level path used by the helper
    tlr._TRIGGER_HISTORY_PATH = trig_path  # type: ignore[attr-defined]

    t0 = datetime(2025, 9, 11, 12, 40, 0, tzinfo=timezone.utc)
    rows = [
        {
            "timestamp": (t0 - timedelta(minutes=2)).strftime(ISO),
            "origin": "reddit",
            "adjusted_score": 0.72,
            "decision": True,
            "model_version": "v0.5.1",
        },
        {
            "timestamp": (t0 - timedelta(minutes=10)).strftime(ISO),
            "origin": "reddit",
            "adjusted_score": 0.33,
            "decision": False,
            "model_version": "v0.5.0",
        },
    ]
    _write_jsonl(trig_path, rows)

    mv = tlr._find_model_version_for_label(
        label_timestamp=t0.strftime(ISO),
        origin="reddit",
        window_minutes=5,
    )
    assert mv == "v0.5.1"

def test_model_version_unknown_when_no_match(tmp_path):
    trig_path = tmp_path / "trigger_history.jsonl"
    tlr._TRIGGER_HISTORY_PATH = trig_path  # type: ignore[attr-defined]

    _write_jsonl(trig_path, [
        {"timestamp": "2025-09-11T12:35:00Z", "origin": "rss_news", "model_version": "v0.5.1"}
    ])

    mv = tlr._find_model_version_for_label(
        label_timestamp="2025-09-11T12:40:00Z",
        origin="reddit",
        window_minutes=5,
    )
    assert mv == "unknown"