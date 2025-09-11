# tests/test_label_feedback_version_tagging.py
import json, tempfile, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# import the function under test
sys.path.append("src")
import trigger_likelihood_router as tlr  # type: ignore

ISO = "%Y-%m-%dT%H:%M:%SZ"

def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def test_model_version_found_within_window(monkeypatch):
    with tempfile.TemporaryDirectory() as dtmp:
        trig_path = Path(dtmp) / "trigger_history.jsonl"
        # monkeypatch router's global path
        tlr._TRIGGER_HISTORY_PATH = trig_path  # type: ignore

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

def test_model_version_unknown_when_no_match(monkeypatch):
    with tempfile.TemporaryDirectory() as dtmp:
        trig_path = Path(dtmp) / "trigger_history.jsonl"
        tlr._TRIGGER_HISTORY_PATH = trig_path  # type: ignore

        _write_jsonl(trig_path, [
            {"timestamp": "2025-09-11T12:35:00Z", "origin": "rss_news", "model_version": "v0.5.1"}
        ])

        mv = tlr._find_model_version_for_label(
            label_timestamp="2025-09-11T12:40:00Z",
            origin="reddit",
            window_minutes=5,
        )
        assert mv == "unknown"