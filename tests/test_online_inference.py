import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
from src import paths
from src.ml.infer import live_backtest_last_24h

def _j(d): import json; return json.dumps(d)

def test_live_inference_runs_without_triggers(tmp_path: Path, monkeypatch):
    # Recent flags but no triggers
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    flags = []
    for i in range(5):
        t = now - timedelta(hours=i+1)
        flags.append({"timestamp": t.isoformat(), "origin": "twitter"})
    (tmp_path/"retraining_log.jsonl").write_text("\n".join(_j(r) for r in flags))
    (tmp_path/"retraining_triggered.jsonl").write_text("")

    monkeypatch.setattr(paths, "RETRAINING_LOG_PATH", tmp_path/"retraining_log.jsonl")
    monkeypatch.setattr(paths, "RETRAINING_TRIGGERED_LOG_PATH", tmp_path/"retraining_triggered.jsonl")

    out = live_backtest_last_24h()
    assert "overall" in out and "per_origin" in out