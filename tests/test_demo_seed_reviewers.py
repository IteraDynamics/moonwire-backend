# tests/test_demo_seed_reviewers.py
import json
from pathlib import Path

from scripts.demo_seed_reviewers import seed_once


def _read_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def test_seed_skips_when_demo_off(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEMO_MODE", raising=False)

    out = seed_once()
    assert out["seeded"] is False
    assert not Path("logs/reviewer_scores.jsonl").exists()
    assert not Path("logs/retraining_log.jsonl").exists()


def test_seed_writes_when_demo_on(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEMO_MODE", "true")

    out = seed_once(n_reviewers=5, signal_id="sig-demo")
    assert out["seeded"] is True
    assert out["reviewers"] == 5
    assert out["signal_id"] == "sig-demo"

    scores = _read_jsonl(Path("logs/reviewer_scores.jsonl"))
    flags = _read_jsonl(Path("logs/retraining_log.jsonl"))
    assert len(scores) >= 5
    assert len(flags) >= 5
    assert all(f["signal_id"] == "sig-demo" for f in flags)
    for s in scores:
        assert 0.30 <= s["score"] <= 1.00
        assert "reviewer_id" in s


def test_seed_timestamps_recent(tmp_path, monkeypatch):
    import time

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEMO_MODE", "true")

    seed_once(n_reviewers=3, signal_id="sig-ts")
    flags = _read_jsonl(Path("logs/retraining_log.jsonl"))
    assert flags
    now = time.time()
    # within ~25h
    for f in flags:
        assert now - f["timestamp"] <= 25 * 3600
