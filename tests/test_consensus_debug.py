# tests/test_consensus_debug.py

import pytest
from pathlib import Path
from src import paths  # Optional: helps clarify intent

@pytest.mark.usefixtures("isolated_logs")
def test_consensus_debug_basic(client, write_flag, write_score, monkeypatch, tmp_path):
    # Patch paths before any writes happen
    test_log_path = tmp_path / "logs" / "retraining_log.jsonl"
    test_score_path = tmp_path / "logs" / "reviewer_scores.jsonl"
    test_log_path.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("src.paths.RETRAINING_LOG_PATH", str(test_log_path))
    monkeypatch.setattr("src.paths.REVIEWER_SCORES_PATH", str(test_score_path))

    # Write test data (goes to patched paths)
    write_flag("sigX", "alice", weight=1.2)
    write_flag("sigX", "bob", weight=None)  # Will fallback to score
    write_score("bob", 1.3)
    write_flag("sigX", "alice", weight=1.2)  # Duplicate

    # Now call the endpoint
    r = client.get("/internal/consensus-debug/sigX")
    assert r.status_code == 200

    payload = r.json()
    assert payload["signal_id"] == "sigX"
    assert payload["total_weight_used"] == pytest.approx(2.5)
    assert payload["triggered"] is True
    assert len(payload["all_flags"]) == 3
    assert len(payload["counted_reviewers"]) == 2

    flags_by_alice = [f for f in payload["all_flags"] if f["reviewer_id"] == "alice"]
    assert flags_by_alice[0]["duplicate"] is False
    assert flags_by_alice[1]["duplicate"] is True