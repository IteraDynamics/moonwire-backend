import pytest
from pathlib import Path
from src import consensus_router  # Explicit import to monkeypatch constant directly


@pytest.mark.usefixtures("isolated_logs")
def test_consensus_debug_basic(client, write_flag, write_score, monkeypatch, tmp_path):
    # Patch retraining path globally before any writes happen
    test_log_path = tmp_path / "logs" / "retraining_log.jsonl"
    test_log_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(consensus_router, "RETRAINING_LOG_PATH", str(test_log_path))

    # Write test data (now goes to patched path)
    write_flag("sigX", "alice", weight=1.2)
    write_flag("sigX", "bob", weight=None)  # Will fallback to score
    write_score("bob", 1.3)
    write_flag("sigX", "alice", weight=1.2)  # Duplicate

    # Now call the endpoint
    r = client.get("/internal/consensus-debug/sigX")
    assert r.status_code == 200
    data = r.json()

    assert data["signal_id"] == "sigX"
    assert data["threshold"] == 2.5
    assert data["triggered"] is True
    assert data["total_weight_used"] == pytest.approx(2.5)
    assert set(data["counted_reviewers"]) == {"alice", "bob"}

    flags = data["all_flags"]
    assert len(flags) == 3

    dup_flags = [f for f in flags if f["reviewer_id"] == "alice"]
    assert dup_flags[0]["duplicate"] is False
    assert dup_flags[1]["duplicate"] is True