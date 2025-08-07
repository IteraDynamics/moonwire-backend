import pytest
from pathlib import Path


@pytest.mark.usefixtures("isolated_logs")
def test_consensus_debug_basic(client, write_flag, write_score, monkeypatch, tmp_path):
    # Patch retraining log path to test directory
    test_log_path = tmp_path / "logs" / "retraining_log.jsonl"
    test_log_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("src.consensus_router.RETRAINING_LOG_PATH", str(test_log_path))

    # Write test data
    write_flag("sigX", "alice", weight=1.2)
    write_flag("sigX", "bob", weight=None)  # Will fallback to score
    write_score("bob", 1.3)
    write_flag("sigX", "alice", weight=1.2)  # Duplicate, should not count

    # Call the endpoint
    r = client.get("/internal/consensus-debug/sigX")
    assert r.status_code == 200
    data = r.json()

    # Validate structure
    assert data["signal_id"] == "sigX"
    assert data["threshold"] == 2.5
    assert data["triggered"] is True  # 1.2 + 1.3 = 2.5
    assert data["total_weight_used"] == pytest.approx(2.5)
    assert set(data["counted_reviewers"]) == {"alice", "bob"}

    flags = data["all_flags"]
    assert len(flags) == 3

    # Check duplicate flag is marked properly
    dup_flags = [f for f in flags if f["reviewer_id"] == "alice"]
    assert len(dup_flags) == 2
    assert dup_flags[0]["duplicate"] is False
    assert dup_flags[1]["duplicate"] is True