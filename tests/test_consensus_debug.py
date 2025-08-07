# tests/test_consensus_debug.py

import pytest
from datetime import datetime

def test_consensus_debug_basic(client, write_flag, write_score):
    # One reviewer with known weight
    write_flag("sigX", "alice", weight=1.2)
    write_flag("sigX", "bob", weight=None)
    write_score("bob", 1.3)

    # Duplicate that shouldn't count
    write_flag("sigX", "alice", weight=1.2)

    r = client.get("/internal/consensus-debug/sigX")
    assert r.status_code == 200
    data = r.json()

    assert data["signal_id"] == "sigX"
    assert len(data["all_flags"]) == 3
    assert data["all_flags"][0]["duplicate"] is False
    assert data["all_flags"][1]["duplicate"] is False
    assert data["all_flags"][2]["duplicate"] is True

    assert set(data["counted_reviewers"]) == {"alice", "bob"}
    assert data["total_weight_used"] == pytest.approx(2.5)
    assert data["triggered"] is True
    assert data["threshold"] == 2.5

    # Ensure timestamps are in ISO format
    for entry in data["all_flags"]:
        datetime.fromisoformat(entry["timestamp"])


def test_consensus_debug_not_found(client):
    r = client.get("/internal/consensus-debug/does_not_exist")
    assert r.status_code == 404