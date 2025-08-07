import pytest
from src.paths import LOGS_DIR
from pathlib import Path


@pytest.fixture(autouse=True)
def clear_logs():
    log_dir = Path(LOGS_DIR)
    for path in log_dir.glob("*.jsonl"):
        path.write_text("")


def test_dashboard_endpoint_returns_expected_structure(client, write_flag, write_score):
    write_flag("sigD1", "alice", weight=1.1)
    write_flag("sigD1", "bob", weight=1.0)
    write_flag("sigD2", "charlie", weight=0.8)
    write_score("charlie", 1.25)

    r = client.get("/internal/consensus-dashboard")
    assert r.status_code == 200
    data = r.json()

    assert isinstance(data, list)
    assert any(x["signal_id"] == "sigD1" for x in data)

    for record in data:
        assert "signal_id" in record
        assert "reviewers" in record
        assert "total_weight" in record
        assert "triggered" in record
        assert "last_flagged_timestamp" in record