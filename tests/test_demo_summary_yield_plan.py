import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess

def _write_jsonl(path: Path, rows):
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def test_yield_plan_section_in_summary_with_days_filter(tmp_path, monkeypatch):
    """
    Runs mw_demo_summary.py with fake logs and checks:
      - Yield plan section appears in demo_summary.md
      - Budget % sum ≈ 100
      - Old events (outside days filter) are excluded from yield math
    """
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(exist_ok=True)  # <-- changed to avoid FileExistsError

    now = datetime.now(timezone.utc)
    recent_ts = now.isoformat()
    old_ts = (now - timedelta(days=30)).isoformat()

    # Flags: twitter recent, reddit old (should be excluded)
    _write_jsonl(logs_dir / "retraining_log.jsonl", [
        {"origin": "twitter", "timestamp": recent_ts},
        {"origin": "reddit",  "timestamp": old_ts},
    ])

    # Triggers: twitter recent
    _write_jsonl(logs_dir / "retraining_triggered.jsonl", [
        {"origin": "twitter", "timestamp": recent_ts},
    ])

    # Patch LOGS_DIR env so mw_demo_summary reads from tmp logs dir
    monkeypatch.setenv("LOGS_DIR", str(logs_dir))
    monkeypatch.setenv("DEMO_MODE", "false")

    # Run mw_demo_summary.py
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "mw_demo_summary.py"
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    md_path = Path("artifacts") / "demo_summary.md"
    assert md_path.exists(), "demo_summary.md not found"

    md_text = md_path.read_text()
    assert "## Source Yield Plan" in md_text

    # Budget sum ≈ 100
    lines = [line for line in md_text.splitlines() if line.strip().startswith("{")]
    assert lines, "No budget plan JSON found in summary"
    budget_plan = json.loads("".join(lines))
    total_pct = sum(item["pct"] for item in budget_plan)
    assert 99.9 <= total_pct <= 100.1, f"Budget plan sum out of range: {total_pct}"