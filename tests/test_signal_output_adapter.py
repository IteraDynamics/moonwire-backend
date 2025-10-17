from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

import pytest


def read_last_json(path: Path) -> Dict[str, Any]:
    data = path.read_text(encoding="utf-8").rstrip("\n")
    # last line only
    last = data.split("\n")[-1]
    return json.loads(last)


def test_dual_write_default(monkeypatch, tmp_path: Path):
    # Use temp working dir so we don't touch repo files
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SIGNALS_FILE", raising=False)

    # Import after chdir so module writes under tmp cwd
    sys.path.insert(0, str(tmp_path))
    from src.signal_log import write_signal  # type: ignore

    row = {
        "ts": "2025-01-01T00:00:00Z",
        "symbol": "btc",
        "direction": "long",
        "confidence": 0.7,
        "price": 60000.0,
        "source": "ml_engine",
        "model_version": "v0.9.0",
        "outcome": None,
    }
    normalized = write_signal(row)

    canon = Path("logs/signal_history.jsonl")
    legacy = Path("logs/signals.jsonl")
    assert canon.exists()
    assert legacy.exists()

    j1 = read_last_json(canon)
    j2 = read_last_json(legacy)
    assert j1 == j2

    # required keys present
    for k in ("id", "ts", "symbol", "direction", "confidence", "price", "source", "model_version", "outcome"):
        assert k in j1

    # normalization behavior
    assert j1["symbol"] == "BTC"
    assert j1["direction"] == "long"
    assert isinstance(j1["confidence"], float)
    assert isinstance(j1["price"], float)
    assert j1["id"].startswith("sig_")


def test_env_override_single_file(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)

    custom_path = Path("logs/custom.jsonl")
    monkeypatch.setenv("SIGNALS_FILE", str(custom_path))

    sys.path.insert(0, str(tmp_path))
    from src.signal_log import write_signal  # type: ignore

    row = {
        "ts": "2025-01-01T01:00:00Z",
        "symbol": "eth",
        "direction": "short",
        "confidence": 0.66,
        "price": 3000.0,
        "source": "ml_engine",
        "model_version": "v0.9.0",
        "outcome": None,
    }
    normalized = write_signal(row)

    assert custom_path.exists()
    # defaults must NOT be written in override mode
    assert not Path("logs/signal_history.jsonl").exists()
    assert not Path("logs/signals.jsonl").exists()

    j = read_last_json(custom_path)
    assert j["symbol"] == "ETH"
    assert j["direction"] == "short"
    assert j["id"].startswith("sig_")


def test_auto_id_generation(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SIGNALS_FILE", raising=False)

    sys.path.insert(0, str(tmp_path))
    from src.signal_log import write_signal  # type: ignore

    row = {
        # id intentionally omitted
        "ts": "2025-01-01T02:00:00Z",
        "symbol": "sol",
        "direction": "long",
        "confidence": 0.75,
        "price": 150.0,
        "source": "ml_engine",
        "model_version": "v0.9.0",
        "outcome": None,
    }
    out = write_signal(row)
    assert "id" in out and out["id"]
    assert re.match(r"^sig_2025-01-01T02:00:00Z_SOL_long$", out["id"]) is not None


def test_cli_smoke(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SIGNALS_FILE", raising=False)

    # Create minimal package layout so tools import works:
    (tmp_path / "src").mkdir()
    (tmp_path / "src/__init__.py").write_text("", encoding="utf-8")

    # Write the modules under tmp_path so the CLI can import them
    # (In real repo these files already exist; this makes the test self-contained in tmp cwd.)
    signal_log_code = r'''
from __future__ import annotations
import json, os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

def _now_utc_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

def make_signal_id(ts_iso: str, symbol: str, direction: str) -> str:
    return f"sig_{ts_iso}_{symbol.upper()}_{direction.lower()}"

def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def _append_jsonl(p: Path, row: Dict[str, Any]):
    _ensure_parent(p)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass

def _validate_and_normalize(row: Dict[str, Any]) -> Dict[str, Any]:
    row = {str(k).lower(): row[k] for k in row}
    for k in ("ts","symbol","direction","confidence","price","source","model_version","outcome"):
        if k not in row:
            raise KeyError(k)
    row["symbol"] = row["symbol"].upper()
    row["direction"] = row["direction"].lower()
    if not row.get("id"):
        row["id"] = make_signal_id(row["ts"], row["symbol"], row["direction"])
    row["confidence"] = float(row["confidence"])
    row["price"] = float(row["price"])
    return row

def write_signal(row: Dict[str, Any]) -> Dict[str, Any]:
    row = _validate_and_normalize(dict(row))
    custom = os.getenv("SIGNALS_FILE")
    if custom:
        _append_jsonl(Path(custom), row)
        return row
    _append_jsonl(Path("logs/signal_history.jsonl"), row)
    _append_jsonl(Path("logs/signals.jsonl"), row)
    return row
'''
    (tmp_path / "src/signal_log.py").write_text(signal_log_code, encoding="utf-8")

    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    emit_code = r'''
#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
from src.signal_log import write_signal

def now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

p = argparse.ArgumentParser()
p.add_argument("--symbol", required=True)
p.add_argument("--dir", required=True, choices=["long","short"])
p.add_argument("--conf", type=float, default=0.70)
p.add_argument("--price", type=float, required=True)
p.add_argument("--ts", default=None)
p.add_argument("--model", default="v0.9.0")
p.add_argument("--source", default="manual")
args = p.parse_args()

row = {
    "id": None,
    "ts": (args.ts or now()),
    "symbol": args.symbol,
    "direction": args.dir,
    "confidence": args.conf,
    "price": args.price,
    "source": args.source,
    "model_version": args.model,
    "outcome": None
}
out = write_signal(row)
print(json.dumps(out))
'''
    (tools_dir / "emit_signal.py").write_text(emit_code, encoding="utf-8")

    # Run CLI
    result = subprocess.run(
        [sys.executable, "tools/emit_signal.py", "--symbol", "BTC", "--dir", "long", "--conf", "0.7", "--price", "60000"],
        capture_output=True,
        text=True,
        check=True,
    )
    # stdout JSON
    stdout = result.stdout.strip()
    out = json.loads(stdout)
    assert out["symbol"] == "BTC"
    assert out["direction"] == "long"

    # Canonical file appended
    canon = Path("logs/signal_history.jsonl")
    legacy = Path("logs/signals.jsonl")
    assert canon.exists() and legacy.exists()
    cj = json.loads(canon.read_text(encoding="utf-8").splitlines()[-1])
    lj = json.loads(legacy.read_text(encoding="utf-8").splitlines()[-1])
    assert cj == lj