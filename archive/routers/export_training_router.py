from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse, JSONResponse
from pathlib import Path
import csv
import json
from datetime import datetime

LOG_FILE = Path("logs/signal_history.jsonl")
router = APIRouter()

def parse_jsonl(path: Path):
    if not path.exists():
        return []
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]

@router.get("/internal/export-training-data")
def export_training_data(
    format: str = Query("csv", enum=["csv", "jsonl"]),
    preview: bool = Query(False),
    asset: str = Query(None),
    adjusted_only: bool = Query(False)
):
    rows = parse_jsonl(LOG_FILE)

    # Apply optional filters
    if asset:
        rows = [r for r in rows if r.get("asset") == asset]
    if adjusted_only:
        rows = [r for r in rows if r.get("adjustment_applied")]

    if preview:
        return JSONResponse(content=rows[:10])

    # CSV streaming
    def csv_stream():
        fieldnames = [
            "timestamp", "asset", "score", "label", "confidence",
            "fallback_type", "adjustment_reason", "adjustment_applied", "type"
        ]
        writer = csv.DictWriter(
            open("/tmp/training_export.csv", "w"), fieldnames=fieldnames
        )
        writer.writeheader()
        for row in rows:
            filtered = {key: row.get(key) for key in fieldnames}
            writer.writerow(filtered)

        yield from open("/tmp/training_export.csv", "r")

    if format == "csv":
        return StreamingResponse(
            csv_stream(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=training_data_{datetime.utcnow().isoformat()}.csv"}
        )
    else:
        return JSONResponse(content=rows)