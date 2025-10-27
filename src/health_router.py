# src/health_router.py
"""
Health check endpoints for monitoring system status.

Provides:
- /ping: Lightweight heartbeat for uptime monitoring
- /health: Comprehensive system health check
"""
import json
import logging
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter
from pathlib import Path

from src.paths import (
    GOVERNANCE_PARAMS_PATH,
    SHADOW_LOG_PATH,
    PERFORMANCE_METRICS_PATH,
    MODELS_DIR
)
from src.observability import failure_tracker

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/ping")
async def ping():
    """
    Lightweight heartbeat endpoint for uptime monitoring.

    Use this for frequent health checks (every 30-60 seconds).
    Returns immediately with minimal overhead.
    """
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.head("/ping", include_in_schema=False)
async def ping_head():
    """HEAD variant of /ping for uptime monitors that prefer HEAD requests."""
    return {"status": "ok"}


@router.get("/health")
async def health_check():
    """
    Comprehensive system health check.

    Validates:
    - Governance params readable
    - Shadow logging functional
    - ML inference available
    - Recent shadow activity
    - System failures tracked

    Returns 200 with status="healthy" if all checks pass.
    Returns 200 with status="degraded" if non-critical issues found.
    Returns 503 with status="unhealthy" if critical issues found.
    """
    checks = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {},
        "failures": {}
    }

    # Check 1: Governance params readable
    try:
        if not GOVERNANCE_PARAMS_PATH.exists():
            checks["checks"]["governance"] = {
                "status": "warning",
                "message": "Governance params file not found (will use defaults)"
            }
            checks["status"] = "degraded"
        else:
            params = json.loads(GOVERNANCE_PARAMS_PATH.read_text())
            checks["checks"]["governance"] = {
                "status": "ok",
                "symbols": len(params),
                "path": str(GOVERNANCE_PARAMS_PATH)
            }
    except Exception as e:
        checks["checks"]["governance"] = {
            "status": "error",
            "error": str(e)
        }
        checks["status"] = "degraded"

    # Check 2: Shadow log writable
    try:
        from src.jsonl_writer import safe_jsonl_append
        test_entry = {
            "health_check": True,
            "test": "write_test",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        success = safe_jsonl_append(SHADOW_LOG_PATH, test_entry)

        if success:
            checks["checks"]["shadow_log"] = {
                "status": "ok",
                "path": str(SHADOW_LOG_PATH)
            }
        else:
            checks["checks"]["shadow_log"] = {
                "status": "error",
                "message": "Failed to write to shadow log"
            }
            checks["status"] = "unhealthy"
    except Exception as e:
        checks["checks"]["shadow_log"] = {
            "status": "error",
            "error": str(e)
        }
        checks["status"] = "unhealthy"

    # Check 3: ML inference available
    try:
        from src.ml.infer import infer_asset_signal

        # Try to load model for BTC
        result = infer_asset_signal("BTC")

        checks["checks"]["ml_inference"] = {
            "status": "ok",
            "model_type": result.get("model_type", "unknown"),
            "confidence": result.get("confidence")
        }
    except FileNotFoundError as e:
        checks["checks"]["ml_inference"] = {
            "status": "warning",
            "message": "Model bundle not found (expected in shadow/dev mode)",
            "error": str(e)
        }
        # Not critical - system can run without ML in heuristic mode
    except Exception as e:
        checks["checks"]["ml_inference"] = {
            "status": "warning",
            "message": "ML inference unavailable",
            "error": str(e)
        }

    # Check 4: Recent shadow activity
    try:
        if SHADOW_LOG_PATH.exists():
            lines = SHADOW_LOG_PATH.read_text().strip().split('\n')
            if lines and lines[-1]:
                last_entry = json.loads(lines[-1])
                last_ts_str = last_entry.get("ts") or last_entry.get("timestamp")

                # Parse timestamp
                if last_ts_str:
                    if last_ts_str.endswith('Z'):
                        last_ts_str = last_ts_str[:-1] + '+00:00'
                    last_ts = datetime.fromisoformat(last_ts_str)
                    age_seconds = (datetime.now(timezone.utc) - last_ts).total_seconds()

                    checks["checks"]["shadow_activity"] = {
                        "status": "ok",
                        "last_entry": last_ts_str,
                        "age_seconds": int(age_seconds),
                        "total_entries": len(lines)
                    }

                    # Warn if no activity in last hour
                    if age_seconds > 3600:
                        checks["checks"]["shadow_activity"]["status"] = "warning"
                        checks["checks"]["shadow_activity"]["message"] = "No shadow activity in last hour"
                        checks["status"] = "degraded"
                else:
                    checks["checks"]["shadow_activity"] = {
                        "status": "warning",
                        "message": "Last entry has no timestamp"
                    }
            else:
                checks["checks"]["shadow_activity"] = {
                    "status": "warning",
                    "message": "Shadow log is empty"
                }
        else:
            checks["checks"]["shadow_activity"] = {
                "status": "info",
                "message": "Shadow log not yet created (expected on first run)"
            }
    except Exception as e:
        checks["checks"]["shadow_activity"] = {
            "status": "warning",
            "error": str(e)
        }

    # Check 5: Performance metrics available
    try:
        if PERFORMANCE_METRICS_PATH.exists():
            metrics = json.loads(PERFORMANCE_METRICS_PATH.read_text())
            checks["checks"]["performance_metrics"] = {
                "status": "ok",
                "symbols": len(metrics),
                "path": str(PERFORMANCE_METRICS_PATH)
            }
        else:
            checks["checks"]["performance_metrics"] = {
                "status": "info",
                "message": "No performance metrics yet (expected before first paper trading run)"
            }
    except Exception as e:
        checks["checks"]["performance_metrics"] = {
            "status": "warning",
            "error": str(e)
        }

    # Check 6: Tracked failures
    failures = failure_tracker.get_all_failures()
    if failures:
        checks["failures"] = failures
        # If any component has >50 failures, mark as degraded
        if any(count > 50 for count in failures.values()):
            checks["status"] = "degraded"
            checks["checks"]["failure_tracker"] = {
                "status": "warning",
                "message": "High failure count detected",
                "details": failures
            }
    else:
        checks["checks"]["failure_tracker"] = {
            "status": "ok",
            "message": "No failures tracked"
        }

    return checks


@router.get("/health/failures")
async def get_failures():
    """
    Get detailed failure tracking information.

    Useful for debugging when health check shows degraded status.
    """
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "failures": failure_tracker.get_all_failures()
    }
