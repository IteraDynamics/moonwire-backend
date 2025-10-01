#!/usr/bin/env bash
# Simple helper to build governance plans locally or in CI.
# Usage:
#   MODELS_DIR=models LOGS_DIR=logs ARTIFACTS_DIR=artifacts ./scripts/ci_build_governance.sh

set -euo pipefail

MODELS_DIR="${MODELS_DIR:-models}"
LOGS_DIR="${LOGS_DIR:-logs}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts}"

mkdir -p "$MODELS_DIR" "$LOGS_DIR" "$ARTIFACTS_DIR"

echo "[governance] Building Drift Response plan…"
PYTHONPATH=. MODELS_DIR="$MODELS_DIR" LOGS_DIR="$LOGS_DIR" ARTIFACTS_DIR="$ARTIFACTS_DIR" \
python -m scripts.governance.drift_response || true

echo "[governance] Building Retrain Automation plan…"
PYTHONPATH=. MODELS_DIR="$MODELS_DIR" LOGS_DIR="$LOGS_DIR" ARTIFACTS_DIR="$ARTIFACTS_DIR" \
python -m scripts.governance.retrain_automation || true

echo "[governance] Done."