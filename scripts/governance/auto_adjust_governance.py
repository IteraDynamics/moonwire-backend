#!/usr/bin/env python3
"""
Automatically adjusts governance_params.json based on paper trading performance.

This is the closed-loop actuator that completes the feedback cycle:
  Paper Trading Performance → Governance Adjustment → Inference Tuning

Run after paper trading completes in CI/CD pipeline.

Usage:
    python scripts/governance/auto_adjust_governance.py [--dry-run]
"""
import json
import sys
import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timezone

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.paths import PERFORMANCE_METRICS_PATH, GOVERNANCE_PARAMS_PATH

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ========== Tuning Rules (Conservative) ==========
# These thresholds determine when to adjust governance parameters

MIN_WIN_RATE = 0.55        # Minimum acceptable win rate
MIN_SHARPE = 1.0           # Minimum acceptable Sharpe ratio
MAX_DRAWDOWN = 0.15        # Maximum acceptable drawdown (15%)
TARGET_WIN_RATE = 0.65     # Excellent performance threshold
TARGET_SHARPE = 1.5        # Excellent Sharpe ratio
TARGET_DRAWDOWN = 0.10     # Excellent drawdown control

# Adjustment deltas (how much to change conf_min per iteration)
DELTA_INCREASE_LARGE = 0.05     # Poor performance → be more selective
DELTA_INCREASE_MEDIUM = 0.03    # Mediocre performance → tighten
DELTA_DECREASE = -0.02          # Excellent performance → relax slightly

# Safety bounds for conf_min
MIN_CONF_THRESHOLD = 0.50       # Never go below 50% confidence
MAX_CONF_THRESHOLD = 0.90       # Never go above 90% (too restrictive)


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file, return empty dict if not found."""
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return {}


def save_json(path: Path, data: Dict[str, Any]) -> None:
    """Save JSON file with pretty formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    logger.info(f"Saved updated governance params to {path}")


def adjust_conf_min(current: float, delta: float) -> float:
    """
    Adjust confidence threshold, capped at safety bounds.

    Args:
        current: Current conf_min value
        delta: Amount to adjust (positive = more selective, negative = less selective)

    Returns:
        New conf_min value, bounded by [MIN_CONF_THRESHOLD, MAX_CONF_THRESHOLD]
    """
    new = current + delta
    bounded = max(MIN_CONF_THRESHOLD, min(MAX_CONF_THRESHOLD, new))
    return round(bounded, 2)


def evaluate_performance(symbol: str, metrics: Dict[str, Any]) -> List[str]:
    """
    Evaluate performance metrics and return list of reasons for adjustment.

    Args:
        symbol: Asset symbol (e.g., "BTC")
        metrics: Performance metrics dict with win_rate, sharpe, max_drawdown

    Returns:
        List of adjustment reasons
    """
    reasons = []

    win_rate = metrics.get("win_rate", 0)
    sharpe = metrics.get("sharpe", 0)
    drawdown = metrics.get("max_drawdown", 0)

    # Check for poor performance (increase selectivity)
    if win_rate < MIN_WIN_RATE:
        reasons.append(f"Low win rate ({win_rate:.2f} < {MIN_WIN_RATE})")

    if sharpe < MIN_SHARPE:
        reasons.append(f"Low Sharpe ratio ({sharpe:.2f} < {MIN_SHARPE})")

    if drawdown > MAX_DRAWDOWN:
        reasons.append(f"High drawdown ({drawdown:.2f} > {MAX_DRAWDOWN})")

    # Check for excellent performance (relax selectivity slightly)
    if (win_rate > TARGET_WIN_RATE and
        sharpe > TARGET_SHARPE and
        drawdown < TARGET_DRAWDOWN):
        reasons.append(f"Excellent performance (win={win_rate:.2f}, sharpe={sharpe:.2f}, dd={drawdown:.2f})")

    return reasons


def auto_adjust_governance(dry_run: bool = False) -> bool:
    """
    Main adjustment logic: read performance metrics, adjust governance params.

    Args:
        dry_run: If True, log changes but don't write files

    Returns:
        True if changes were made, False otherwise
    """
    logger.info("=" * 60)
    logger.info("GOVERNANCE AUTO-ADJUSTMENT")
    logger.info(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    logger.info(f"Dry run: {dry_run}")
    logger.info("=" * 60)

    # Load inputs
    metrics = load_json(PERFORMANCE_METRICS_PATH)
    params = load_json(GOVERNANCE_PARAMS_PATH)

    if not metrics:
        logger.warning("No performance metrics found - skipping adjustment")
        return False

    changes = []
    symbols_adjusted = []

    for symbol, perf in metrics.items():
        # Initialize default params if symbol not in governance
        if symbol not in params:
            params[symbol] = {"conf_min": 0.60, "debounce_min": 15}
            logger.info(f"Initializing governance params for {symbol}")

        current_conf = params[symbol]["conf_min"]
        new_conf = current_conf

        # Evaluate performance and get reasons
        reasons = evaluate_performance(symbol, perf)

        if not reasons:
            logger.info(f"{symbol}: No adjustment needed (conf_min={current_conf})")
            continue

        # Determine adjustment based on reasons
        for reason in reasons:
            if "Low win rate" in reason or "Low Sharpe" in reason:
                new_conf = adjust_conf_min(new_conf, DELTA_INCREASE_MEDIUM)

            if "High drawdown" in reason:
                new_conf = adjust_conf_min(new_conf, DELTA_INCREASE_LARGE)

            if "Excellent performance" in reason:
                new_conf = adjust_conf_min(new_conf, DELTA_DECREASE)

        # Apply adjustment if changed
        if new_conf != current_conf:
            change_msg = f"{symbol}: conf_min {current_conf} → {new_conf} | {', '.join(reasons)}"
            changes.append(change_msg)
            symbols_adjusted.append(symbol)
            params[symbol]["conf_min"] = new_conf
            logger.info(f"✓ {change_msg}")
        else:
            logger.info(f"{symbol}: Evaluated but no net change (conf_min={current_conf})")

    # Summary
    if changes:
        logger.info("=" * 60)
        logger.info(f"ADJUSTMENTS APPLIED: {len(changes)}")
        for change in changes:
            logger.info(f"  - {change}")
        logger.info("=" * 60)

        if not dry_run:
            save_json(GOVERNANCE_PARAMS_PATH, params)
            logger.info("✅ Governance params updated successfully")
        else:
            logger.info("🏃 DRY RUN - No files written")

        return True
    else:
        logger.info("No governance changes needed")
        return False


def commit_changes() -> bool:
    """
    Commit governance changes to git (for CI automation).

    Returns:
        True if commit successful, False otherwise
    """
    try:
        # Check if running in CI
        if not os.getenv("CI"):
            logger.info("Not in CI environment - skipping git commit")
            return False

        # Configure git
        subprocess.run(["git", "config", "user.name", "MoonWire Bot"], check=True)
        subprocess.run(["git", "config", "user.email", "bot@moonwire.ai"], check=True)

        # Add governance file
        subprocess.run(["git", "add", str(GOVERNANCE_PARAMS_PATH)], check=True)

        # Check if there are changes to commit
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            capture_output=True
        )

        if result.returncode == 0:
            logger.info("No changes to commit")
            return False

        # Commit
        commit_msg = f"🤖 Auto-adjust governance params based on performance\n\nTimestamp: {datetime.now(timezone.utc).isoformat()}"
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)

        logger.info("✅ Changes committed successfully")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Git operation failed: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Auto-adjust governance parameters based on paper trading performance"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing files"
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Commit changes to git (CI mode)"
    )

    args = parser.parse_args()

    # Run adjustment
    changed = auto_adjust_governance(dry_run=args.dry_run)

    # Commit if requested and changes were made
    if args.commit and changed and not args.dry_run:
        commit_changes()

    # Exit with appropriate code for CI
    sys.exit(0 if changed or args.dry_run else 1)
