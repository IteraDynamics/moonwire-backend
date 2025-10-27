# src/observability.py
"""
Observability and failure tracking for MoonWire.

Tracks recurring failures and provides alerting mechanisms
for critical system degradation.
"""
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class FailureTracker:
    """
    Tracks and alerts on recurring failures.

    Critical for production monitoring - ensures that silent failures
    in non-blocking code paths (like shadow logging) are still observable.
    """

    def __init__(self):
        self.counters: Dict[str, int] = defaultdict(int)
        self.last_alert: Dict[str, datetime] = {}

    def record_failure(self, component: str, error: Exception, context: Optional[Dict] = None) -> None:
        """
        Record a failure and alert if threshold exceeded.

        Args:
            component: Component name (e.g., "shadow_write", "ml_inference")
            error: The exception that occurred
            context: Additional context for debugging
        """
        self.counters[component] += 1
        count = self.counters[component]

        log_extra = {
            "component": component,
            "error": str(error),
            "error_type": type(error).__name__,
            "failure_count": count,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        if context:
            log_extra.update(context)

        logger.error(f"{component} failure #{count}", extra=log_extra)

        # Alert every 10 failures
        if count % 10 == 0:
            self._send_alert(component, count, error, context)

    def _send_alert(
        self,
        component: str,
        count: int,
        error: Exception,
        context: Optional[Dict] = None
    ) -> None:
        """
        Send critical alert for recurring failures.

        TODO: Integrate with alerting system (Discord/Slack webhook, PagerDuty, etc.)
        """
        msg = f"🚨 ALERT: {component} has failed {count} times. Last error: {error}"
        logger.critical(msg, extra={
            "alert_type": "recurring_failure",
            "component": component,
            "failure_count": count,
            "error": str(error),
            "context": context
        })

        # TODO: Uncomment when webhook configured
        # self._send_webhook_alert(component, count, error)

    def get_failure_count(self, component: str) -> int:
        """Get total failure count for a component."""
        return self.counters.get(component, 0)

    def reset_counter(self, component: str) -> None:
        """Reset failure counter (e.g., after issue resolved)."""
        if component in self.counters:
            del self.counters[component]
        if component in self.last_alert:
            del self.last_alert[component]

    def get_all_failures(self) -> Dict[str, int]:
        """Get all failure counts for diagnostics."""
        return dict(self.counters)


# Global singleton instance
failure_tracker = FailureTracker()
