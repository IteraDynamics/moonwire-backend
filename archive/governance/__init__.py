# scripts/governance/__init__.py
"""
Governance package

Contains automation/policy modules that keep MoonWire healthy:
- drift_response: detects persistent miscalibration and proposes/applies safe threshold tweaks
- retrain_automation: decides when to retrain, evaluates candidates, and (optionally) promotes

Import patterns supported:
    from scripts.governance import drift_response, retrain_automation
    from scripts.governance.retrain_automation import build_retrain_plan
"""

from __future__ import annotations

__all__ = [
    "drift_response",
    "retrain_automation",
    "__version__",
]

# Version of this governance policy bundle (not the app version)
__version__ = "v1"

# Eagerly import submodules so attributes exist on the package namespace.
# If a submodule is missing, raise immediately with a clear error.
try:
    from . import drift_response  # noqa: F401
except Exception as e:  # pragma: no cover
    raise ImportError(f"Failed to import scripts.governance.drift_response: {e}") from e

try:
    from . import retrain_automation  # noqa: F401
except Exception as e:  # pragma: no cover
    raise ImportError(f"Failed to import scripts.governance.retrain_automation: {e}") from e