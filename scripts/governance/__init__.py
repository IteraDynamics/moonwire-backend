"""
Governance package (automated policy actions).

Exports:
- drift_response: policy engine for automated threshold tightening based on
  persistent calibration drift.
"""
from . import drift_response  # re-export for convenience

__all__ = ["drift_response"]
