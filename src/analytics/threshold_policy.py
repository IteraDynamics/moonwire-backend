from __future__ import annotations

def threshold_for_regime(regime: str) -> float:
    """
    Map volatility regime to a suggested consensus/trigger threshold.
    calm: lowers bar (easier)
    normal: baseline
    turbulent: raises bar (harder)
    """
    mapping = {"calm": 2.2, "normal": 2.5, "turbulent": 3.0}
    return float(mapping.get(regime, 2.5))
