"""Shared movement rules: element types, SFPE speed-density, effective width and capacity.

Every model uses these functions so that a difference between models comes
from how the model represents crowd movement, not from different inputs.

SFPE hydraulic relations (SFPE Handbook, "Employing the hydraulic model"):
    speed           S  = k (1 - a D)        for D >= d_free, else S(d_free)
    specific flow   Fs = S D                 [persons / (m s)]
    max Fs at D = 1/(2a):  Fs_max = k / (4a)  (= 1.32 p/(m s) for k = 1.40)
    flow            F  = Fs * W_eff,  W_eff = clear width - 2 * boundary layer
"""
from __future__ import annotations

import numpy as np

from crowdsim.models.params import MovementParams


def element_type(kind: str, dz: float, length: float) -> str:
    """Map a graph edge to an SFPE element type: level, ramp, stair or aisle."""
    if kind == "stair":
        return "stair"
    if kind == "aisle":
        return "aisle"
    if kind == "ramp":
        return "ramp"
    if kind == "vomitory" and length > 0 and abs(dz) / length > 0.10:
        return "stair"  # a stepped vomitory
    return "level"


def effective_width(mp: MovementParams, kind: str, width: float) -> float:
    bl = mp.boundary_layer.get(kind, 0.0)
    return max(width - 2.0 * bl, 0.30)


def max_specific_flow(mp: MovementParams, etype: str) -> float:
    return mp.k[etype] / (4.0 * mp.sfpe_a)


def capacity(mp: MovementParams, kind: str, etype: str, width: float) -> float:
    """Maximum flow through an element [persons/s]."""
    return max_specific_flow(mp, etype) * effective_width(mp, kind, width)


def free_speed_factor(mp: MovementParams, etype: str, ascending: bool) -> float:
    """Free walking speed on this element relative to level walking (k_type / k_level)."""
    f = mp.k[etype] / mp.k["level"]
    if ascending and etype in ("stair", "aisle"):
        f *= mp.ascending_factor
    return f


def density_speed_ratio(mp: MovementParams, density) -> np.ndarray:
    """SFPE speed at density D relative to the free speed (1 for D <= d_free, 0 at 1/a)."""
    d = np.maximum(np.asarray(density, dtype=float), mp.d_free)
    return np.clip((1.0 - mp.sfpe_a * d) / (1.0 - mp.sfpe_a * mp.d_free), 0.0, 1.0)
