"""Crowd density, Fruin level of service, crush-risk and queue metrics from position snapshots."""
from __future__ import annotations

import numpy as np

# Fruin (1971) walkway level of service: minimum space per person [m^2/person]
# for LOS A..E (converted from 35, 25, 15, 10, 5 ft^2); below the last value = LOS F.
FRUIN_WALKWAY_AREA = np.array([3.2516, 2.3226, 1.3935, 0.9290, 0.4645])
FRUIN_LABELS = np.array(list("ABCDEF"))

# Densities around 5 persons/m^2 and above are widely treated as dangerous
# (risk of crushing); kept as a parameter so the report can state its source.
CRUSH_DENSITY = 5.0


def density_grid(xy: np.ndarray, cell: float = 1.0, bounds: tuple | None = None):
    """Persons per m^2 on a square grid.

    Returns (density[nx, ny], x_edges, y_edges). ``bounds`` = (xmin, xmax, ymin, ymax);
    if omitted the grid covers the points.
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if cell <= 0:
        raise ValueError("cell must be > 0")
    if bounds is None:
        if xy.size == 0:
            return np.zeros((0, 0)), np.array([]), np.array([])
        xmin, ymin = np.floor(xy.min(axis=0) / cell) * cell
        xmax, ymax = (np.floor(xy.max(axis=0) / cell) + 1) * cell
    else:
        xmin, xmax, ymin, ymax = bounds
    xe = np.arange(xmin, xmax + 0.5 * cell, cell)
    ye = np.arange(ymin, ymax + 0.5 * cell, cell)
    counts, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[xe, ye]) if xy.size else (
        np.zeros((len(xe) - 1, len(ye) - 1)), None, None)
    return counts / (cell * cell), xe, ye


def peak_density(snapshots, cell: float = 1.0) -> float:
    """Highest cell density over a sequence of (N, 2) position snapshots."""
    peak = 0.0
    for xy in snapshots:
        dens, _, _ = density_grid(xy, cell)
        if dens.size:
            peak = max(peak, float(dens.max()))
    return peak


def fruin_los(density) -> np.ndarray:
    """Fruin walkway level of service (A..F) for density values [persons/m^2]."""
    d = np.atleast_1d(np.asarray(density, dtype=float))
    with np.errstate(divide="ignore"):
        area = np.where(d > 0, 1.0 / d, np.inf)
    idx = np.sum(area[:, None] < FRUIN_WALKWAY_AREA[None, :], axis=1)
    return FRUIN_LABELS[idx]


def crush_risk(density, threshold: float = CRUSH_DENSITY) -> np.ndarray:
    return np.asarray(density, dtype=float) >= threshold


def area_above(density_grid_values: np.ndarray, threshold: float, cell: float = 1.0) -> float:
    """Floor area [m^2] whose density is at or above threshold."""
    return float(np.sum(np.asarray(density_grid_values) >= threshold) * cell * cell)


def queue_length(xy: np.ndarray, centre, radius: float) -> int:
    """People within ``radius`` of an exit centre (a simple queue measure for one snapshot)."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    return int(np.sum(np.linalg.norm(xy - np.asarray(centre, dtype=float), axis=1) <= radius))
