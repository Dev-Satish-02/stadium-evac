"""Evacuation performance metrics, computed identically for every model.

Every Phase 2 model returns an :class:`EvacuationResult`; the comparison in the
report is only fair because all models are scored by these same functions.

Conventions
-----------
* ``exit_time[i]`` is when agent i passed an exit [s]; ``np.inf`` if never.
* ``exit_id[i]`` is the exit index used by agent i; ``-1`` if never.
* Percentile times (T50, T90, ...) are fractions of the WHOLE crowd, so a run
  in which only 80 % got out has T90 = inf (it never reached 90 %).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EvacuationResult:
    exit_time: np.ndarray
    exit_id: np.ndarray
    exit_width: np.ndarray            # (n_exits,) clear width of each exit [m]
    model: str = ""
    scenario: str = ""
    seed: int = 0
    compute_time_s: float = float("nan")
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.exit_time = np.asarray(self.exit_time, dtype=float)
        self.exit_id = np.asarray(self.exit_id, dtype=int)
        self.exit_width = np.asarray(self.exit_width, dtype=float)
        if self.exit_time.shape != self.exit_id.shape:
            raise ValueError("exit_time and exit_id must have the same length")
        done = np.isfinite(self.exit_time)
        if np.any(self.exit_id[done] < 0) or np.any(self.exit_id[~done] >= 0):
            raise ValueError("exit_id must be >= 0 exactly for agents with a finite exit_time")
        if done.any() and self.exit_id[done].max() >= self.exit_width.size:
            raise ValueError("exit_id refers to an exit without a width")

    @property
    def n(self) -> int:
        return int(self.exit_time.size)


# ------------------------------------------------------------ time metrics
def evacuated_fraction(exit_time: np.ndarray) -> float:
    exit_time = np.asarray(exit_time, dtype=float)
    return float(np.isfinite(exit_time).mean()) if exit_time.size else float("nan")


def evacuation_time(exit_time: np.ndarray) -> float:
    """Time the last evacuated person left (total evacuation time if everyone got out)."""
    t = np.asarray(exit_time, dtype=float)
    t = t[np.isfinite(t)]
    return float(t.max()) if t.size else float("nan")


def percentile_time(exit_time: np.ndarray, q: float) -> float:
    """Time by which q % of the whole crowd had left (inf if never reached)."""
    if not 0 < q <= 100:
        raise ValueError("q must be in (0, 100]")
    t = np.sort(np.asarray(exit_time, dtype=float))
    if t.size == 0:
        return float("nan")
    k = math.ceil(q / 100.0 * t.size)
    return float(t[k - 1])  # inf automatically if fewer than k finite times


def evacuation_curve(exit_time: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Cumulative number of people out at each time in t_grid."""
    t = np.sort(np.asarray(exit_time, dtype=float))
    return np.searchsorted(t, np.asarray(t_grid, dtype=float), side="right")


# ------------------------------------------------------------ exit metrics
def exit_counts(exit_id: np.ndarray, n_exits: int) -> np.ndarray:
    ids = np.asarray(exit_id, dtype=int)
    return np.bincount(ids[ids >= 0], minlength=n_exits)[:n_exits]


def exit_specific_flow(exit_time: np.ndarray, exit_id: np.ndarray, exit_width: np.ndarray,
                       lower: float = 0.1, upper: float = 0.9, min_people: int = 10) -> np.ndarray:
    """Steady specific flow through each exit [persons / (m s)].

    Uses the passages between that exit's ``lower`` and ``upper`` quantiles
    (dropping the start-up and tail-off), i.e. flow = (passages) / (duration * width).
    NaN for exits with fewer than ``min_people`` users. Compare with the
    bottleneck benchmark of roughly 1.3 p/(m s) reported in the literature.
    """
    exit_time = np.asarray(exit_time, dtype=float)
    exit_id = np.asarray(exit_id, dtype=int)
    widths = np.asarray(exit_width, dtype=float)
    out = np.full(widths.size, np.nan)
    for e in range(widths.size):
        t = np.sort(exit_time[(exit_id == e) & np.isfinite(exit_time)])
        if t.size < min_people:
            continue
        i0 = int(math.floor(lower * (t.size - 1)))
        i1 = int(math.ceil(upper * (t.size - 1)))
        dt = t[i1] - t[i0]
        if i1 > i0 and dt > 0:
            out[e] = (i1 - i0) / dt / widths[e]
    return out


def exit_clearance_times(exit_time: np.ndarray, exit_id: np.ndarray, n_exits: int) -> np.ndarray:
    """Last passage time at each exit (NaN if unused)."""
    exit_time = np.asarray(exit_time, dtype=float)
    exit_id = np.asarray(exit_id, dtype=int)
    out = np.full(n_exits, np.nan)
    for e in range(n_exits):
        t = exit_time[(exit_id == e) & np.isfinite(exit_time)]
        if t.size:
            out[e] = t.max()
    return out


def exit_balance(exit_time: np.ndarray, exit_id: np.ndarray, exit_width: np.ndarray,
                 open_exits: np.ndarray | None = None) -> dict:
    """How evenly the open exits were used.

    * ``load_cv``: coefficient of variation of people per metre of exit width
      (0 = perfectly balanced).
    * ``clearance_spread_s``: latest minus earliest exit clearance time.
    * ``unused_open_exits``: open exits nobody used.
    """
    widths = np.asarray(exit_width, dtype=float)
    n = widths.size
    open_mask = np.ones(n, bool) if open_exits is None else np.asarray(open_exits, bool)
    counts = exit_counts(exit_id, n)[open_mask]
    load = counts / widths[open_mask]
    clear = exit_clearance_times(exit_time, exit_id, n)[open_mask]
    used = np.isfinite(clear)
    return {
        "load_cv": float(load.std() / load.mean()) if load.size and load.mean() > 0 else float("nan"),
        "clearance_spread_s": float(np.nanmax(clear) - np.nanmin(clear)) if used.any() else float("nan"),
        "unused_open_exits": int((~used).sum()),
    }


# ------------------------------------------------------------ summary
def summarise(res: EvacuationResult, open_exits: np.ndarray | None = None) -> dict:
    """Headline numbers for one run (one row of the results table)."""
    flows = exit_specific_flow(res.exit_time, res.exit_id, res.exit_width)
    bal = exit_balance(res.exit_time, res.exit_id, res.exit_width, open_exits)
    return {
        "model": res.model, "scenario": res.scenario, "seed": res.seed, "agents": res.n,
        "evacuated_fraction": evacuated_fraction(res.exit_time),
        "T50_s": percentile_time(res.exit_time, 50),
        "T90_s": percentile_time(res.exit_time, 90),
        "T95_s": percentile_time(res.exit_time, 95),
        "T100_s": evacuation_time(res.exit_time),
        "mean_exit_flow_pms": float(np.nanmean(flows)) if np.isfinite(flows).any() else float("nan"),
        "max_exit_flow_pms": float(np.nanmax(flows)) if np.isfinite(flows).any() else float("nan"),
        **bal,
        "compute_time_s": res.compute_time_s,
    }


class Timer:
    """Wall-clock timer: ``with Timer() as t: ...; t.elapsed``."""

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        self.elapsed = float("nan")
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self._t0
