"""Stand-alone microscopic runs on simple domains (rooms, corridors) for validation."""
from __future__ import annotations

import numpy as np

from crowdsim.models.ca import CAEngine
from crowdsim.models.domain import Domain
from crowdsim.models.params import ModelParams
from crowdsim.models.sf import SFEngine


def lattice_cells(domain: Domain, n: int, step: int = 2, rng: np.random.Generator | None = None,
                  x_max: float | None = None) -> np.ndarray:
    """n walkable nav cells on a lattice `step` cells apart (0.8 m for step 2), chosen at random."""
    ix, iy = np.indices(domain.shape)
    m = domain.walk & (ix % step == 0) & (iy % step == 0) & np.isfinite(domain.S)
    if x_max is not None:
        cx, _ = domain.cell_center(ix, iy)
        m &= cx <= x_max
    cand = np.argwhere(m)
    if len(cand) < n:
        raise ValueError(f"only {len(cand)} lattice cells for {n} people")
    rng = rng or np.random.default_rng(0)
    return cand[np.sort(rng.choice(len(cand), size=n, replace=False))]


def run_room(kind: str, domain: Domain, cells: np.ndarray, speed: np.ndarray, radius: np.ndarray,
             params: ModelParams, seed: int, t_max: float = 600.0, herding: float = 0.0):
    """Evacuate people starting at `cells`. Returns (exit_time, exit_id, engine)."""
    n = len(cells)
    rng = np.random.default_rng(seed)
    if kind == "ca":
        eng = CAEngine(domain, params, rng, speed, herding)
        eng.add(np.arange(n), cells[:, 0], cells[:, 1])
    elif kind == "sf":
        eng = SFEngine(domain, params, rng, speed, radius, herding)
        x, y = domain.cell_center(cells[:, 0], cells[:, 1])
        eng.add(np.arange(n), x, y)
    else:
        raise ValueError(kind)
    times = np.full(n, np.inf)
    ids = np.full(n, -1, dtype=int)
    while eng.n > 0 and eng.t < t_max:
        g, e, tt = eng.step()
        times[g] = tt
        ids[g] = e
    return times, ids, eng
