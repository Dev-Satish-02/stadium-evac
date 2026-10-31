"""Model 2 - Floor-field cellular automaton (Burstedde et al. 2001; Kirchner & Schadschneider 2002).

* Space: square cells (``micro.nav_cell`` = 0.4 m), at most one person per cell.
* Each step a person picks one of its 8 neighbouring cells (or stays) with
  probability  p ~ exp(-k_s * dS / l + k_d * D)  over free cells (l = step
  length, 1 or sqrt(2) cells, so progress is counted per metre walked), where S is the
  static floor field (distance to the nearest open exit, in cells) and D the
  dynamic floor field (a decaying trail left by moving people; its weight
  k_d is scaled by the scenario's herding, so it only acts when people follow
  each other).
* Heterogeneous speeds: a person attempts a move with probability v / v_max,
  where v_max = cell / dt; diagonal moves succeed with probability 1/sqrt(2)
  so that speed is the same in every direction.
* Density-dependent speed: v is multiplied by the shared SFPE speed-density
  ratio of the local density (other people in the 3 x 3 block), floored at
  ``min_speed_ratio``.
  Without it a 0.4 m CA passes ~4-5 persons/(m s) through a door, about
  twice the measured value; with it the CA is calibrated into the observed
  range (see the validation suite). Set min_speed_ratio: 1.0 to switch off.
* Conflicts (several people choose the same cell): one random winner; with
  probability mu (friction, increased by herding/panic) nobody moves.
* Parallel update; stepping onto an exit cell = evacuated at that time (the
  person occupies the exit cell for one step, so a door cell passes at most
  one person every two steps).
"""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy import ndimage

from crowdsim.models.domain import Domain
from crowdsim.models.movement import density_speed_ratio
from crowdsim.models.params import ModelParams

NAME = "ca"
LABEL = "Cellular automaton"

OFF9 = np.array([(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)])
INV_SQRT2 = 1.0 / np.sqrt(2.0)
STEP_LEN = np.hypot(OFF9[:, 0], OFF9[:, 1]).clip(min=1.0)  # 1 for stay/straight, sqrt(2) for diagonal


class CAEngine:
    def __init__(self, domain: Domain, params: ModelParams, rng: np.random.Generator, speed_all: np.ndarray,
                 herding: float = 0.0, v_max: float | None = None):
        self.d, self.rng = domain, rng
        self.cell = domain.cell
        self.v_all = np.asarray(speed_all, dtype=float)
        self.vmax = float(v_max if v_max is not None else max(self.v_all.max(), 0.1))
        self.dt = self.cell / self.vmax
        self.t = 0.0
        self.passable = domain.walk | (domain.exit_id >= 0)
        finite = np.isfinite(domain.S)
        big = (domain.S[finite].max() if finite.any() else 0.0) + 100.0
        self.S = np.where(finite, domain.S, big) / self.cell  # in cells
        self.occ = np.full(domain.shape, -1, dtype=np.int64)
        self.D = np.zeros(domain.shape)
        self.k_s = params.ca.k_s
        self.k_d = params.ca.k_d * herding
        self.mu = min(0.9, params.ca.friction + params.ca.friction_herding * herding)
        self.alpha, self.delta = params.ca.alpha, params.ca.delta
        self.mp = params.movement
        self.slowdown = params.ca.min_speed_ratio < 1.0
        self.min_ratio = params.ca.min_speed_ratio
        self.t_cross = np.full(self.v_all.size, np.inf)
        self.gid = np.empty(0, dtype=np.int64)
        self.ix = np.empty(0, dtype=np.int64)
        self.iy = np.empty(0, dtype=np.int64)
        self.inject_cells = {}
        for node, pts in domain.inject.items():
            ix, iy = domain.nav_index(pts[:, 0], pts[:, 1])
            cells = np.unique(np.column_stack([ix, iy]), axis=0)
            ok = domain.walk[cells[:, 0], cells[:, 1]]
            self.inject_cells[node] = cells[ok]

    # ------------------------------------------------------------ population
    @property
    def n(self) -> int:
        return int(self.gid.size)

    def add(self, gids, ix, iy) -> None:
        gids, ix, iy = (np.asarray(a, dtype=np.int64) for a in (gids, ix, iy))
        if np.any(self.occ[ix, iy] >= 0) or not np.all(self.domain_ok(ix, iy)):
            raise ValueError("CA: cannot place people on occupied or non-walkable cells")
        self.occ[ix, iy] = gids
        self.gid = np.r_[self.gid, gids]
        self.ix = np.r_[self.ix, ix]
        self.iy = np.r_[self.iy, iy]

    def domain_ok(self, ix, iy) -> np.ndarray:
        return self.d.walk[ix, iy]

    def place_random(self, gids: np.ndarray, region_mask: np.ndarray) -> None:
        """Place people on distinct random free cells inside region_mask (validation set-ups)."""
        cand = np.argwhere(region_mask & self.d.walk & (self.occ < 0))
        if len(cand) < len(gids):
            raise ValueError("CA: not enough free cells for the initial crowd")
        pick = cand[self.rng.choice(len(cand), size=len(gids), replace=False)]
        self.add(gids, pick[:, 0], pick[:, 1])

    def inject(self, queues: dict) -> np.ndarray:
        placed = []
        for node, q in queues.items():
            if not q:
                continue
            cells = self.inject_cells.get(node)
            if cells is None or len(cells) == 0:
                continue
            free = cells[self.occ[cells[:, 0], cells[:, 1]] < 0]
            k = min(len(free), len(q))
            if k == 0:
                continue
            g = np.array([q.popleft() for _ in range(k)], dtype=np.int64)
            self.add(g, free[:k, 0], free[:k, 1])
            placed.append(g)
        return np.concatenate(placed) if placed else np.empty(0, dtype=np.int64)

    def positions(self) -> np.ndarray:
        x, y = self.d.cell_center(self.ix, self.iy)
        return np.column_stack([x, y])

    # ------------------------------------------------------------ dynamics
    def _update_dynamic_field(self) -> None:
        if self.k_d <= 0:
            return
        D = self.D
        mean8 = (ndimage.uniform_filter(D, size=3, mode="constant") * 9.0 - D) / 8.0
        self.D = np.where(self.d.walk, (1.0 - self.delta) * ((1.0 - self.alpha) * D + self.alpha * mean8), 0.0)

    def step(self):
        """One CA step. Returns (gids, exit_ids, exit_times) of people who left."""
        rng = self.rng
        t_end = self.t + self.dt
        out_g, out_e, out_t = self._leave()
        n = self.n
        if n == 0:
            self.t = t_end
            self._update_dynamic_field()
            return out_g, out_e, out_t
        nx, ny = self.d.shape
        nix = np.clip(self.ix[:, None] + OFF9[None, :, 0], 0, nx - 1)
        niy = np.clip(self.iy[:, None] + OFF9[None, :, 1], 0, ny - 1)
        allowed = self.passable[nix, niy] & (self.occ[nix, niy] < 0)
        allowed[:, 0] = True
        # progress towards the exit per unit of distance stepped (diagonal steps are sqrt(2) long)
        expo = -self.k_s * (self.S[nix, niy] - self.S[self.ix, self.iy][:, None]) / STEP_LEN[None, :]
        if self.k_d > 0:
            expo += self.k_d * self.D[nix, niy]
        w = np.exp(np.clip(expo, -50.0, 50.0)) * allowed
        speed = self.v_all[self.gid]
        if self.slowdown:
            occ = (self.occ >= 0).astype(float)
            count = ndimage.uniform_filter(occ, size=3, mode="constant") * 9.0
            others = np.maximum(count[self.ix, self.iy] - 1.0, 0.0)  # neighbours in the 3 x 3 block
            dens = others / (9.0 * self.cell ** 2)
            g = np.maximum(density_speed_ratio(self.mp, dens), self.min_ratio)
            speed = speed * g
        attempt = rng.random(n) < speed / self.vmax
        w[~attempt, 1:] = 0.0
        w[~attempt, 0] = 1.0
        cum = np.cumsum(w, axis=1)
        r = rng.random(n) * cum[:, -1]
        choice = np.minimum((cum < r[:, None]).sum(axis=1), 8)
        diag = choice >= 5
        choice[diag & (rng.random(n) > INV_SQRT2)] = 0
        choice[self.d.exit_id[self.ix, self.iy] >= 0] = 0  # people on exit cells leave at the next step
        mv = np.flatnonzero(choice != 0)
        if mv.size:
            tx = nix[mv, choice[mv]]
            ty = niy[mv, choice[mv]]
            lin = tx * ny + ty
            perm = rng.permutation(mv.size)
            uniq, first, counts = np.unique(lin[perm], return_index=True, return_counts=True)
            win = perm[first]
            blocked = (counts > 1) & (rng.random(uniq.size) < self.mu)
            win = win[~blocked]
            a = mv[win]
            ox, oy = self.ix[a], self.iy[a]
            nxw, nyw = tx[win], ty[win]
            self.occ[ox, oy] = -1
            if self.k_d > 0:
                np.add.at(self.D, (ox, oy), 1.0)
            self.occ[nxw, nyw] = self.gid[a]
            self.ix[a] = nxw
            self.iy[a] = nyw
            ex = self.d.exit_id[nxw, nyw] >= 0
            self.t_cross[self.gid[a[ex]]] = t_end  # crossed the exit line now; removed next step
        self._update_dynamic_field()
        self.t = t_end
        return out_g, out_e, out_t

    def _leave(self):
        """Remove people standing on exit cells (they stepped there in the previous step)."""
        if self.n == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0)
        ex = self.d.exit_id[self.ix, self.iy]
        on = ex >= 0
        if not on.any():
            return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0)
        g = self.gid[on].copy()
        self.occ[self.ix[on], self.iy[on]] = -1
        keep = ~on
        self.gid, self.ix, self.iy = self.gid[keep], self.ix[keep], self.iy[keep]
        return g, ex[on].astype(np.int64), self.t_cross[g].copy()


def new_queues(nodes) -> dict:
    return {int(n): deque() for n in nodes}
