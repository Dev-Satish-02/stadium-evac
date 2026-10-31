"""Model 3 - Social force model (Helbing, Farkas & Vicsek, Nature 2000).

m dv_i/dt = m (v0_i e_i - v_i) / tau
          + sum_j [ A exp((r_ij - d_ij)/B) + k g(r_ij - d_ij) ] n_ij + kappa g(r_ij - d_ij) dv_ji^t t_ij
          + sum_W [ A exp((r_i - d_iW)/B) + k g(r_i - d_iW) ] n_iW - kappa g(r_i - d_iW) (v_i . t_iW) t_iW

with g(x) = x if x > 0 else 0 (body force and sliding friction only on contact).
Integration: semi-implicit Euler; the stiff sliding-friction terms are treated
implicitly (a 2 x 2 solve per person), which keeps Helbing's kappa stable at
practical time steps.
The desired direction e_i follows the floor field of the domain; with herding
p > 0 (panic) it becomes  e_i = norm((1 - p) e_i0 + p <e_j>)  averaged over
neighbours within herd_radius. Speeds are capped at vmax_factor * v0_i.
A small random force (individual fluctuations, Helbing & Molnar 1995; std
``noise`` N per component, redrawn every step) lets arches at narrow doors
break up instead of clogging forever.
Neighbour search uses a cell list; walls come from the domain's signed
distance field. Kernels are compiled with numba.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit
from scipy.spatial import cKDTree

from crowdsim.models.domain import Domain
from crowdsim.models.params import ModelParams

NAME = "sf"
LABEL = "Social force"


@njit(cache=True)
def _cell_list(x, y, n, gx0, gy0, gcell, gnx, gny, start, count, order):
    for c in range(gnx * gny):
        count[c] = 0
    cell = np.empty(n, dtype=np.int64)
    for i in range(n):
        cx = int((x[i] - gx0) / gcell)
        cy = int((y[i] - gy0) / gcell)
        cx = min(max(cx, 0), gnx - 1)
        cy = min(max(cy, 0), gny - 1)
        c = cx * gny + cy
        cell[i] = c
        count[c] += 1
    s = 0
    for c in range(gnx * gny):
        start[c] = s
        s += count[c]
    fill = start.copy()
    for i in range(n):
        c = cell[i]
        order[fill[c]] = i
        fill[c] += 1


@njit(cache=True)
def _sf_step(x, y, vx, vy, v0, r, n, dt, tau, mass, A, B, kb, kap, cutoff, vmax_f, herd_p, herd_r,
             nx0, ny0, ncell, dirx, diry, wx0, wy0, wcell, sd, wnx, wny,
             gx0, gy0, gcell, gnx, gny, start, count, order, nvx, nvy, noise_x, noise_y):
    nnx, nny = dirx.shape
    mwx, mwy = sd.shape
    cut2 = cutoff * cutoff
    herd2 = herd_r * herd_r
    for i in range(n):
        xi, yi, vxi, vyi, ri = x[i], y[i], vx[i], vy[i], r[i]
        ix = min(max(int((xi - nx0) / ncell), 0), nnx - 1)
        iy = min(max(int((yi - ny0) / ncell), 0), nny - 1)
        ex, ey = dirx[ix, iy], diry[ix, iy]
        fx, fy = 0.0, 0.0
        hx, hy, hn = 0.0, 0.0, 0
        # semi-implicit sliding friction: (I + dt/m * M) v_new = v + dt/m * (F + Fj)
        m11, m12, m22 = 0.0, 0.0, 0.0
        fjx, fjy = 0.0, 0.0
        cx = min(max(int((xi - gx0) / gcell), 0), gnx - 1)
        cy = min(max(int((yi - gy0) / gcell), 0), gny - 1)
        for ddx in range(-1, 2):
            ccx = cx + ddx
            if ccx < 0 or ccx >= gnx:
                continue
            for ddy in range(-1, 2):
                ccy = cy + ddy
                if ccy < 0 or ccy >= gny:
                    continue
                c = ccx * gny + ccy
                for k in range(start[c], start[c] + count[c]):
                    j = order[k]
                    if j == i:
                        continue
                    dx = xi - x[j]
                    dy = yi - y[j]
                    d2 = dx * dx + dy * dy
                    if herd_p > 0.0 and d2 < herd2:
                        sp = math.sqrt(vx[j] * vx[j] + vy[j] * vy[j])
                        if sp > 0.1:
                            hx += vx[j] / sp
                            hy += vy[j] / sp
                            hn += 1
                    if d2 >= cut2 or d2 < 1e-12:
                        continue
                    d = math.sqrt(d2)
                    nxij, nyij = dx / d, dy / d
                    over = ri + r[j] - d
                    f = A * math.exp(min(over / B, 20.0))
                    if over > 0.0:
                        f += kb * over
                        tx, ty = -nyij, nxij
                        c = kap * over
                        vjt = vx[j] * tx + vy[j] * ty
                        fjx += c * vjt * tx
                        fjy += c * vjt * ty
                        m11 += c * tx * tx
                        m12 += c * tx * ty
                        m22 += c * ty * ty
                    fx += f * nxij
                    fy += f * nyij
        if herd_p > 0.0 and hn > 0:
            mx, my = hx / hn, hy / hn
            ex = (1.0 - herd_p) * ex + herd_p * mx
            ey = (1.0 - herd_p) * ey + herd_p * my
            en = math.sqrt(ex * ex + ey * ey)
            if en > 1e-9:
                ex /= en
                ey /= en
        fx += mass * (v0[i] * ex - vxi) / tau + noise_x[i]
        fy += mass * (v0[i] * ey - vyi) / tau + noise_y[i]
        wi = min(max(int((xi - wx0) / wcell), 0), mwx - 1)
        wj = min(max(int((yi - wy0) / wcell), 0), mwy - 1)
        over = ri - sd[wi, wj]
        if over > -cutoff:
            nwx, nwy = wnx[wi, wj], wny[wi, wj]
            f = A * math.exp(min(over / B, 20.0))
            if over > 0.0:
                f += kb * over
                tx, ty = -nwy, nwx
                c = kap * over
                m11 += c * tx * tx
                m12 += c * tx * ty
                m22 += c * ty * ty
            fx += f * nwx
            fy += f * nwy
        h = dt / mass
        bx = vxi + h * (fx + fjx)
        by = vyi + h * (fy + fjy)
        a11 = 1.0 + h * m11
        a12 = h * m12
        a22 = 1.0 + h * m22
        det = a11 * a22 - a12 * a12
        nvxi = (a22 * bx - a12 * by) / det
        nvyi = (a11 * by - a12 * bx) / det
        sp = math.sqrt(nvxi * nvxi + nvyi * nvyi)
        vcap = vmax_f * v0[i]
        if sp > vcap:
            nvxi *= vcap / sp
            nvyi *= vcap / sp
        nvx[i] = nvxi
        nvy[i] = nvyi
    for i in range(n):
        vx[i] = nvx[i]
        vy[i] = nvy[i]
        x[i] += vx[i] * dt
        y[i] += vy[i] * dt


class SFEngine:
    def __init__(self, domain: Domain, params: ModelParams, rng: np.random.Generator, speed_all: np.ndarray,
                 radius_all: np.ndarray, herding: float = 0.0, capacity: int | None = None):
        self.d, self.p, self.rng = domain, params.sf, rng
        self.v_all = np.asarray(speed_all, dtype=float)
        self.r_all = np.asarray(radius_all, dtype=float) + params.sf.space_margin
        self.herd = float(herding)
        self.dt = params.sf.dt
        self.t = 0.0
        cap = int(capacity if capacity is not None else self.v_all.size)
        self.x, self.y = np.zeros(cap), np.zeros(cap)
        self.vx, self.vy = np.zeros(cap), np.zeros(cap)
        self.v0, self.r = np.zeros(cap), np.zeros(cap)
        self.gid = np.zeros(cap, dtype=np.int64)
        self.nvx, self.nvy = np.zeros(cap), np.zeros(cap)
        self._zeros = np.zeros(cap)
        self._n = 0
        xmin, xmax, ymin, ymax = domain.bounds
        self.bounds = (xmin, xmax, ymin, ymax)
        self.gcell = max(params.sf.cutoff, params.sf.herd_radius if herding > 0 else 0.0)
        self.gnx = int(np.ceil((xmax - xmin) / self.gcell)) + 1
        self.gny = int(np.ceil((ymax - ymin) / self.gcell)) + 1
        self.start = np.zeros(self.gnx * self.gny, dtype=np.int64)
        self.count = np.zeros(self.gnx * self.gny, dtype=np.int64)
        self.order = np.zeros(cap, dtype=np.int64)
        self.escaped = 0
        self.slot_radius = 0.55

    @property
    def n(self) -> int:
        return self._n

    def _noise(self, n: int):
        if self.p.noise <= 0:
            return self._zeros[:n], self._zeros[:n]
        f = self.rng.normal(0.0, self.p.noise, size=(2, n))
        return f[0], f[1]

    def add(self, gids, xs, ys) -> None:
        gids = np.asarray(gids, dtype=np.int64)
        k = gids.size
        if k == 0:
            return
        s = slice(self._n, self._n + k)
        self.x[s], self.y[s] = xs, ys
        ix, iy = self.d.nav_index(xs, ys)
        v0 = self.v_all[gids]
        self.vx[s] = 0.5 * v0 * self.d.dirx[ix, iy]
        self.vy[s] = 0.5 * v0 * self.d.diry[ix, iy]
        self.v0[s], self.r[s], self.gid[s] = v0, self.r_all[gids], gids
        self._n += k

    def place_random(self, gids: np.ndarray, region_mask: np.ndarray) -> None:
        """Validation set-ups: distinct random nav cells (0.4 m apart) inside region_mask."""
        cand = np.argwhere(region_mask & self.d.walk & (self.d.S < np.inf))
        if len(cand) < len(gids):
            raise ValueError("SF: not enough space for the initial crowd")
        pick = cand[self.rng.choice(len(cand), size=len(gids), replace=False)]
        x, y = self.d.cell_center(pick[:, 0], pick[:, 1])
        self.add(gids, x, y)

    def inject(self, queues: dict) -> np.ndarray:
        placed = []
        tree = cKDTree(np.column_stack([self.x[:self._n], self.y[:self._n]])) if self._n else None
        for node, q in queues.items():
            if not q:
                continue
            slots = self.d.inject.get(node)
            if slots is None:
                continue
            if tree is not None:
                dist, _ = tree.query(slots)
                free = slots[dist > self.slot_radius]
            else:
                free = slots
            k = min(len(free), len(q))
            if k == 0:
                continue
            g = np.array([q.popleft() for _ in range(k)], dtype=np.int64)
            self.add(g, free[:k, 0], free[:k, 1])
            placed.append(g)
        return np.concatenate(placed) if placed else np.empty(0, dtype=np.int64)

    def positions(self) -> np.ndarray:
        return np.column_stack([self.x[:self._n], self.y[:self._n]])

    def step(self):
        """One social-force step. Returns (gids, exit_ids, exit_times) of people who left."""
        n, p, d = self._n, self.p, self.d
        t_end = self.t + self.dt
        if n == 0:
            self.t = t_end
            return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0)
        gx0, gy0 = self.bounds[0], self.bounds[2]
        _cell_list(self.x, self.y, n, gx0, gy0, self.gcell, self.gnx, self.gny, self.start, self.count, self.order)
        _sf_step(self.x, self.y, self.vx, self.vy, self.v0, self.r, n, self.dt, p.tau, p.mass, p.A, p.B,
                 p.k_body, p.kappa, p.cutoff, p.vmax_factor, self.herd, p.herd_radius,
                 d.x0, d.y0, d.cell, d.dirx, d.diry, d.wx0, d.wy0, d.wcell, d.sd, d.wnx, d.wny,
                 gx0, gy0, self.gcell, self.gnx, self.gny, self.start, self.count, self.order, self.nvx, self.nvy,
                 *self._noise(n))
        xmin, xmax, ymin, ymax = self.bounds
        x, y = self.x[:n], self.y[:n]
        out = (x < xmin) | (x >= xmax) | (y < ymin) | (y >= ymax) | ~np.isfinite(x) | ~np.isfinite(y)
        if out.any():  # numerical escape (should not happen); park them inside and count
            self.escaped += int(out.sum())
            np.clip(x, xmin + 1e-6, xmax - 1e-6, out=x)
            np.clip(y, ymin + 1e-6, ymax - 1e-6, out=y)
            x[~np.isfinite(x)] = xmin + 1e-6
            y[~np.isfinite(y)] = ymin + 1e-6
        ix, iy = d.nav_index(x, y)
        ex = d.exit_id[ix, iy]
        leaving = ex >= 0
        self.t = t_end
        if not leaving.any():
            return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0)
        gone_g, gone_e = self.gid[:n][leaving].copy(), ex[leaving].astype(np.int64)
        keep = ~leaving
        k = int(keep.sum())
        for arr in (self.x, self.y, self.vx, self.vy, self.v0, self.r, self.gid):
            arr[:k] = arr[:n][keep]
        self._n = k
        return gone_g, gone_e, np.full(gone_g.size, t_end)
