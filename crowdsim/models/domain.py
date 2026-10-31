"""2-D walkable domains for the microscopic models (CA and social force).

A Domain is rasterised from two predicates, walkable(x, y) and exit_id(x, y):

* navigation grid (``micro.nav_cell``, default 0.4 m): walkable cells, exit
  cells, the static floor field S (geodesic distance to the nearest open exit,
  8-neighbour Dijkstra, no corner cutting, with a cost penalty within 0.8 m of
  walls so routes do not hug walls and door jambs) and a smoothed walking
  direction;
* wall grid (``sf.wall_cell``, default 0.15 m): signed distance to the nearest
  wall surface and the wall normal, from Euclidean distance transforms.

Builders: :func:`concourse_domain` (the exit concourse of a stadium),
:func:`room_domain` and :func:`corridor_domain` (validation tests).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from crowdsim.models.params import ModelParams
from crowdsim.stadium.model import Stadium

SQRT2 = float(np.sqrt(2.0))
NEIGH8 = np.array([(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)])


@dataclass(eq=False)
class Domain:
    name: str
    cell: float
    x0: float
    y0: float
    walk: np.ndarray          # (nx, ny) bool
    exit_id: np.ndarray       # (nx, ny) int, -1 if not an exit cell
    S: np.ndarray             # (nx, ny) distance to nearest open exit [m], inf if none
    dirx: np.ndarray          # (nx, ny) unit walking direction
    diry: np.ndarray
    wcell: float
    wx0: float
    wy0: float
    sd: np.ndarray            # (mx, my) signed distance to wall surface [m] (>0 in free space)
    wnx: np.ndarray           # (mx, my) unit normal pointing away from the wall
    wny: np.ndarray
    exit_width: np.ndarray    # (n_exits,) indexed like exit ids
    exit_center: np.ndarray   # (n_exits, 2)
    inject: dict = field(default_factory=dict)  # node index -> (k, 2) spawn slot positions

    @property
    def shape(self) -> tuple[int, int]:
        return self.walk.shape

    @property
    def walk_area(self) -> float:
        return float(self.walk.sum() * self.cell ** 2)

    def nav_index(self, x, y):
        ix = np.clip(((np.asarray(x) - self.x0) / self.cell).astype(int), 0, self.walk.shape[0] - 1)
        iy = np.clip(((np.asarray(y) - self.y0) / self.cell).astype(int), 0, self.walk.shape[1] - 1)
        return ix, iy

    def cell_center(self, ix, iy):
        return self.x0 + (np.asarray(ix) + 0.5) * self.cell, self.y0 + (np.asarray(iy) + 0.5) * self.cell

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        nx, ny = self.walk.shape
        return self.x0, self.x0 + nx * self.cell, self.y0, self.y0 + ny * self.cell


# ---------------------------------------------------------------- generic builder
def _grid(bounds, cell):
    xmin, xmax, ymin, ymax = bounds
    nx = int(np.ceil((xmax - xmin) / cell))
    ny = int(np.ceil((ymax - ymin) / cell))
    xs = xmin + (np.arange(nx) + 0.5) * cell
    ys = ymin + (np.arange(ny) + 0.5) * cell
    return xs, ys


WALL_CLEARANCE = 0.8   # paths are discouraged from passing closer than this to a wall [m]
WALL_PENALTY = 2.0     # extra cost factor right next to a wall (fades to 0 at WALL_CLEARANCE)


def floor_field(passable: np.ndarray, sources: np.ndarray, cell: float, walls: np.ndarray | None = None) -> np.ndarray:
    """Geodesic travel cost [m] from the source cells over passable cells (8-neighbour, no corner cutting).

    If ``walls`` is given, cells near a wall cost more (people keep some distance
    from walls and door jambs instead of hugging corners)."""
    nx, ny = passable.shape
    cost = np.ones(passable.shape)
    if walls is not None and walls.any():
        dw = ndimage.distance_transform_edt(~walls, sampling=cell) - 0.5 * cell
        cost = 1.0 + WALL_PENALTY * np.clip(1.0 - dw / WALL_CLEARANCE, 0.0, 1.0)
    idx = -np.ones(passable.shape, dtype=np.int64)
    flat = np.flatnonzero(passable.ravel())
    idx.ravel()[flat] = np.arange(flat.size)
    ii, jj = np.unravel_index(flat, passable.shape)
    rows, cols, wts = [], [], []
    for di, dj in NEIGH8:
        ni, nj = ii + di, jj + dj
        ok = (ni >= 0) & (ni < nx) & (nj >= 0) & (nj < ny)
        ok[ok] &= passable[ni[ok], nj[ok]]
        if di != 0 and dj != 0:  # diagonal: both orthogonal neighbours must be passable
            a_ok = (ii + di >= 0) & (ii + di < nx)
            b_ok = (jj + dj >= 0) & (jj + dj < ny)
            ok &= a_ok & b_ok
            ok[ok] &= passable[ii[ok] + di, jj[ok]] & passable[ii[ok], jj[ok] + dj]
        rows.append(np.flatnonzero(ok))
        cols.append(idx[ni[ok], nj[ok]])
        step = cell * (SQRT2 if di and dj else 1.0)
        wts.append(step * 0.5 * (cost[ii[ok], jj[ok]] + cost[ni[ok], nj[ok]]))
    g = coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(flat.size, flat.size)).tocsr()
    src = idx[sources & passable]
    out = np.full(passable.shape, np.inf)
    if src.size:
        d = dijkstra(g, directed=True, indices=src, min_only=True)
        out.ravel()[flat] = d
    return out


def direction_field(S: np.ndarray, walk: np.ndarray, passable: np.ndarray):
    """Unit direction of steepest descent of S, smoothed over 3 x 3 cells."""
    nx, ny = S.shape
    dx = np.zeros(S.shape)
    dy = np.zeros(S.shape)
    Sp = np.pad(np.where(passable, S, np.inf), 1, constant_values=np.inf)
    cur = np.where(np.isfinite(S), S, np.inf)
    best = cur.copy()
    for di, dj in NEIGH8:
        nb = Sp[1 + di:1 + di + nx, 1 + dj:1 + dj + ny]
        better = nb < best - 1e-12
        best = np.where(better, nb, best)
        norm = np.hypot(di, dj)
        dx = np.where(better, di / norm, dx)
        dy = np.where(better, dj / norm, dy)
    dx = np.where(walk, dx, 0.0)
    dy = np.where(walk, dy, 0.0)
    w = walk.astype(float)
    sx = ndimage.uniform_filter(dx * w, size=3, mode="constant")
    sy = ndimage.uniform_filter(dy * w, size=3, mode="constant")
    n = np.hypot(sx, sy)
    ok = n > 1e-9
    ux = np.where(ok, sx / np.where(ok, n, 1.0), dx)
    uy = np.where(ok, sy / np.where(ok, n, 1.0), dy)
    return ux.astype(np.float64), uy.astype(np.float64)


def wall_field(free: np.ndarray, wcell: float):
    """Signed distance to the wall surface and outward wall normal on the wall grid."""
    d_in, ind_in = ndimage.distance_transform_edt(free, sampling=wcell, return_indices=True)
    d_out, ind_out = ndimage.distance_transform_edt(~free, sampling=wcell, return_indices=True)
    ii, jj = np.indices(free.shape)
    # free cells: vector from nearest wall cell to here; wall cells: from here to nearest free cell
    vx = np.where(free, ii - ind_in[0], ind_out[0] - ii).astype(float)
    vy = np.where(free, jj - ind_in[1], ind_out[1] - jj).astype(float)
    n = np.hypot(vx, vy)
    n[n == 0] = 1.0
    sd = np.where(free, d_in - 0.5 * wcell, -(d_out - 0.5 * wcell))
    return sd.astype(np.float64), (vx / n).astype(np.float64), (vy / n).astype(np.float64)


def build_domain(name: str, bounds, walk_fn, exit_fn, exit_width, exit_center, params: ModelParams,
                 inject: dict | None = None) -> Domain:
    cell, wcell = params.micro.nav_cell, params.sf.wall_cell
    xs, ys = _grid(bounds, cell)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    walk = walk_fn(X, Y)
    exit_id = np.where(walk, -1, exit_fn(X, Y)).astype(int)
    passable = walk | (exit_id >= 0)
    S = floor_field(passable, exit_id >= 0, cell, walls=~passable)
    S = np.where(exit_id >= 0, 0.0, S)
    dirx, diry = direction_field(S, walk, passable)
    wxs, wys = _grid(bounds, wcell)
    WX, WY = np.meshgrid(wxs, wys, indexing="ij")
    free = walk_fn(WX, WY) | (exit_fn(WX, WY) >= 0)
    sd, wnx, wny = wall_field(free, wcell)
    return Domain(name=name, cell=cell, x0=bounds[0], y0=bounds[2], walk=walk, exit_id=exit_id, S=S,
                  dirx=dirx, diry=diry, wcell=wcell, wx0=bounds[0], wy0=bounds[2], sd=sd, wnx=wnx, wny=wny,
                  exit_width=np.asarray(exit_width, float), exit_center=np.asarray(exit_center, float).reshape(-1, 2),
                  inject=inject or {})


# ---------------------------------------------------------------- builders
def room_domain(width: float, depth: float, door: float, params: ModelParams, name: str = "room") -> Domain:
    """Rectangular room [0, width] x [0, depth] with one door of the given width centred in the wall x = width."""
    ed = params.micro.exit_depth
    yc = depth / 2.0

    def walk_fn(X, Y):
        return (X > 0) & (X < width) & (Y > 0) & (Y < depth)

    def exit_fn(X, Y):
        return np.where((X >= width) & (X < width + ed) & (np.abs(Y - yc) <= door / 2.0), 0, -1)

    return build_domain(name, (-1.0, width + ed + 1.0, -1.0, depth + 1.0), walk_fn, exit_fn, [door],
                        [(width, yc)], params)


def corridor_domain(length: float, width: float, params: ModelParams) -> Domain:
    """Straight corridor open at x = length (the exit spans the full width)."""
    return room_domain(length, width, width + 1.0, params, name="corridor")


class _CurveLocator:
    """Fast (s, d) bowl coordinates for many points (nearest curve sample + tangent projection)."""

    def __init__(self, stadium: Stadium):
        c = stadium.curve
        self.curve = c
        self.tree = cKDTree(c.points)
        self.tang = np.column_stack([-c.normals[:, 1], c.normals[:, 0]])
        self.ds = c.length / c.m

    def sd(self, X, Y):
        p = np.column_stack([np.ravel(X), np.ravel(Y)])
        _, i = self.tree.query(p)
        rel = p - self.curve.points[i]
        d = np.einsum("ij,ij->i", rel, self.curve.normals[i])
        t = np.einsum("ij,ij->i", rel, self.tang[i])
        s = ((i + t / self.ds) / self.curve.m) % 1.0
        return s.reshape(np.shape(X)), d.reshape(np.shape(X))


def concourse_domain(stadium: Stadium, level: str, params: ModelParams, inject_nodes=()) -> Domain:
    """The ring-shaped concourse `level` with its open gates as exits.

    Exit ids are indices into ``stadium.exits``. ``inject_nodes`` are network node
    indices where people arrive (vomitory landings, stair/ramp feet); each gets
    spawn slots spread across the width of the element that leads there.
    """
    spec = stadium.spec.concourse(level)
    d_in, d_out = spec.d_inner, spec.d_outer
    ed = params.micro.exit_depth
    loc = _CurveLocator(stadium)
    curve = stadium.curve
    ring_out = curve.ring_length(d_out)
    gates = [(i, e) for i, e in enumerate(stadium.exits) if e.kind == "gate" and e.level == level and e.enabled]
    gate_arc = np.array([float(curve.arc(e.s, d_out)) for _, e in gates])
    gate_w = np.array([e.width for _, e in gates])
    gate_ids = np.array([i for i, _ in gates], dtype=int)

    def walk_fn(X, Y):
        _, d = loc.sd(X, Y)
        return (d >= d_in) & (d <= d_out)

    def exit_fn(X, Y):
        s, d = loc.sd(X, Y)
        out = np.full(np.shape(X), -1, dtype=int)
        band = (d > d_out) & (d <= d_out + ed)
        if not gates or not band.any():
            return out
        a = curve.arc(s[band], d_out)
        diff = (a[:, None] - gate_arc[None, :] + ring_out / 2.0) % ring_out - ring_out / 2.0
        inside = np.abs(diff) <= gate_w[None, :] / 2.0
        hit = inside.any(axis=1)
        vals = np.full(a.size, -1, dtype=int)
        vals[hit] = gate_ids[np.argmax(inside[hit], axis=1)]
        out[band] = vals
        return out

    poly = curve.polyline(d_out + ed + 1.5)
    bounds = (poly[:, 0].min(), poly[:, 0].max(), poly[:, 1].min(), poly[:, 1].max())
    widths = np.array([e.width for e in stadium.exits], dtype=float)
    centers = np.array([[e.x, e.y] for e in stadium.exits], dtype=float)

    g = stadium.graph
    inject = {}
    for ni in inject_nodes:
        name = stadium.nodes[ni]
        nd = g.nodes[name]
        w = max([dd["width"] for _, _, dd in g.edges(name, data=True)
                 if dd["kind"] in ("vomitory", "stair", "ramp")] or [2.0])
        s, _ = loc.sd(np.array([nd["x"]]), np.array([nd["y"]]))
        tang = curve.xy(s[0] + 1e-4, 0.5 * (d_in + d_out)) - curve.xy(s[0] - 1e-4, 0.5 * (d_in + d_out))
        tang = tang / np.linalg.norm(tang)
        k = max(1, int(np.floor(w / 0.5)))
        offs = (np.arange(k) - (k - 1) / 2.0) * 0.5
        inject[int(ni)] = np.array([nd["x"], nd["y"]]) + offs[:, None] * tang[None, :]
    return build_domain(f"{stadium.spec.key}:{level}", bounds, walk_fn, exit_fn, widths, centers, params, inject)
