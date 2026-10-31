"""Records where every person is during a run (for replay videos).

Positions are plan-view (x, y). People still on the network part of a route
(rows, aisles, vomitories, stairs, ramps) are placed along their route from
the network model's progress on the element; people on the exit concourse
use the true 2-D positions of the cellular-automaton / social-force model.
Evacuated people are NaN (not drawn).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crowdsim.models.network import DONE, EDGE, MICRO, PRE, QUEUE, ROW, STUCK

SEATED, ACTIVE, JAMMED = 0, 1, 2
LANE = 0.55  # spacing of drawn queue positions [m]


@dataclass(eq=False)
class Trace:
    """Recorded frames of one run."""
    t: np.ndarray            # (F,) simulated time [s]
    xy: np.ndarray           # (F, N, 2) float32, NaN = not on the plan (evacuated)
    state: np.ndarray        # (F, N) uint8: SEATED / ACTIVE / JAMMED
    exit_time: np.ndarray    # (N,) final exit times, for the evacuation curve
    model: str


class Tracer:
    """Pass to a model run; it calls ``snap`` every simulation step and keeps one frame per ``frame_dt``."""

    def __init__(self, stadium, net, crowd, frame_dt: float = 2.0):
        seats, idx = stadium.seats, crowd.labels["seat"]
        self.seat_xy = np.column_stack([seats.x[idx], seats.y[idx]]).astype(float)
        self.net, self.crowd = net, crowd
        self.lane_pos = (np.arange(crowd.n) * 0.6180339887) % 1.0 - 0.5  # in [-0.5, 0.5)
        self.frame_dt = float(frame_dt)
        self._next = 0.0
        self._t: list[float] = []
        self._xy: list[np.ndarray] = []
        self._state: list[np.ndarray] = []

    def positions(self, t: float, eng, micro: dict | None = None):
        """(xy, state) of everyone right now; evacuated people are NaN."""
        net, crowd = self.net, self.crowd
        n = crowd.n
        st = eng.status
        xy = np.full((n, 2), np.nan)
        state = np.full(n, ACTIVE, dtype=np.uint8)

        m = st == PRE
        xy[m] = self.seat_xy[m]
        state[m] = SEATED

        m = np.flatnonzero(st == ROW)
        if m.size:
            lat = crowd.lateral[m]
            frac = np.where(lat > 1e-9, 1.0 - eng.remain[m] / np.maximum(lat, 1e-9), 1.0).clip(0, 1)[:, None]
            node_xy = net.node_xyz[eng.node[m], :2]
            xy[m] = self.seat_xy[m] + frac * (node_xy - self.seat_xy[m])

        # people waiting at a junction stand in a queue on the element they arrived on
        # (lanes 0.55 m apart across its width, rows 0.55 m apart back from the junction)
        m = np.flatnonzero((st == QUEUE) | (st == MICRO) | (st == STUCK))
        if m.size:
            node_xy = net.node_xyz[eng.node[m], :2]
            pe = eng.prev_edge[m]
            key = np.where(pe >= 0, pe, -1 - eng.node[m])
            order = np.lexsort((m, eng.t_queue[m], key))
            ks = key[order]
            first = np.r_[True, ks[1:] != ks[:-1]]
            start = np.maximum.accumulate(np.where(first, np.arange(ks.size), 0))
            rank = np.empty(m.size, dtype=int)
            rank[order] = np.arange(ks.size) - start
            on_edge = pe >= 0
            pos = node_xy.copy()
            if on_edge.any():
                e = pe[on_edge]
                lanes = np.maximum(1, (net.w_eff[e] / LANE).astype(int))
                r = rank[on_edge]
                back = np.minimum((r // lanes) * LANE, net.length[e])
                far = net.node_xyz[net.other_end(e, eng.node[m][on_edge]), :2]
                u = far - node_xy[on_edge]
                L = np.maximum(np.hypot(u[:, 0], u[:, 1]), 1e-9)[:, None]
                u = u / L
                perp = np.column_stack([-u[:, 1], u[:, 0]])
                lane = (r % lanes) - (lanes - 1) / 2.0
                pos[on_edge] = node_xy[on_edge] + u * back[:, None] + perp * (lane * LANE)[:, None]
            if (~on_edge).any():  # still at the end of their row: queue back along the row
                mm = m[~on_edge]
                v = self.seat_xy[mm] - node_xy[~on_edge]
                L = np.maximum(np.hypot(v[:, 0], v[:, 1]), 1e-9)
                back = np.minimum(rank[~on_edge] * LANE, L)
                pos[~on_edge] = node_xy[~on_edge] + v * (back / L)[:, None]
            xy[m] = pos
        state[st == QUEUE] = JAMMED
        state[st == STUCK] = JAMMED

        m = np.flatnonzero(st == EDGE)
        if m.size:
            e = eng.edge[m]
            frac = (1.0 - eng.remain[m] / np.maximum(net.length[e], 1e-9)).clip(0, 1)[:, None]
            a = net.node_xyz[eng.node[m], :2]
            b = net.node_xyz[eng.target[m], :2]
            u = b - a
            L = np.maximum(np.hypot(u[:, 0], u[:, 1]), 1e-9)[:, None]
            perp = np.column_stack([-u[:, 1], u[:, 0]]) / L
            off = (self.lane_pos[m] * 0.8 * net.w_eff[e])[:, None]  # fixed place across the width per person
            xy[m] = a + frac * u + perp * off

        if micro:
            for mic in micro.values():
                if mic.n:
                    gid = mic.gid[: mic.n]
                    xy[gid] = mic.positions()

        xy[st == DONE] = np.nan
        return xy, state

    def due(self, t: float) -> bool:
        if t + 1e-9 < self._next:
            return False
        self._next = t + self.frame_dt
        return True

    def snap(self, t: float, eng, micro: dict | None = None) -> None:
        if not self.due(t):
            return
        xy, state = self.positions(t, eng, micro)
        self._t.append(float(t))
        self._xy.append(xy.astype(np.float32))
        self._state.append(state)

    def finish(self, exit_time: np.ndarray, model: str) -> Trace:
        return Trace(t=np.array(self._t), xy=np.stack(self._xy), state=np.stack(self._state),
                     exit_time=np.asarray(exit_time, dtype=float), model=model)
