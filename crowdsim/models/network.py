"""Model 4 - Mesoscopic queueing network with congestion-aware routing.

People move element by element along the navigation graph:

* each element admits at most C_e = Fs_max * W_eff persons/s (SFPE capacity),
  and stores at most jam_density * length * W_eff persons; a person waiting
  at the end of an element still occupies it, so queues spill back upstream;
* walking speed on an element = desired speed x element factor x SFPE
  density ratio of the people walking on it (people slow down in a dense
  moving crowd; people already queueing at the element's end limit its
  storage but leave at the element capacity - a spatial-queue model);
* at every junction a person picks the next element: 'static' routing takes
  the shortest route, 'dynamic' routing minimises walk time + expected wait
  + remaining distance, and with probability = herding they follow the
  busiest option instead (herd behaviour).

Only moves that get strictly closer to an exit are allowed, so routes cannot
loop and the network cannot gridlock.

The same engine drives the hybrid CA / social-force models upstream of the
exit concourse (``micro_nodes``): a person reaching such a node is handed over
and keeps occupying the element they arrived on until the micro model has
room to place them (two-way coupling).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crowdsim.metrics import EvacuationResult, Timer
from crowdsim.models.base import Crowd, RunOutput
from crowdsim.models.egressnet import EgressNet
from crowdsim.models.movement import density_speed_ratio
from crowdsim.models.params import ModelParams

NAME = "network"
LABEL = "Network (queueing)"

PRE, ROW, QUEUE, EDGE, MICRO, DONE, STUCK = range(7)
MIN_DENSITY_AREA = 4.0  # m^2 - smaller elements are left out of the density statistic


class NetworkEngine:
    def __init__(self, net: EgressNet, crowd: Crowd, params: ModelParams, rng: np.random.Generator,
                 micro_nodes: np.ndarray | None = None, routing: str | None = None):
        self.net, self.crowd, self.p, self.rng = net, crowd, params, rng
        self.mp = params.movement
        n, E = crowd.n, net.n_edges
        self.status = np.full(n, PRE, dtype=np.int8)
        self.node = crowd.start_node.astype(int).copy()
        self.edge = np.full(n, -1, dtype=int)       # edge being walked, or chosen next edge while queued
        self.target = np.full(n, -1, dtype=int)
        self.prev_edge = np.full(n, -1, dtype=int)  # element still occupied while waiting at its end
        self.remain = crowd.lateral.astype(float).copy()
        self.t_queue = np.zeros(n)
        self.exit_time = np.full(n, np.inf)
        self.exit_id = np.full(n, -1, dtype=int)
        self.occ = np.zeros(E)
        self.tokens = np.zeros(E)
        self.qcount = np.zeros(E)
        self.micro = np.zeros(net.n_nodes, dtype=bool)
        if micro_nodes is not None and len(micro_nodes):
            self.micro[np.asarray(micro_nodes, dtype=int)] = True
        self.dynamic = (routing or params.network.routing) == "dynamic"
        self.herding = float(crowd.herding)
        self.area = net.length * net.w_eff
        self.big = self.area >= MIN_DENSITY_AREA
        reach = np.isfinite(net.dist[self.node])
        self.status[~reach] = STUCK

    # ------------------------------------------------------------ helpers
    def _choose(self, idx: np.ndarray) -> np.ndarray:
        net = self.net
        nodes = self.node[idx]
        if not self.dynamic:
            return net.next_edge[nodes]
        ce, cn = net.cand_edge[nodes], net.cand_node[nodes]
        valid = ce >= 0
        cef = np.where(valid, ce, 0)
        cnf = np.where(valid, cn, 0)
        v = self.crowd.speed[idx][:, None]
        f = np.where(net.edge_u[cef] == nodes[:, None], net.f_uv[cef], net.f_vu[cef])
        load = self.qcount[cef] + self.occ[cef]
        cost = net.length[cef] / (v * f) + load / net.capacity[cef] + net.dist[cnf] / v
        cost = np.where(valid, cost, np.inf)
        pick = np.argmin(cost, axis=1)
        if self.herding > 0:
            herd = self.rng.random(idx.size) < self.herding
            if herd.any():
                busy = np.where(valid, load, -1.0)
                pick = np.where(herd, np.argmax(busy, axis=1), pick)
        return ce[np.arange(idx.size), pick]

    def _arrive(self, idx: np.ndarray, nodes: np.ndarray, from_edge: np.ndarray, t_arr: np.ndarray,
                handed: list) -> None:
        net = self.net
        self.node[idx] = nodes
        ex = net.exit_of_node[nodes]
        done = ex >= 0
        if done.any():
            d = idx[done]
            self.status[d] = DONE
            self.exit_time[d] = t_arr[done]
            self.exit_id[d] = ex[done]
            fe = from_edge[done]
            np.subtract.at(self.occ, fe[fe >= 0], 1.0)
            self.edge[d] = -1
        rest = ~done
        mic = rest & self.micro[nodes]
        if mic.any():
            m = idx[mic]
            self.status[m] = MICRO
            self.prev_edge[m] = from_edge[mic]
            self.edge[m] = -1
            self.t_queue[m] = t_arr[mic]
            handed.append(m)
        q = rest & ~self.micro[nodes]
        if q.any():
            m = idx[q]
            self.prev_edge[m] = from_edge[q]
            self.t_queue[m] = t_arr[q]
            nxt = self._choose(m)
            ok = nxt >= 0
            self.status[m[ok]] = QUEUE
            self.edge[m[ok]] = nxt[ok]
            self.status[m[~ok]] = STUCK

    def release(self, idx: np.ndarray) -> None:
        """The micro model has placed these handed-over people: free the element they occupied."""
        pe = self.prev_edge[idx]
        np.subtract.at(self.occ, pe[pe >= 0], 1.0)
        self.prev_edge[idx] = -1

    def finish(self, idx: np.ndarray, t: np.ndarray, exit_ids: np.ndarray) -> None:
        self.status[idx] = DONE
        self.exit_time[idx] = t
        self.exit_id[idx] = exit_ids

    # ------------------------------------------------------------ one step
    def step(self, t: float, dt: float) -> np.ndarray:
        """Advance from t to t + dt. Returns people handed to the micro model in this step."""
        net, mp = self.net, self.mp
        st = self.status
        handed: list[np.ndarray] = []

        t_end = t + dt
        lead = np.full(self.crowd.n, dt)
        starting = (st == PRE) & (self.crowd.premovement < t_end)
        lead[starting] = t_end - np.maximum(self.crowd.premovement[starting], t)
        st[starting] = ROW

        m = np.flatnonzero(st == ROW)
        if m.size:
            spd = self.crowd.speed[m] * mp.row_factor
            self.remain[m] -= spd * lead[m]
            arr = m[self.remain[m] <= 0]
            if arr.size:
                t_arr = t + dt + self.remain[arr] / (self.crowd.speed[arr] * mp.row_factor)
                self._arrive(arr, self.node[arr], np.full(arr.size, -1), t_arr, handed)

        m = np.flatnonzero(st == EDGE)
        if m.size:
            e = self.edge[m]
            # density of the people still walking on the element (those waiting at its end
            # are a queue: they limit storage, but discharge at capacity)
            waiting = (self.prev_edge >= 0) & ((st == QUEUE) | (st == MICRO) | (st == STUCK))
            wait_end = np.bincount(self.prev_edge[waiting], minlength=net.n_edges)
            dens = np.maximum(self.occ[e] - wait_end[e], 1.0) / self.area[e]
            spd = self.crowd.speed[m] * net.factor(e, self.node[m]) * np.maximum(density_speed_ratio(mp, dens), 0.02)
            self.remain[m] -= spd * dt
            reached = self.remain[m] <= 0
            if reached.any():
                arr = m[reached]
                t_arr = t + dt + self.remain[arr] / spd[reached]
                self._arrive(arr, self.target[arr], e[reached], t_arr, handed)

        self.tokens = np.minimum(self.tokens + net.capacity * dt, net.capacity * dt + 1.0)
        q = np.flatnonzero(st == QUEUE)
        self.qcount = np.bincount(self.edge[q], minlength=net.n_edges).astype(float) if q.size else np.zeros(net.n_edges)
        if q.size:
            e = self.edge[q]
            order = np.lexsort((q, self.t_queue[q], e))
            q, e = q[order], e[order]
            first = np.r_[True, e[1:] != e[:-1]]
            start = np.maximum.accumulate(np.where(first, np.arange(e.size), 0))
            rank = np.arange(e.size) - start
            allowed = np.minimum(np.floor(self.tokens), np.maximum(net.storage - self.occ, 0.0))
            adm = rank < allowed[e]
            if adm.any():
                a, ea = q[adm], e[adm]
                pe = self.prev_edge[a]
                np.subtract.at(self.occ, pe[pe >= 0], 1.0)
                self.prev_edge[a] = -1
                np.add.at(self.occ, ea, 1.0)
                np.subtract.at(self.tokens, ea, 1.0)
                st[a] = EDGE
                self.target[a] = net.other_end(ea, self.node[a])
                # walk the rest of this step on the new element (no time lost at junctions)
                head = t_end - np.maximum(self.t_queue[a], t)
                dens = self.occ[ea] / self.area[ea]
                spd = (self.crowd.speed[a] * net.factor(ea, self.node[a])
                       * np.maximum(density_speed_ratio(mp, dens), 0.02))
                self.remain[a] = net.length[ea] - spd * head
                self.qcount -= np.bincount(ea, minlength=net.n_edges)
        return np.concatenate(handed) if handed else np.empty(0, dtype=int)

    # ------------------------------------------------------------ reporting
    def max_density(self) -> float:
        if not self.big.any():
            return 0.0
        return float(np.max(self.occ[self.big] / self.area[self.big]))

    def counts(self) -> dict:
        st = self.status
        return {"evacuated": int(np.sum(st == DONE)), "waiting_to_move": int(np.sum(st == PRE)),
                "in_rows": int(np.sum(st == ROW)), "queued": int(np.sum(st == QUEUE)),
                "walking": int(np.sum(st == EDGE)), "handed_over": int(np.sum(st == MICRO)),
                "stuck": int(np.sum(st == STUCK))}

    def active(self) -> bool:
        return bool(np.any((self.status != DONE) & (self.status != STUCK)))


def run(net: EgressNet, crowd: Crowd, params: ModelParams, seed: int = 0, routing: str | None = None,
        tracer=None, **_) -> RunOutput:
    rng = np.random.default_rng(seed)
    dt, rec = params.network.dt, params.micro.record_dt
    with Timer() as tm:
        eng = NetworkEngine(net, crowd, params, rng, routing=routing)
        rows, t, next_rec = [], 0.0, 0.0
        peak, peak_t, last_exit_t, last_done = 0.0, 0.0, 0.0, 0
        while t < params.network.t_max and eng.active():
            eng.step(t, dt)
            t += dt
            if tracer is not None:
                tracer.snap(t, eng)
            dens = eng.max_density()
            if dens > peak:
                peak, peak_t = dens, t
            if t >= next_rec:
                rows.append({"t": t, **eng.counts(), "max_density": dens})
                next_rec += rec
            done = int(np.sum(eng.status == DONE))
            if done > last_done:
                last_done, last_exit_t = done, t
            elif t - last_exit_t > params.micro.stall_time and t > crowd.premovement.max():
                break
        rows.append({"t": t, **eng.counts(), "max_density": eng.max_density()})
    res = EvacuationResult(exit_time=eng.exit_time, exit_id=eng.exit_id, exit_width=net.exit_width,
                           model=NAME, seed=seed, compute_time_s=tm.elapsed)
    return RunOutput(result=res, crowd=crowd, timeseries=pd.DataFrame(rows), peak_density=peak,
                     peak_density_time=peak_t,
                     density_note=f"highest mean density on any element of >= {MIN_DENSITY_AREA:g} m^2",
                     info={"routing": "dynamic" if eng.dynamic else "static", "stuck": int(np.sum(eng.status == STUCK)),
                           "end_time_s": t})
