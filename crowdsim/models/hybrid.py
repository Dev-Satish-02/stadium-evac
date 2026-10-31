"""Step 9 - Hybrid linkage: network model upstream, microscopic model on the exit concourse.

A stadium evacuation spans seats, aisles, vomitories, stairs and concourses
on several levels. Running a microscopic model over all of that would cost
hours, so (as in multi-scale hybrid models in the literature) the stadium is
split:

* upstream (rows, aisles, cross-aisles, vomitories, upper concourses,
  stairs/ramps, pitch gates) - the queueing network (Model 4);
* the concourse(s) that contain the exit gates - the CA (Model 2) or the
  social force model (Model 3), on a 2-D domain rasterised from the stadium.

People reaching a node of the exit concourse are handed over and placed at
spawn slots across the arriving vomitory/stair. Until there is room they stay
on the element they came from, so congestion on the concourse backs up into
the stands (two-way coupling).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crowdsim.metrics import EvacuationResult, Timer, density_grid
from crowdsim.models.base import Crowd, RunOutput
from crowdsim.models.ca import CAEngine, new_queues
from crowdsim.models.domain import Domain, concourse_domain
from crowdsim.models.egressnet import EgressNet
from crowdsim.models.network import DONE, NetworkEngine
from crowdsim.models.params import ModelParams
from crowdsim.models.sf import SFEngine
from crowdsim.stadium.model import Stadium

GATE_QUEUE_RADIUS = 5.0  # people within this distance of an open gate count as queueing [m]


def micro_levels(stadium: Stadium) -> list[str]:
    return sorted({e.level for e in stadium.exits if e.kind == "gate" and e.enabled})


def build_domains(stadium: Stadium, net: EgressNet, params: ModelParams) -> tuple[dict, dict]:
    g = stadium.graph
    level_nodes = {}
    for lv in micro_levels(stadium):
        level_nodes[lv] = np.array([i for i, n in enumerate(net.nodes)
                                    if net.node_level[i] == lv and g.nodes[n]["kind"] == "concourse"], dtype=int)
    domains = {lv: concourse_domain(stadium, lv, params, inject_nodes=nodes) for lv, nodes in level_nodes.items()}
    return domains, level_nodes


def _micro_stats(dom: Domain, xy: np.ndarray, cell: float) -> tuple[float, int]:
    if len(xy) == 0:
        return 0.0, 0
    dens, _, _ = density_grid(xy, cell, bounds=dom.bounds)
    open_ = dom.exit_center[np.isin(np.arange(len(dom.exit_center)), np.unique(dom.exit_id[dom.exit_id >= 0]))]
    q = 0
    if len(open_):
        d = np.linalg.norm(xy[:, None, :] - open_[None, :, :], axis=2)
        q = int(np.sum(d.min(axis=1) <= GATE_QUEUE_RADIUS))
    return float(dens.max()), q


def run_hybrid(kind: str, stadium: Stadium, net: EgressNet, crowd: Crowd, params: ModelParams,
               seed: int = 0, domains: tuple | None = None, tracer=None,
               inject_log: list | None = None) -> RunOutput:
    """kind = 'ca' or 'sf'. ``inject_log`` (a list) receives (gids, time, xy) for every placement on the concourse."""
    rng = np.random.default_rng(seed)
    dt_net, rec = params.network.dt, params.micro.record_dt
    with Timer() as tm:
        doms, level_nodes = domains if domains is not None else build_domains(stadium, net, params)
        micro_nodes = np.concatenate(list(level_nodes.values())) if level_nodes else np.empty(0, int)
        node_level = {int(n): lv for lv, nodes in level_nodes.items() for n in nodes}
        eng = NetworkEngine(net, crowd, params, rng, micro_nodes=micro_nodes)
        if kind == "ca":
            vmax = max(float(crowd.speed.max()), 0.1)
            micro = {lv: CAEngine(d, params, rng, crowd.speed, crowd.herding, v_max=vmax) for lv, d in doms.items()}
        elif kind == "sf":
            micro = {lv: SFEngine(d, params, rng, crowd.speed, crowd.radius, crowd.herding) for lv, d in doms.items()}
        else:
            raise ValueError(f"unknown micro model '{kind}'")
        queues = {lv: new_queues(nodes) for lv, nodes in level_nodes.items()}
        rows = []
        t, next_rec = 0.0, 0.0
        peak, peak_t, peak_xy = 0.0, 0.0, None
        last_done, last_exit_t = 0, 0.0
        while t < params.network.t_max and eng.active():
            handed = eng.step(t, dt_net)
            t += dt_net
            if handed.size:
                for h in handed[np.argsort(eng.t_queue[handed], kind="stable")]:
                    nd = int(eng.node[h])
                    queues[node_level[nd]][nd].append(int(h))
            for lv, m in micro.items():
                placed = m.inject(queues[lv])
                if placed.size:
                    eng.release(placed)
                    if inject_log is not None:  # newly placed people are the last rows of positions()
                        inject_log.append((placed.copy(), float(m.t), m.positions()[-placed.size:].copy()))
                while m.t < t - 1e-9:
                    g, e, tt = m.step()
                    if g.size:
                        eng.finish(g, tt, e)
            if tracer is not None:
                tracer.snap(t, eng, micro)
            if t >= next_rec:
                next_rec += rec
                in_micro, waiting, dmax, gq = 0, 0, 0.0, 0
                for lv, m in micro.items():
                    xy = m.positions()
                    dm, q = _micro_stats(doms[lv], xy, params.micro.density_cell)
                    in_micro += m.n
                    waiting += sum(len(q_) for q_ in queues[lv].values())
                    gq += q
                    if dm > dmax:
                        dmax = dm
                    if dm > peak:
                        peak, peak_t, peak_xy = dm, t, xy.copy()
                c = eng.counts()
                c.pop("handed_over")
                rows.append({"t": t, **c, "on_concourse": in_micro, "waiting_to_enter": waiting,
                             "max_density": dmax, "gate_queue": gq})
            done = int(np.sum(eng.status == DONE))
            if done > last_done:
                last_done, last_exit_t = done, t
            elif t - last_exit_t > params.micro.stall_time and t > crowd.premovement.max():
                break
    res = EvacuationResult(exit_time=eng.exit_time, exit_id=eng.exit_id, exit_width=net.exit_width,
                           model=kind, seed=seed, compute_time_s=tm.elapsed)
    info = {"micro_levels": list(doms), "end_time_s": t,
            "micro_area_m2": round(sum(d.walk_area for d in doms.values()), 1),
            "escaped_numerically": int(sum(getattr(m, "escaped", 0) for m in micro.values())),
            "micro_dt_s": float(next(iter(micro.values())).dt) if micro else float("nan")}
    return RunOutput(result=res, crowd=crowd, timeseries=pd.DataFrame(rows), peak_density=peak,
                     peak_density_time=peak_t, peak_snapshot=peak_xy,
                     density_note=f"highest {params.micro.density_cell:g} m cell density on the exit concourse",
                     info=info)
