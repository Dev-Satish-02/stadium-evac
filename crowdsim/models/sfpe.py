"""Model 1 - Hydraulic (SFPE) controlling-component calculation.

The classic engineering hand calculation, applied to every person:

1. Everyone takes the static shortest route to the nearest open exit.
2. Each egress element e carries N_e people and passes at most
   C_e = Fs_max * W_eff people per second (SFPE capacity).
3. On each route the *controlling component* is the element with the
   largest N_e / C_e (longest time to pass everyone who uses it).
4. A person reaches their controlling component after pre-movement, the
   row walk and free walking; it then serves people first-come-first-served
   at C_e; after it they walk freely to the exit.

No crowd interaction except at the single controlling queue, no route
changes, no density effects on speed: it is a transparent baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crowdsim.metrics import EvacuationResult, Timer
from crowdsim.models.base import Crowd, RunOutput
from crowdsim.models.egressnet import EgressNet
from crowdsim.models.params import ModelParams

NAME = "sfpe"
LABEL = "Hydraulic (SFPE)"


def tree_quantities(net: EgressNet, start_counts: np.ndarray) -> dict:
    """Propagate people counts down the shortest-path tree and find controlling components."""
    n = net.n_nodes
    order = np.argsort(-np.where(np.isfinite(net.dist), net.dist, -1.0))  # far -> near
    node_count = start_counts.astype(float).copy()
    edge_count = np.zeros(net.n_edges)
    for i in order:
        e = net.next_edge[i]
        if e < 0 or not np.isfinite(net.dist[i]):
            continue
        edge_count[e] += node_count[i]
        node_count[net.next_node[i]] += node_count[i]

    with np.errstate(divide="ignore", invalid="ignore"):
        edge_time = np.where(net.capacity > 0, edge_count / net.capacity, np.inf)
    ctrl_edge = np.full(n, -1, dtype=int)
    ctrl_val = np.full(n, -np.inf)
    eff_total = np.full(n, np.inf)   # sum(length / factor) to the exit
    eff_before = np.full(n, np.inf)  # ... up to the start of the controlling edge
    exit_node = np.full(n, -1, dtype=int)
    for i in order[::-1]:  # near -> far
        if not np.isfinite(net.dist[i]):
            continue
        if net.exit_of_node[i] >= 0:
            eff_total[i] = 0.0
            eff_before[i] = 0.0
            exit_node[i] = i
            continue
        e, j = net.next_edge[i], net.next_node[i]
        if e < 0 or exit_node[j] < 0:
            continue
        f = net.f_uv[e] if net.edge_u[e] == i else net.f_vu[e]
        step = net.length[e] / f
        eff_total[i] = step + eff_total[j]
        exit_node[i] = exit_node[j]
        if edge_time[e] >= ctrl_val[j]:
            ctrl_edge[i], ctrl_val[i], eff_before[i] = e, edge_time[e], 0.0
        else:
            ctrl_edge[i], ctrl_val[i], eff_before[i] = ctrl_edge[j], ctrl_val[j], step + eff_before[j]
    return {"edge_count": edge_count, "edge_time": edge_time, "ctrl_edge": ctrl_edge,
            "eff_total": eff_total, "eff_before": eff_before, "exit_node": exit_node}


def fifo_departures(arrival: np.ndarray, cap: float) -> np.ndarray:
    """Departure times from a first-come-first-served server passing `cap` persons per second."""
    order = np.argsort(arrival, kind="stable")
    a = arrival[order]
    k = np.arange(a.size)
    dep_sorted = np.maximum.accumulate(a - k / cap) + k / cap
    out = np.empty_like(arrival)
    out[order] = dep_sorted
    return out


def run(net: EgressNet, crowd: Crowd, params: ModelParams, seed: int = 0, **_) -> RunOutput:
    with Timer() as tm:
        mp = params.movement
        start_counts = np.bincount(crowd.start_node, minlength=net.n_nodes)
        tq = tree_quantities(net, start_counts)
        node = crowd.start_node
        v = crowd.speed
        reach = tq["exit_node"][node] >= 0
        exit_time = np.full(crowd.n, np.inf)
        exit_id = np.full(crowd.n, -1, dtype=int)
        t_row = crowd.lateral / (v * mp.row_factor)
        arrive = crowd.premovement + t_row + tq["eff_before"][node] / v
        after = (tq["eff_total"][node] - tq["eff_before"][node]) / v
        ctrl = tq["ctrl_edge"][node]
        # people who start at an exit node (not in practice) leave after their walk
        direct = reach & (ctrl < 0)
        exit_time[direct] = arrive[direct]
        for e in np.unique(ctrl[reach & (ctrl >= 0)]):
            m = reach & (ctrl == e)
            exit_time[m] = fifo_departures(arrive[m], net.capacity[e]) + after[m]
        exit_id[reach] = net.exit_of_node[tq["exit_node"][node[reach]]]
    res = EvacuationResult(exit_time=exit_time, exit_id=exit_id, exit_width=net.exit_width, model=NAME,
                           seed=seed, compute_time_s=tm.elapsed)
    ctrl_used, counts = np.unique(ctrl[reach & (ctrl >= 0)], return_counts=True)
    worst = int(ctrl_used[np.argmax(tq["edge_time"][ctrl_used])]) if ctrl_used.size else -1
    info = {
        "controlling_elements": int(ctrl_used.size),
        "worst_element_kind": str(net.kind[worst]) if worst >= 0 else "",
        "worst_element_time_s": float(tq["edge_time"][worst]) if worst >= 0 else float("nan"),
        "controlling_kinds": {str(k): int(counts[net.kind[ctrl_used] == k].sum())
                              for k in np.unique(net.kind[ctrl_used])} if ctrl_used.size else {},
    }
    t_end = float(np.max(exit_time[np.isfinite(exit_time)])) if np.isfinite(exit_time).any() else 0.0
    grid = np.arange(0.0, t_end + 5.0, params.micro.record_dt)
    ts = pd.DataFrame({"t": grid, "evacuated": np.searchsorted(np.sort(exit_time), grid, side="right")})
    return RunOutput(result=res, crowd=crowd, timeseries=ts,
                     density_note="not modelled (hydraulic model has no spatial crowd state)", info=info)
