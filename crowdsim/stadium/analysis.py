"""Static analysis of a built stadium: egress distances, summary table, integrity checks."""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from crowdsim.stadium.model import SINK, Stadium
from crowdsim.stadium.network import distances_to_sink


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str = ""


def seat_egress_distance(stadium: Stadium) -> np.ndarray:
    """Shortest 3-D walking distance from each seat to the nearest enabled exit [m]."""
    node_d = distances_to_sink(stadium.graph, stadium.nodes)
    return stadium.seats.lateral + node_d[stadium.seats.node]


def analytic_capacity_estimate(stadium: Stadium) -> tuple[float, int]:
    """Independent estimate: sum over rows of (row length - aisle widths) / seat width.

    Returns (estimate, allowed rounding error) - each block loses < 1 seat to rounding.
    """
    est, tol = 0.0, 0
    for tg, ts in zip(stadium.tiers, stadium.spec.tiers):
        for d in tg.d_rows:
            usable = stadium.curve.ring_length(float(d)) - ts.aisles * ts.aisle_width
            est += usable / stadium.spec.seat_width
            tol += ts.aisles
    return est, tol


def summary(stadium: Stadium) -> dict:
    spec, seats = stadium.spec, stadium.seats
    egress = seat_egress_distance(stadium)
    finite = egress[np.isfinite(egress)]
    gates = stadium.gates()
    enabled = stadium.enabled_exits()
    total_w = sum(e.width for e in enabled)
    per_tier = {t.id: int(np.sum(seats.tier == i)) for i, t in enumerate(spec.tiers)}
    vom = [c for c in stadium.connectors if c.kind == "vomitory"]
    return {
        "preset": spec.key,
        "name": spec.name,
        "sport": spec.sport,
        "scenario": stadium.scenario or "base",
        "capacity": stadium.capacity,
        "seats_per_tier": per_tier,
        "levels": len(stadium.levels),
        "aisles": int(sum(t.aisles for t in spec.tiers)),
        "vomitories": len(vom),
        "vomitory_width_m": round(sum(c.width for c in vom), 2),
        "stairs": sum(1 for c in stadium.connectors if c.kind == "stair"),
        "ramps": sum(1 for c in stadium.connectors if c.kind == "ramp"),
        "gates": len(gates),
        "pitch_gates": sum(1 for e in stadium.exits if e.kind == "pitch"),
        "enabled_exits": len(enabled),
        "exit_width_m": round(total_w, 2),
        "exit_width_per_1000_m": round(1000.0 * total_w / max(stadium.capacity, 1), 3),
        "max_seat_height_m": round(float(seats.z.max()), 2),
        "egress_mean_m": round(float(finite.mean()), 1) if finite.size else float("nan"),
        "egress_p95_m": round(float(np.percentile(finite, 95)), 1) if finite.size else float("nan"),
        "egress_max_m": round(float(finite.max()), 1) if finite.size else float("nan"),
        "unreachable_seats": int(np.sum(~np.isfinite(egress))),
        "graph_nodes": stadium.graph.number_of_nodes(),
        "graph_edges": stadium.graph.number_of_edges(),
    }


def validate(stadium: Stadium) -> list[Check]:
    """Integrity checks a built stadium must pass before any model uses it."""
    spec, seats, g = stadium.spec, stadium.seats, stadium.graph
    out: list[Check] = []

    out.append(Check("capacity > 0", stadium.capacity > 0, f"{stadium.capacity} seats"))

    est, tol = analytic_capacity_estimate(stadium)
    diff = abs(stadium.capacity - est)
    out.append(Check("capacity matches analytic estimate", diff <= tol,
                     f"built {stadium.capacity}, estimate {est:.0f}, |diff| {diff:.0f} <= {tol}"))

    ok_band = True
    for i, tg in enumerate(stadium.tiers):
        m = seats.tier == i
        ok_band &= bool(np.all((seats.d[m] > tg.d_start) & (seats.d[m] < tg.d_end)))
    out.append(Check("every seat lies inside its tier band", ok_band))

    spacing_ok = True
    min_gap = np.inf
    for i, tg in enumerate(stadium.tiers):
        for r in range(len(tg.d_rows)):
            m = (seats.tier == i) & (seats.row == r)
            if m.sum() < 2:
                continue
            arc = np.sort(stadium.curve.arc(seats.s[m], float(tg.d_rows[r])))
            gaps = np.diff(arc)
            min_gap = min(min_gap, float(gaps.min()))
            spacing_ok &= bool(gaps.min() >= spec.seat_width - 1e-6)
    out.append(Check("no overlapping seats (spacing >= seat width)", spacing_ok,
                     f"min centre spacing {min_gap:.3f} m"))

    lateral_ok = bool(np.all(seats.lateral > 0) and np.all(np.isfinite(seats.lateral)))
    out.append(Check("row walking distances are positive and finite", lateral_ok))

    comps = list(nx.connected_components(g))
    main = next(c for c in comps if SINK in c)
    isolated = g.number_of_nodes() - len(main)
    out.append(Check("navigation graph is one connected piece", isolated == 0,
                     f"{isolated} node(s) cut off"))

    egress = seat_egress_distance(stadium)
    n_bad = int(np.sum(~np.isfinite(egress)))
    out.append(Check("every seat can reach an exit", n_bad == 0, f"{n_bad} unreachable seat(s)"))

    gates = stadium.gates()
    want = sum(c.exits.count for c in spec.concourses if c.exits)
    out.append(Check("gate count matches spec", len(gates) == want, f"{len(gates)} gates"))

    lengths = np.array([d["length"] for _, _, d in g.edges(data=True) if d["kind"] != "sink"])
    out.append(Check("all walkway lengths positive", bool(np.all(lengths > 0)),
                     f"min {lengths.min():.3f} m"))

    # height is respected: walking distance can never be shorter than the straight line to an exit
    exit_xyz = np.array([[e.x, e.y, e.z] for e in stadium.enabled_exits()])
    seat_xyz = np.column_stack([seats.x, seats.y, seats.z])
    sample = seat_xyz[:: max(1, len(seat_xyz) // 2000)]
    straight = np.min(np.linalg.norm(sample[:, None, :] - exit_xyz[None, :, :], axis=2), axis=1)
    walk = egress[:: max(1, len(seat_xyz) // 2000)]
    out.append(Check("walking distance >= straight-line distance to nearest exit",
                     bool(np.all(walk + 1e-6 >= straight)), f"checked {len(sample)} seats"))
    return out
