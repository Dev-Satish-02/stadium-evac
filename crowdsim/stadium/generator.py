"""Parametric stadium generator: StadiumSpec -> Stadium (geometry, seats, navigation graph).

Layout rules (shared by every preset)
-------------------------------------
* Seating tiers are concentric bands around the field boundary.
* Each tier is split into blocks by ``aisles`` straight radial aisles.
* Rows are stepped: row r sits at offset d_rows[r] and height z_rows[r].
* A lateral cross-aisle crosses the tier in front of row ``cross_aisle.row``
  (row 0 = front at the cross-aisle, row = rows puts it at the back).
  Spectators walk along their row to the nearer aisle, along the aisle
  (up or down the steps) to the cross-aisle, along the cross-aisle to a
  vomitory, and through the vomitory to a concourse.
* Concourses are rings joined to lower concourses by stairs or ramps;
  exits (gates) sit on the outer edge of a concourse.
* Optional pitch gates at the front of a tier lead onto the field.
"""
from __future__ import annotations

import math
from dataclasses import replace

import networkx as nx
import numpy as np

from crowdsim.geometry import RingCurve
from crowdsim.stadium.model import SINK, Connector, Exit, Level, Seats, Stadium, TierGeometry
from crowdsim.stadium.network import add_edge, add_node, dist3, sync_exit_edges
from crowdsim.stadium.spec import StadiumSpec, TierSpec

CURVE_SAMPLES = 4096


# ---------------------------------------------------------------- helpers
def evenly_spaced(count: int, phase: float) -> np.ndarray:
    """count positions in s, evenly spaced, shifted by phase (fraction of one spacing)."""
    return np.sort(((np.arange(count) + phase) / count) % 1.0)


def _skey(s: float) -> float:
    return round(float(s) % 1.0, 9) % 1.0


def concourse_node(cid: str, s: float) -> str:
    return f"{cid}/C/{_skey(s):.9f}"


def aisle_node(tid: str, k: int, r: int) -> str:
    return f"{tid}/A/{k}/{r}"


def cross_node(tid: str, k: int) -> str:
    return f"{tid}/X/{k}"


# ---------------------------------------------------------------- tiers
def tier_geometry(index: int, t: TierSpec) -> TierGeometry:
    r = np.arange(t.rows)
    cr, cw = t.cross_aisle.row, t.cross_aisle.width
    d_rows = t.d_start + (r + 0.5) * t.row_depth + np.where(r >= cr, cw, 0.0)
    z_rows = t.z_base + r * t.riser
    s_aisles = evenly_spaced(t.aisles, t.aisle_phase)
    vom = tuple(k for k in range(t.aisles) if (k - t.vomitories.offset) % t.vomitories.every == 0)
    pitch = ()
    if t.pitch_gates is not None:
        pg = t.pitch_gates
        pitch = tuple(k for k in range(t.aisles) if (k - pg.offset) % pg.every == 0)
    return TierGeometry(
        id=t.id, index=index, s_aisles=s_aisles, d_rows=d_rows, z_rows=z_rows,
        d_cross=t.d_start + cr * t.row_depth + cw / 2.0, z_cross=t.z_base + cr * t.riser,
        d_start=t.d_start, d_end=t.d_start + t.rows * t.row_depth + cw,
        aisle_width=t.aisle_width, cross_width=cw, vomitory_aisles=vom, pitch_gate_aisles=pitch,
    )


def _row_seats(curve: RingCurve, tg: TierGeometry, d: float, seat_width: float):
    """Seat arc positions in one row, plus the aisle each seat uses and its lateral walk."""
    K = len(tg.s_aisles)
    w = tg.aisle_width
    a = curve.arc(tg.s_aisles, d)
    total = curve.ring_length(d)
    starts = a + w / 2.0
    ends = np.roll(a, -1) - w / 2.0
    ends[-1] += total
    avail = ends - starts
    nseat = np.floor(avail / seat_width + 1e-9).astype(int).clip(min=0)
    arcs, aisles, lateral = [], [], []
    for k in range(K):
        n = nseat[k]
        if n == 0:
            continue
        pos = starts[k] + (avail[k] - n * seat_width) / 2.0 + seat_width * (np.arange(n) + 0.5)
        left, right = pos - starts[k], ends[k] - pos
        use_left = left <= right
        arcs.append(pos)
        aisles.append(np.where(use_left, k, (k + 1) % K))
        lateral.append(np.minimum(left, right) + w / 2.0)
    if not arcs:
        return np.empty(0), np.empty(0, int), np.empty(0)
    return np.concatenate(arcs), np.concatenate(aisles), np.concatenate(lateral)


def build_seats(curve: RingCurve, tiers: list[TierGeometry], seat_width: float, node_index: dict) -> Seats:
    cols = {k: [] for k in ("x", "y", "z", "s", "d", "tier", "row", "aisle", "lateral", "node")}
    for tg in tiers:
        for r, (d, z) in enumerate(zip(tg.d_rows, tg.z_rows)):
            arc, aisle, lat = _row_seats(curve, tg, float(d), seat_width)
            if arc.size == 0:
                continue
            s = curve.s_at_arc(arc, float(d))
            xy = curve.xy(s, float(d))
            n = arc.size
            cols["x"].append(xy[:, 0]); cols["y"].append(xy[:, 1])
            cols["z"].append(np.full(n, z)); cols["s"].append(s); cols["d"].append(np.full(n, d))
            cols["tier"].append(np.full(n, tg.index)); cols["row"].append(np.full(n, r))
            cols["aisle"].append(aisle); cols["lateral"].append(lat)
            cols["node"].append(np.array([node_index[aisle_node(tg.id, int(k), r)] for k in aisle]))
    out = {}
    for key, parts in cols.items():
        dtype = int if key in ("tier", "row", "aisle", "node") else float
        out[key] = np.concatenate(parts).astype(dtype) if parts else np.empty(0, dtype)
    return Seats(**out)


# ---------------------------------------------------------------- builder
def build_stadium(spec: StadiumSpec) -> Stadium:
    curve = RingCurve.superellipse(spec.field.a, spec.field.b, spec.field.n, samples=CURVE_SAMPLES)
    g = nx.Graph()
    add_node(g, SINK, (np.nan, np.nan, np.nan), level="", kind="sink")
    tiers = [tier_geometry(i, t) for i, t in enumerate(spec.tiers)]
    conc = {c.id: c for c in spec.concourses}

    # 1. collect every attachment point on each concourse ring
    attach: dict[str, set[float]] = {c.id: set() for c in spec.concourses}
    for tg, ts in zip(tiers, spec.tiers):
        for k in tg.vomitory_aisles:
            attach[ts.vomitories.concourse].add(_skey(tg.s_aisles[k]))
    for c in spec.concourses:
        for cn in c.connectors:
            for s in evenly_spaced(cn.count, cn.phase):
                attach[c.id].add(_skey(s))
                attach[cn.to].add(_skey(s))
        if c.exits:
            for s in evenly_spaced(c.exits.count, c.exits.phase):
                attach[c.id].add(_skey(s))

    # 2. concourse rings
    for c in spec.concourses:
        d_mid = 0.5 * (c.d_inner + c.d_outer)
        keys = sorted(attach[c.id])
        for s in keys:
            add_node(g, concourse_node(c.id, s), (*curve.xy(s, d_mid), c.z), level=c.id, kind="concourse")
        if len(keys) >= 2:
            for i, s in enumerate(keys):
                s_next = keys[(i + 1) % len(keys)]
                add_edge(g, concourse_node(c.id, s), concourse_node(c.id, s_next),
                         length=curve.arc_between(s, s_next, d_mid), width=c.d_outer - c.d_inner,
                         kind="concourse")

    # 3. tiers: aisle nodes, aisles, cross-aisles, vomitories, pitch gates
    connectors: list[Connector] = []
    exits: list[Exit] = []
    for tg, ts in zip(tiers, spec.tiers):
        K, cr = ts.aisles, ts.cross_aisle.row
        for k in range(K):
            sk = tg.s_aisles[k]
            add_node(g, cross_node(ts.id, k), (*curve.xy(sk, tg.d_cross), tg.z_cross), level=ts.id, kind="cross")
            for r in range(ts.rows):
                add_node(g, aisle_node(ts.id, k, r), (*curve.xy(sk, tg.d_rows[r]), tg.z_rows[r]),
                         level=ts.id, kind="aisle")
            for r in range(ts.rows):
                if r < cr:
                    nxt = cross_node(ts.id, k) if r == cr - 1 else aisle_node(ts.id, k, r + 1)
                else:
                    nxt = cross_node(ts.id, k) if r == cr else aisle_node(ts.id, k, r - 1)
                a = aisle_node(ts.id, k, r)
                add_edge(g, a, nxt, length=dist3(g, a, nxt), width=ts.aisle_width, kind="aisle")
        for k in range(K):
            k2 = (k + 1) % K
            add_edge(g, cross_node(ts.id, k), cross_node(ts.id, k2),
                     length=curve.arc_between(tg.s_aisles[k], tg.s_aisles[k2], tg.d_cross),
                     width=tg.cross_width, kind="cross_aisle")
        v = ts.vomitories
        for k in tg.vomitory_aisles:
            a, b = cross_node(ts.id, k), concourse_node(v.concourse, tg.s_aisles[k])
            length = max(dist3(g, a, b), v.min_length)
            add_edge(g, a, b, length=length, width=v.width, kind="vomitory")
            connectors.append(Connector(id=f"{ts.id}/V{k}", kind="vomitory", from_level=ts.id,
                                        to_level=v.concourse, s=float(tg.s_aisles[k]), width=v.width,
                                        length=length, dz=conc[v.concourse].z - tg.z_cross))
        if ts.pitch_gates is not None:
            for k in tg.pitch_gate_aisles:
                pg = f"{ts.id}/PG/{k}"
                xy = curve.xy(tg.s_aisles[k], tg.d_start)
                add_node(g, pg, (*xy, ts.z_base), level=ts.id, kind="pitch_gate")
                front = aisle_node(ts.id, k, 0)
                add_edge(g, front, pg, length=dist3(g, front, pg), width=ts.pitch_gates.width, kind="pitch_access")
                exits.append(Exit(index=len([e for e in exits if e.kind == "pitch"]), id="", kind="pitch",
                                  level=ts.id, s=float(tg.s_aisles[k]), x=float(xy[0]), y=float(xy[1]),
                                  z=ts.z_base, width=ts.pitch_gates.width, node=pg, enabled=False))

    # 4. stairs / ramps between concourses, and gates
    gate_no = 0
    for c in spec.concourses:
        for ci, cn in enumerate(c.connectors):
            dz = c.z - conc[cn.to].z
            length = dz / math.sin(math.radians(cn.angle_deg))
            for j, s in enumerate(evenly_spaced(cn.count, cn.phase)):
                add_edge(g, concourse_node(c.id, s), concourse_node(cn.to, s),
                         length=length, width=cn.width, kind=cn.kind)
                connectors.append(Connector(id=f"{c.id}/{cn.kind}{ci}.{j}", kind=cn.kind, from_level=c.id,
                                            to_level=cn.to, s=float(s), width=cn.width, length=length, dz=-dz))
        if c.exits:
            for s in evenly_spaced(c.exits.count, c.exits.phase):
                en = f"{c.id}/E/{gate_no}"
                xy = curve.xy(s, c.d_outer)
                add_node(g, en, (*xy, c.z), level=c.id, kind="exit")
                cnode = concourse_node(c.id, s)
                add_edge(g, cnode, en, length=dist3(g, cnode, en), width=c.exits.width, kind="exit")
                exits.append(Exit(index=gate_no, id=f"G{gate_no}", kind="gate", level=c.id, s=float(s),
                                  x=float(xy[0]), y=float(xy[1]), z=c.z, width=c.exits.width, node=en))
                gate_no += 1
    exits = [e if e.kind == "gate" else _named_pitch(e) for e in exits]
    exits.sort(key=lambda e: (e.kind != "gate", e.index))
    sync_exit_edges(g, exits)

    nodes = tuple(g.nodes)
    node_index = {n: i for i, n in enumerate(nodes)}
    seats = build_seats(curve, tiers, spec.seat_width, node_index)
    levels = tuple(
        [Level(id=t.id, kind="seating", z_min=float(min(tg.z_rows.min(), tg.z_cross)),
               z_max=float(max(tg.z_rows.max(), tg.z_cross)), d_inner=tg.d_start, d_outer=tg.d_end)
         for t, tg in zip(spec.tiers, tiers)]
        + [Level(id=c.id, kind="concourse", z_min=c.z, z_max=c.z, d_inner=c.d_inner, d_outer=c.d_outer)
           for c in spec.concourses])
    return Stadium(spec=spec, curve=curve, levels=levels, tiers=tuple(tiers), seats=seats,
                   exits=tuple(exits), connectors=tuple(connectors), graph=g, nodes=nodes,
                   node_index=node_index)


def _named_pitch(e: Exit) -> Exit:
    return replace(e, id=f"P{e.index}")
