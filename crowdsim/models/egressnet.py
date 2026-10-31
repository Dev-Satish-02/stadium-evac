"""Array form of the navigation graph used by the network-based models.

Built once per (stadium, scenario): edge capacities and storage from the
SFPE rules, static distances to the nearest open exit, the static
shortest-path tree, and for each node the candidate next edges that move
strictly closer to an exit (so no route can loop).
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from crowdsim.models.movement import capacity, effective_width, element_type, free_speed_factor
from crowdsim.models.params import MovementParams
from crowdsim.stadium.model import SINK, Stadium


@dataclass(frozen=True, eq=False)
class EgressNet:
    nodes: tuple[str, ...]
    node_level: np.ndarray        # (n,) level id of each node
    node_xyz: np.ndarray          # (n, 3)
    edge_u: np.ndarray            # (E,) node index
    edge_v: np.ndarray
    kind: np.ndarray              # (E,) edge kind (str)
    etype: np.ndarray             # (E,) SFPE element type (str)
    length: np.ndarray            # [m]
    width: np.ndarray             # clear width [m]
    w_eff: np.ndarray             # effective width [m]
    capacity: np.ndarray          # max flow [p/s]
    storage: np.ndarray           # max persons on the element
    f_uv: np.ndarray              # free-speed factor when walking u -> v
    f_vu: np.ndarray              # ... and v -> u
    dist: np.ndarray              # (n,) shortest walk to an open exit [m]
    next_edge: np.ndarray         # (n,) edge on the static shortest path (-1 at exits / cut off)
    next_node: np.ndarray
    cand_edge: np.ndarray         # (n, maxdeg) edges leading strictly closer to an exit, -1 padded
    cand_node: np.ndarray
    exit_of_node: np.ndarray      # (n,) index into the exits list, -1 if not an open exit node
    exit_width: np.ndarray        # (n_exits,) clear width of every exit (open or not)
    exit_open: np.ndarray         # (n_exits,) bool

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return int(self.edge_u.size)

    def factor(self, edge: np.ndarray, from_node: np.ndarray) -> np.ndarray:
        """Free-speed factor for walking along `edge` starting at `from_node`."""
        return np.where(self.edge_u[edge] == from_node, self.f_uv[edge], self.f_vu[edge])

    def other_end(self, edge: np.ndarray, from_node: np.ndarray) -> np.ndarray:
        return np.where(self.edge_u[edge] == from_node, self.edge_v[edge], self.edge_u[edge])

    @classmethod
    def from_stadium(cls, stadium: Stadium, mp: MovementParams) -> "EgressNet":
        exit_nodes = {e.node: i for i, e in enumerate(stadium.exits) if e.enabled}
        widths = np.array([e.width for e in stadium.exits], dtype=float)
        open_ = np.array([e.enabled for e in stadium.exits], dtype=bool)
        return cls.from_graph(stadium.graph, stadium.nodes, exit_nodes, widths, open_, mp)

    @classmethod
    def from_graph(cls, g: nx.Graph, nodes: tuple[str, ...], exit_nodes: dict, exit_width, exit_open,
                   mp: MovementParams) -> "EgressNet":
        index = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)
        xyz = np.array([[g.nodes[v].get("x", np.nan), g.nodes[v].get("y", np.nan), g.nodes[v].get("z", 0.0)]
                        for v in nodes], dtype=float)
        level = np.array([g.nodes[v].get("level", "") for v in nodes], dtype=object)
        eu, ev, kinds, etypes, lengths, widths, weffs, caps, stor, fuv, fvu = ([] for _ in range(11))
        edge_id = {}
        for a, b, d in g.edges(data=True):
            if d["kind"] == "sink" or a == SINK or b == SINK:
                continue
            ia, ib = index[a], index[b]
            dz = xyz[ib, 2] - xyz[ia, 2]
            et = element_type(d["kind"], dz, d["length"])
            we = effective_width(mp, d["kind"], d["width"])
            edge_id[(min(ia, ib), max(ia, ib))] = len(eu)
            eu.append(ia); ev.append(ib); kinds.append(d["kind"]); etypes.append(et)
            lengths.append(d["length"]); widths.append(d["width"]); weffs.append(we)
            caps.append(capacity(mp, d["kind"], et, d["width"]))
            stor.append(max(1.0, np.floor(mp.jam_density * d["length"] * we)))
            fuv.append(free_speed_factor(mp, et, ascending=dz > 0.05))
            fvu.append(free_speed_factor(mp, et, ascending=dz < -0.05))

        dist = np.full(n, np.inf)
        next_node = np.full(n, -1, dtype=int)
        next_edge = np.full(n, -1, dtype=int)
        if SINK in g:
            pred, dmap = nx.dijkstra_predecessor_and_distance(g, SINK, weight="length")
            for v, dv in dmap.items():
                if v == SINK:
                    continue
                i = index[v]
                dist[i] = dv
                p = pred.get(v, [])
                if p and p[0] != SINK:
                    j = index[p[0]]
                    next_node[i] = j
                    next_edge[i] = edge_id[(min(i, j), max(i, j))]
        exit_of_node = np.full(n, -1, dtype=int)
        for name, ei in exit_nodes.items():
            exit_of_node[index[name]] = ei
            next_node[index[name]] = -1
            next_edge[index[name]] = -1

        eu_a, ev_a = np.array(eu, dtype=int), np.array(ev, dtype=int)
        cands: list[list[tuple[int, int]]] = [[] for _ in range(n)]
        for e, (a, b) in enumerate(zip(eu_a, ev_a)):
            if dist[b] < dist[a] - 1e-9:
                cands[a].append((e, b))
            elif dist[a] < dist[b] - 1e-9:
                cands[b].append((e, a))
        for i in range(n):
            if exit_of_node[i] >= 0:
                cands[i] = []
        maxdeg = max(1, max(len(c) for c in cands))
        cand_edge = np.full((n, maxdeg), -1, dtype=int)
        cand_node = np.full((n, maxdeg), -1, dtype=int)
        for i, c in enumerate(cands):
            c.sort(key=lambda en: dist[en[1]] + lengths[en[0]])  # best static option first
            for j, (e, b) in enumerate(c):
                cand_edge[i, j], cand_node[i, j] = e, b

        return cls(nodes=tuple(nodes), node_level=level, node_xyz=xyz, edge_u=eu_a, edge_v=ev_a,
                   kind=np.array(kinds, dtype=object), etype=np.array(etypes, dtype=object),
                   length=np.array(lengths, float), width=np.array(widths, float), w_eff=np.array(weffs, float),
                   capacity=np.array(caps, float), storage=np.array(stor, float),
                   f_uv=np.array(fuv, float), f_vu=np.array(fvu, float), dist=dist, next_edge=next_edge,
                   next_node=next_node, cand_edge=cand_edge, cand_node=cand_node, exit_of_node=exit_of_node,
                   exit_width=np.asarray(exit_width, float), exit_open=np.asarray(exit_open, bool))
