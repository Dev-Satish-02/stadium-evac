"""Navigation-graph helpers shared by the generator, scenarios and later models."""
from __future__ import annotations

import math
from typing import Iterable

import networkx as nx
import numpy as np

from crowdsim.stadium.model import SINK, Exit


def add_node(g: nx.Graph, name: str, xyz, level: str, kind: str) -> None:
    x, y, z = (float(v) for v in xyz)
    g.add_node(name, x=x, y=y, z=z, level=level, kind=kind)


def dist3(g: nx.Graph, a: str, b: str) -> float:
    na, nb = g.nodes[a], g.nodes[b]
    return math.dist((na["x"], na["y"], na["z"]), (nb["x"], nb["y"], nb["z"]))


def add_edge(g: nx.Graph, a: str, b: str, *, length: float, width: float, kind: str) -> None:
    """Add an undirected edge; if one already exists keep the shorter one."""
    if a == b:
        return
    if not (length >= 0 and math.isfinite(length)):
        raise ValueError(f"invalid edge length {length} for {a} - {b}")
    if g.has_edge(a, b) and g.edges[a, b]["length"] <= length:
        return
    dz = g.nodes[b]["z"] - g.nodes[a]["z"] if kind != "sink" else 0.0
    g.add_edge(a, b, length=float(length), width=float(width), kind=kind, dz=float(dz))


def sync_exit_edges(g: nx.Graph, exits: Iterable[Exit]) -> None:
    """Make the graph match each exit's enabled flag and width.

    The clear width is stored on the walkway that passes through the gate
    (kind 'exit' or 'pitch_access', the element models use for capacity) and
    on the exit -> SINK edge."""
    for e in exits:
        for nb in g.neighbors(e.node):
            d = g.edges[e.node, nb]
            if d["kind"] in ("exit", "pitch_access"):
                d["width"] = float(e.width)
        if e.enabled:
            if g.has_edge(e.node, SINK):
                g.edges[e.node, SINK]["width"] = float(e.width)
            else:
                g.add_edge(e.node, SINK, length=0.0, width=float(e.width), kind="sink", dz=0.0)
        elif g.has_edge(e.node, SINK):
            g.remove_edge(e.node, SINK)


def distances_to_sink(g: nx.Graph, nodes: tuple[str, ...]) -> np.ndarray:
    """Shortest 3-D walking distance from every node to the nearest enabled exit (inf if cut off)."""
    if SINK not in g:
        return np.full(len(nodes), np.inf)
    dist = nx.single_source_dijkstra_path_length(g, SINK, weight="length")
    return np.array([dist.get(n, np.inf) for n in nodes], dtype=float)
