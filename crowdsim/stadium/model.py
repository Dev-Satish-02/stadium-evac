"""Built stadium data model shared by every evacuation model.

Height is handled as a multi-level network of 2-D plans:
each seating tier and each concourse is a level with its own floor height(s);
aisles (with stepped rows), vomitories, stairs and ramps join the levels and
carry a 3-D length (vertical drop included), a clear width and a kind, so
that Phase 2 models can apply kind-specific speeds and flow capacities.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import networkx as nx
import numpy as np

from crowdsim.geometry import RingCurve
from crowdsim.stadium.spec import StadiumSpec

SINK = "OUT"  # every enabled exit is joined to this virtual node

# edge kinds used in the navigation graph
EDGE_KINDS = ("aisle", "cross_aisle", "vomitory", "concourse", "stair", "ramp",
              "exit", "pitch_access", "sink")


@dataclass(frozen=True)
class Level:
    id: str
    kind: str          # "seating" or "concourse"
    z_min: float
    z_max: float
    d_inner: float     # radial band of the level in plan [m from field boundary]
    d_outer: float


@dataclass(frozen=True, eq=False)
class TierGeometry:
    """Derived geometry of one seating tier."""
    id: str
    index: int
    s_aisles: np.ndarray   # (K,) aisle centre-lines, bowl coordinate s
    d_rows: np.ndarray     # (rows,) row centre offsets [m]
    z_rows: np.ndarray     # (rows,) row floor heights [m]
    d_cross: float         # cross-aisle centre offset [m]
    z_cross: float
    d_start: float
    d_end: float
    aisle_width: float
    cross_width: float
    vomitory_aisles: tuple[int, ...]
    pitch_gate_aisles: tuple[int, ...]


@dataclass(frozen=True, eq=False)
class Seats:
    """Structure-of-arrays for every seat in the stadium."""
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    s: np.ndarray
    d: np.ndarray
    tier: np.ndarray      # tier index
    row: np.ndarray
    aisle: np.ndarray     # aisle the seat leaves by
    lateral: np.ndarray   # walking distance along the row to that aisle's centre-line [m]
    node: np.ndarray      # index into Stadium.nodes of the aisle node at this row

    @property
    def n(self) -> int:
        return int(self.x.size)


@dataclass(frozen=True)
class Exit:
    index: int            # gate index (G0, G1, ...) or pitch-gate index (P0, ...)
    id: str
    kind: str             # "gate" or "pitch"
    level: str
    s: float
    x: float
    y: float
    z: float
    width: float
    node: str             # graph node joined to SINK
    enabled: bool = True


@dataclass(frozen=True)
class Connector:
    """A vertical link between levels (vomitory, stair or ramp) for plotting and reporting."""
    id: str
    kind: str
    from_level: str
    to_level: str
    s: float
    width: float
    length: float
    dz: float


@dataclass(frozen=True, eq=False)
class Stadium:
    spec: StadiumSpec
    curve: RingCurve
    levels: tuple[Level, ...]
    tiers: tuple[TierGeometry, ...]
    seats: Seats
    exits: tuple[Exit, ...]
    connectors: tuple[Connector, ...]
    graph: nx.Graph
    nodes: tuple[str, ...]
    node_index: dict = field(repr=False)
    scenario: str | None = None

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def capacity(self) -> int:
        return self.seats.n

    def gates(self) -> list[Exit]:
        return [e for e in self.exits if e.kind == "gate"]

    def enabled_exits(self) -> list[Exit]:
        return [e for e in self.exits if e.enabled]

    def level(self, lid: str) -> Level:
        for lv in self.levels:
            if lv.id == lid:
                return lv
        raise KeyError(lid)

    def with_changes(self, **changes) -> "Stadium":
        return replace(self, **changes)

    def node_xyz(self) -> np.ndarray:
        """(n_nodes, 3) positions; the SINK row is NaN."""
        g = self.graph
        return np.array([[g.nodes[n]["x"], g.nodes[n]["y"], g.nodes[n]["z"]] for n in self.nodes])
