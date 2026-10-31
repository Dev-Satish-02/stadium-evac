"""Stadium preset specification: typed dataclasses parsed from YAML with validation.

See ``config/stadiums/football_bowl.yaml`` for an annotated example of every key.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from crowdsim.configio import ConfigError, Section, load_yaml, stadium_path


@dataclass(frozen=True)
class FieldSpec:
    shape: str   # only "superellipse" for now
    a: float     # half-length of the field boundary (front of seating) along x [m]
    b: float     # half-width along y [m]
    n: float     # 2 = ellipse/oval, 4 = rounded rectangle, 6+ = near rectangle


@dataclass(frozen=True)
class CrossAisleSpec:
    row: int      # cross-aisle sits in front of this row index (0 = front, rows = back)
    width: float  # [m]


@dataclass(frozen=True)
class VomitorySpec:
    every: int          # a vomitory on every n-th radial aisle
    offset: int         # ... starting at this aisle index
    width: float        # tunnel clear width [m]
    min_length: float   # minimum tunnel length [m]
    concourse: str      # id of the concourse the tunnel leads to


@dataclass(frozen=True)
class PitchGateSpec:
    every: int
    offset: int
    width: float


@dataclass(frozen=True)
class TierSpec:
    id: str
    rows: int
    row_depth: float      # seat row depth (tread) [m]
    riser: float          # height gained per row [m]
    d_start: float        # offset of the front row edge from the field boundary [m]
    z_base: float         # floor height of the front row [m]
    aisles: int           # number of radial aisles (= number of seating blocks)
    aisle_width: float    # [m]
    aisle_phase: float    # shift of aisle positions as a fraction of one block
    cross_aisle: CrossAisleSpec
    vomitories: VomitorySpec
    pitch_gates: PitchGateSpec | None


@dataclass(frozen=True)
class ExitSpec:
    count: int
    width: float
    phase: float


@dataclass(frozen=True)
class ConnectorSpec:
    kind: str        # "stair" or "ramp"
    to: str          # id of the lower concourse
    count: int
    width: float
    angle_deg: float  # pitch of the stair / ramp
    phase: float


@dataclass(frozen=True)
class ConcourseSpec:
    id: str
    z: float
    d_inner: float
    d_outer: float
    exits: ExitSpec | None
    connectors: tuple[ConnectorSpec, ...]


@dataclass(frozen=True)
class StadiumSpec:
    key: str
    name: str
    sport: str
    description: str
    seat_width: float
    field: FieldSpec
    tiers: tuple[TierSpec, ...]
    concourses: tuple[ConcourseSpec, ...]
    source: str

    def concourse(self, cid: str) -> ConcourseSpec:
        for c in self.concourses:
            if c.id == cid:
                return c
        raise KeyError(cid)


# ---------------------------------------------------------------- parsing
def _every_offset(sec: Section) -> tuple[int, int]:
    every = sec.int("every", min=1)
    offset = sec.int("offset", 0, min=0)
    if offset >= every:
        raise ConfigError(f"{sec.where}.offset must be < every ({every})")
    return every, offset


def _parse_tier(sec: Section) -> TierSpec:
    rows = sec.int("rows", min=1)
    ca = sec.section("cross_aisle")
    cross = CrossAisleSpec(row=ca.int("row", min=0, max=rows), width=ca.float("width", min=0, strict_min=True))
    ca.finish()
    vs = sec.section("vomitories")
    v_every, v_off = _every_offset(vs)
    vom = VomitorySpec(every=v_every, offset=v_off, width=vs.float("width", min=0, strict_min=True),
                       min_length=vs.float("min_length", 0.0, min=0.0), concourse=vs.str("concourse"))
    vs.finish()
    pitch = None
    ps = sec.section("pitch_gates", None)
    if ps is not None:
        p_every, p_off = _every_offset(ps)
        pitch = PitchGateSpec(every=p_every, offset=p_off, width=ps.float("width", min=0, strict_min=True))
        ps.finish()
    tier = TierSpec(
        id=sec.str("id"), rows=rows,
        row_depth=sec.float("row_depth", min=0.5, max=1.5),
        riser=sec.float("riser", min=0.0, max=0.8),
        d_start=sec.float("d_start", min=0.0),
        z_base=sec.float("z_base"),
        aisles=sec.int("aisles", min=2),
        aisle_width=sec.float("aisle_width", min=0.6, max=5.0),
        aisle_phase=sec.float("aisle_phase", 0.5, min=0.0, max=1.0),
        cross_aisle=cross, vomitories=vom, pitch_gates=pitch,
    )
    sec.finish()
    return tier


def _parse_concourse(sec: Section) -> ConcourseSpec:
    cid = sec.str("id")
    d_inner = sec.float("d_inner", min=0.0)
    d_outer = sec.float("d_outer", min=0.0)
    if d_outer <= d_inner:
        raise ConfigError(f"{sec.where}: d_outer must be > d_inner")
    exits = None
    es = sec.section("exits", None)
    if es is not None:
        exits = ExitSpec(count=es.int("count", min=1), width=es.float("width", min=0.5, max=20.0),
                         phase=es.float("phase", 0.0, min=0.0, max=1.0))
        es.finish()
    conns = []
    for i, raw in enumerate(sec.list("connectors", [])):
        cs = Section(raw, f"{sec.where}.connectors[{i}]")
        kind = cs.str("kind")
        if kind not in ("stair", "ramp"):
            raise ConfigError(f"{cs.where}.kind must be 'stair' or 'ramp', got {kind!r}")
        conns.append(ConnectorSpec(kind=kind, to=cs.str("to"), count=cs.int("count", min=1),
                                   width=cs.float("width", min=0.6, max=20.0),
                                   angle_deg=cs.float("angle_deg", min=0.0, max=45.0, strict_min=True),
                                   phase=cs.float("phase", 0.0, min=0.0, max=1.0)))
        cs.finish()
    con = ConcourseSpec(id=cid, z=sec.float("z"), d_inner=d_inner, d_outer=d_outer,
                        exits=exits, connectors=tuple(conns))
    sec.finish()
    return con


def parse_stadium_spec(data: dict, key: str, source: str = "<dict>") -> StadiumSpec:
    root = Section(data, source)
    fs = root.section("field")
    shape = fs.str("shape")
    if shape != "superellipse":
        raise ConfigError(f"{fs.where}.shape: only 'superellipse' is supported")
    field = FieldSpec(shape=shape, a=fs.float("a", min=1.0), b=fs.float("b", min=1.0),
                      n=fs.float("n", min=2.0, max=20.0))
    fs.finish()
    tiers = tuple(_parse_tier(Section(t, f"{source}.tiers[{i}]")) for i, t in enumerate(root.list("tiers")))
    concourses = tuple(_parse_concourse(Section(c, f"{source}.concourses[{i}]"))
                       for i, c in enumerate(root.list("concourses")))
    spec = StadiumSpec(key=key, name=root.str("name"), sport=root.str("sport"),
                       description=root.str("description", ""),
                       seat_width=root.float("seat_width", min=0.4, max=0.8),
                       field=field, tiers=tiers, concourses=concourses, source=source)
    root.finish()
    _cross_validate(spec)
    return spec


def _cross_validate(spec: StadiumSpec) -> None:
    if not spec.tiers:
        raise ConfigError(f"{spec.source}: at least one tier is required")
    if not spec.concourses:
        raise ConfigError(f"{spec.source}: at least one concourse is required")
    ids = [t.id for t in spec.tiers] + [c.id for c in spec.concourses]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ConfigError(f"{spec.source}: duplicate level ids {dupes}")
    conc = {c.id: c for c in spec.concourses}
    for t in spec.tiers:
        if t.vomitories.concourse not in conc:
            raise ConfigError(f"{spec.source}: tier '{t.id}' vomitories lead to unknown concourse "
                              f"'{t.vomitories.concourse}'")
        if t.vomitories.every > t.aisles:
            raise ConfigError(f"{spec.source}: tier '{t.id}' vomitories.every exceeds number of aisles")
        if t.pitch_gates and t.pitch_gates.every > t.aisles:
            raise ConfigError(f"{spec.source}: tier '{t.id}' pitch_gates.every exceeds number of aisles")
    for c in spec.concourses:
        for k in c.connectors:
            if k.to not in conc or k.to == c.id:
                raise ConfigError(f"{spec.source}: concourse '{c.id}' connector leads to invalid '{k.to}'")
            if conc[k.to].z >= c.z:
                raise ConfigError(f"{spec.source}: connector from '{c.id}' to '{k.to}' must go downwards")
    if not any(c.exits for c in spec.concourses):
        raise ConfigError(f"{spec.source}: no exits defined on any concourse")


def load_stadium_spec(name_or_path: str) -> StadiumSpec:
    path = stadium_path(name_or_path)
    return parse_stadium_spec(load_yaml(path), key=Path(path).stem, source=Path(path).name)
