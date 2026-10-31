"""Model parameters (config/models.yaml) as typed, validated dataclasses.

Command-line overrides use dotted keys, e.g. ``--set sf.dt=0.01 ca.k_s=5``.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace

from crowdsim.configio import ConfigError, Section, load_yaml
from crowdsim.paths import CONFIG_DIR

MODELS_YAML = CONFIG_DIR / "models.yaml"
ELEMENT_TYPES = ("level", "ramp", "stair", "aisle")
EDGE_KINDS = ("aisle", "cross_aisle", "vomitory", "concourse", "stair", "ramp", "exit", "pitch_access")


@dataclass(frozen=True)
class MovementParams:
    sfpe_a: float
    d_free: float
    k: dict
    boundary_layer: dict
    ascending_factor: float
    row_factor: float
    jam_density: float


@dataclass(frozen=True)
class NetworkParams:
    dt: float
    routing: str
    t_max: float


@dataclass(frozen=True)
class CAParams:
    k_s: float
    k_d: float
    alpha: float
    delta: float
    friction: float
    friction_herding: float
    min_speed_ratio: float


@dataclass(frozen=True)
class SFParams:
    dt: float
    tau: float
    mass: float
    A: float
    B: float
    k_body: float
    kappa: float
    cutoff: float
    vmax_factor: float
    herd_radius: float
    space_margin: float
    noise: float
    wall_cell: float


@dataclass(frozen=True)
class MicroParams:
    nav_cell: float
    exit_depth: float
    density_cell: float
    record_dt: float
    stall_time: float


@dataclass(frozen=True)
class ModelParams:
    movement: MovementParams
    network: NetworkParams
    ca: CAParams
    sf: SFParams
    micro: MicroParams

    def with_overrides(self, overrides: dict[str, float | str] | None) -> "ModelParams":
        """Apply {'sf.dt': 0.01, 'movement.k.stair': 1.0, ...}."""
        p = self
        for key, value in (overrides or {}).items():
            parts = key.split(".")
            if len(parts) < 2 or parts[0] not in {f.name for f in fields(ModelParams)}:
                raise ConfigError(f"unknown parameter '{key}'")
            group = getattr(p, parts[0])
            name = parts[1]
            if name not in {f.name for f in fields(group)}:
                raise ConfigError(f"unknown parameter '{key}'")
            current = getattr(group, name)
            if isinstance(current, dict):
                if len(parts) != 3 or parts[2] not in current:
                    raise ConfigError(f"unknown parameter '{key}' (choices: {sorted(current)})")
                new = dict(current)
                new[parts[2]] = float(value)
                group = replace(group, **{name: new})
            else:
                if len(parts) != 2:
                    raise ConfigError(f"unknown parameter '{key}'")
                group = replace(group, **{name: type(current)(value) if not isinstance(current, str) else str(value)})
            p = replace(p, **{parts[0]: group})
        _validate(p)
        return p


def _floats(sec: Section, cls, zero_ok: tuple = ()):
    vals = {}
    for f in fields(cls):
        if f.type in ("float", float):
            vals[f.name] = sec.float(f.name, min=0.0, strict_min=f.name not in zero_ok)
    return vals


def _validate(p: ModelParams) -> None:
    if p.network.routing not in ("dynamic", "static"):
        raise ConfigError("network.routing must be 'dynamic' or 'static'")
    for t in ELEMENT_TYPES:
        if p.movement.k.get(t, 0) <= 0:
            raise ConfigError(f"movement.k.{t} must be > 0")
    for k in EDGE_KINDS:
        if p.movement.boundary_layer.get(k, -1) < 0:
            raise ConfigError(f"movement.boundary_layer.{k} must be >= 0")
    if not 0 < p.movement.sfpe_a * p.movement.jam_density < 1:
        raise ConfigError("movement.jam_density must be below 1/sfpe_a (where SFPE speed reaches 0)")
    ratio = p.micro.density_cell / p.micro.nav_cell
    if abs(ratio - round(ratio)) > 1e-6:
        raise ConfigError("micro.density_cell must be a whole multiple of micro.nav_cell")
    if p.sf.dt > 0.05:
        raise ConfigError("sf.dt above 0.05 s makes the social force model unstable")


def parse_model_params(data: dict, source: str = "models.yaml") -> ModelParams:
    root = Section(data, source)
    mv = root.section("movement")
    k = mv.section("k")
    kd = {t: k.float(t, min=0.0, strict_min=True) for t in ELEMENT_TYPES}
    k.finish()
    bl = mv.section("boundary_layer")
    bld = {t: bl.float(t, min=0.0) for t in EDGE_KINDS}
    bl.finish()
    movement = MovementParams(sfpe_a=mv.float("sfpe_a", min=0.0, strict_min=True),
                              d_free=mv.float("d_free", min=0.0), k=kd, boundary_layer=bld,
                              ascending_factor=mv.float("ascending_factor", min=0.1, max=1.0),
                              row_factor=mv.float("row_factor", min=0.1, max=1.0),
                              jam_density=mv.float("jam_density", min=0.5))
    mv.finish()
    sec = root.section("sfpe")
    sec.str("description", "")
    sec.finish()
    sec = root.section("network")
    network = NetworkParams(dt=sec.float("dt", min=0.0, strict_min=True, max=2.0), routing=sec.str("routing"),
                            t_max=sec.float("t_max", min=1.0))
    sec.finish()
    sec = root.section("ca")
    ca = CAParams(k_s=sec.float("k_s", min=0.0),
                  k_d=sec.float("k_d", min=0.0), alpha=sec.float("alpha", min=0.0, max=1.0),
                  delta=sec.float("delta", min=0.0, max=1.0), friction=sec.float("friction", min=0.0, max=1.0),
                  friction_herding=sec.float("friction_herding", min=0.0, max=1.0),
                  min_speed_ratio=sec.float("min_speed_ratio", min=0.05, max=1.0))
    sec.finish()
    sec = root.section("sf")
    sf = SFParams(**_floats(sec, SFParams, zero_ok=("space_margin", "noise")))
    sec.finish()
    sec = root.section("micro")
    micro = MicroParams(**_floats(sec, MicroParams))
    sec.finish()
    root.finish()
    p = ModelParams(movement=movement, network=network, ca=ca, sf=sf, micro=micro)
    _validate(p)
    return p


def load_model_params(overrides: dict | None = None) -> ModelParams:
    return parse_model_params(load_yaml(MODELS_YAML)).with_overrides(overrides)


def parse_overrides(items: list[str] | None) -> dict:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ConfigError(f"--set expects key=value, got '{item}'")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out
