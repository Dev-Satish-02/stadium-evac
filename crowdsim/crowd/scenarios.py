"""Evacuation scenarios: loading, parameter sweeps, and applying a scenario to a stadium."""
from __future__ import annotations

from dataclasses import dataclass, replace

from crowdsim.configio import ConfigError, Section, load_yaml, scenario_path
from crowdsim.crowd.population import LogNormal, parse_lognormal
from crowdsim.stadium.model import Stadium
from crowdsim.stadium.network import sync_exit_edges

SWEEPABLE = ("occupancy", "speed_factor", "exit_width_factor")


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    occupancy: float              # fraction of seats occupied
    speed_factor: float           # multiplies every agent's free walking speed (urgency)
    mix: dict | None              # class shares overriding the population default
    premovement: LogNormal | None # overrides the population default if given
    blocked_exits: tuple[int, ...]  # gate indices (G0, G1, ...) closed in this scenario
    exit_width_factor: float      # multiplies every gate width
    use_pitch_gates: bool         # open the pitch gates (only stadiums that have them)
    herding: float                # 0 = individual route choice, 1 = follow the crowd (used by Phase 2 models)
    sweep_param: str | None = None
    sweep_values: tuple[float, ...] = ()
    base: str | None = None       # parent scenario name for sweep variants

    def expand(self) -> list["Scenario"]:
        """A sweep scenario becomes one concrete scenario per value; others return [self]."""
        if not self.sweep_param:
            return [self]
        return [replace(self, name=f"{self.name}@{self.sweep_param}={v:g}", sweep_param=None,
                        sweep_values=(), base=self.name, **{self.sweep_param: float(v)})
                for v in self.sweep_values]


def parse_scenario(data: dict, source: str) -> Scenario:
    root = Section(data, source)
    mix = None
    if root.has("mix"):
        ms = root.section("mix")
        mix = {k: ms.float(k, min=0.0, max=1.0) for k in ms.keys()}
        ms.finish()
    pre = parse_lognormal(root.section("premovement")) if root.has("premovement") else None
    blocked = root.list("blocked_exits", [])
    if not all(isinstance(b, int) and not isinstance(b, bool) and b >= 0 for b in blocked):
        raise ConfigError(f"{source}.blocked_exits: must be a list of gate indices (0, 1, ...)")
    sweep_param, sweep_values = None, ()
    if root.has("sweep"):
        sw = root.section("sweep")
        sweep_param = sw.str("param")
        if sweep_param not in SWEEPABLE:
            raise ConfigError(f"{sw.where}.param must be one of {SWEEPABLE}")
        vals = sw.list("values")
        if not vals or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 for v in vals):
            raise ConfigError(f"{sw.where}.values must be a non-empty list of positive numbers")
        sweep_values = tuple(float(v) for v in vals)
        sw.finish()
    sc = Scenario(
        name=root.str("name"), description=root.str("description", ""),
        occupancy=root.float("occupancy", min=0.0, max=1.0, strict_min=True),
        speed_factor=root.float("speed_factor", 1.0, min=0.2, max=3.0),
        mix=mix, premovement=pre, blocked_exits=tuple(sorted(set(blocked))),
        exit_width_factor=root.float("exit_width_factor", 1.0, min=0.1, max=5.0),
        use_pitch_gates=root.bool("use_pitch_gates", False),
        herding=root.float("herding", 0.0, min=0.0, max=1.0),
        sweep_param=sweep_param, sweep_values=sweep_values,
    )
    root.finish()
    if sc.sweep_param == "occupancy" and any(v > 1 for v in sc.sweep_values):
        raise ConfigError(f"{source}: occupancy sweep values must be <= 1")
    return sc


def load_scenario(name_or_path: str) -> Scenario:
    path = scenario_path(name_or_path)
    return parse_scenario(load_yaml(path), source=path.name)


def apply_scenario(stadium: Stadium, sc: Scenario) -> Stadium:
    """Return a copy of the stadium with this scenario's exit changes applied."""
    gates = stadium.gates()
    bad = [i for i in sc.blocked_exits if i >= len(gates)]
    if bad:
        raise ConfigError(f"scenario '{sc.name}': blocked gate(s) {bad} do not exist in "
                          f"'{stadium.spec.key}' (gates G0..G{len(gates) - 1})")
    exits = []
    for e in stadium.exits:
        if e.kind == "gate":
            e = replace(e, width=e.width * sc.exit_width_factor, enabled=e.index not in sc.blocked_exits)
        else:
            e = replace(e, enabled=sc.use_pitch_gates)
        exits.append(e)
    g = stadium.graph.copy()
    sync_exit_edges(g, exits)
    return stadium.with_changes(graph=g, exits=tuple(exits), scenario=sc.name)
