"""Agents: the occupied seats of a stadium with per-person attributes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crowdsim.crowd.population import Population
from crowdsim.crowd.scenarios import Scenario
from crowdsim.stadium.analysis import seat_egress_distance
from crowdsim.stadium.model import Stadium


@dataclass(frozen=True, eq=False)
class Agents:
    """Structure-of-arrays crowd. Index i is one person."""
    seat: np.ndarray          # seat index into Stadium.seats
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    tier: np.ndarray
    row: np.ndarray
    aisle: np.ndarray
    node: np.ndarray          # starting graph node index (aisle node at the seat row)
    lateral: np.ndarray       # walk along the row to the aisle [m]
    cls: np.ndarray           # mobility-class index into class_names
    class_names: tuple[str, ...]
    speed: np.ndarray         # desired free walking speed [m/s] (already x scenario speed_factor)
    radius: np.ndarray        # body radius [m]
    premovement: np.ndarray   # delay before moving [s]
    egress_distance: np.ndarray  # static shortest walk to the nearest open exit [m] (inf = cut off)
    seed: int
    scenario: str

    @property
    def n(self) -> int:
        return int(self.seat.size)

    def summary(self) -> dict:
        reachable = np.isfinite(self.egress_distance)
        out = {"scenario": self.scenario, "seed": self.seed, "agents": self.n}
        for i, name in enumerate(self.class_names):
            out[f"share_{name}"] = round(float(np.mean(self.cls == i)), 4) if self.n else 0.0
        out.update({
            "speed_mean": round(float(self.speed.mean()), 3),
            "speed_min": round(float(self.speed.min()), 3),
            "speed_max": round(float(self.speed.max()), 3),
            "premove_median_s": round(float(np.median(self.premovement)), 1),
            "premove_p90_s": round(float(np.percentile(self.premovement, 90)), 1),
            "egress_mean_m": round(float(self.egress_distance[reachable].mean()), 1) if reachable.any() else float("nan"),
            "egress_max_m": round(float(self.egress_distance[reachable].max()), 1) if reachable.any() else float("nan"),
            "unreachable": int((~reachable).sum()),
        })
        return out


def populate(stadium: Stadium, scenario: Scenario, population: Population, seed: int) -> Agents:
    """Fill seats according to the scenario. Same (stadium, scenario, seed) -> identical crowd."""
    rng = np.random.default_rng(seed)
    seats = stadium.seats
    n = int(round(scenario.occupancy * seats.n))
    idx = np.sort(rng.choice(seats.n, size=n, replace=False))
    shares = population.shares(scenario.mix)
    cls = rng.choice(len(population.classes), size=n, p=shares)
    speed = np.empty(n)
    radius = np.empty(n)
    for i, mc in enumerate(population.classes):
        m = cls == i
        k = int(m.sum())
        if k:
            speed[m] = mc.speed.sample(rng, k)
            radius[m] = mc.radius.sample(rng, k)
    speed *= scenario.speed_factor
    pre_dist = scenario.premovement or population.premovement
    egress = seat_egress_distance(stadium)[idx]
    return Agents(
        seat=idx, x=seats.x[idx], y=seats.y[idx], z=seats.z[idx], tier=seats.tier[idx], row=seats.row[idx],
        aisle=seats.aisle[idx], node=seats.node[idx], lateral=seats.lateral[idx], cls=cls,
        class_names=population.names, speed=speed, radius=radius,
        premovement=pre_dist.sample(rng, n), egress_distance=egress, seed=seed, scenario=scenario.name,
    )
