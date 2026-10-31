"""The four evacuation models behind one interface.

    sfpe     Model 1  hydraulic / SFPE controlling-component calculation
    ca       Model 2  floor-field cellular automaton (hybrid: network upstream, CA on the exit concourse)
    sf       Model 3  social force model (hybrid: network upstream, SF on the exit concourse)
    network  Model 4  mesoscopic queueing network with congestion-aware routing
"""
from __future__ import annotations

from dataclasses import dataclass

from crowdsim.crowd import Population, Scenario, apply_scenario, populate
from crowdsim.models import network as _network
from crowdsim.models import sfpe as _sfpe
from crowdsim.models.base import Crowd, RunOutput
from crowdsim.models.egressnet import EgressNet
from crowdsim.models.hybrid import build_domains, run_hybrid
from crowdsim.models.params import ModelParams, load_model_params
from crowdsim.stadium.model import Stadium

MODEL_ORDER = ("sfpe", "ca", "sf", "network")
MODEL_LABELS = {"sfpe": "M1 Hydraulic (SFPE)", "ca": "M2 Cellular automaton", "sf": "M3 Social force",
                "network": "M4 Network (queueing)"}


@dataclass(eq=False)
class Prepared:
    """A stadium with a scenario applied, its egress network, and (lazily) micro domains."""
    stadium: Stadium
    scenario: Scenario
    net: EgressNet
    params: ModelParams
    _domains: tuple | None = None

    def domains(self):
        if self._domains is None:
            self._domains = build_domains(self.stadium, self.net, self.params)
        return self._domains


def prepare(stadium: Stadium, scenario: Scenario, params: ModelParams) -> Prepared:
    st = apply_scenario(stadium, scenario)
    return Prepared(stadium=st, scenario=scenario, net=EgressNet.from_stadium(st, params.movement), params=params)


def make_crowd(prep: Prepared, population: Population, seed: int) -> Crowd:
    agents = populate(prep.stadium, prep.scenario, population, seed)
    return Crowd.from_agents(agents, herding=prep.scenario.herding)


def run_model(model: str, prep: Prepared, crowd: Crowd, seed: int, tracer=None) -> RunOutput:
    if model == "sfpe":
        out = _sfpe.run(prep.net, crowd, prep.params, seed=seed)
    elif model == "network":
        out = _network.run(prep.net, crowd, prep.params, seed=seed, tracer=tracer)
    elif model in ("ca", "sf"):
        out = run_hybrid(model, prep.stadium, prep.net, crowd, prep.params, seed=seed, domains=prep.domains(), tracer=tracer)
    else:
        raise ValueError(f"unknown model '{model}'. Choose from {MODEL_ORDER}")
    out.result.scenario = prep.scenario.name
    return out


__all__ = ["MODEL_LABELS", "MODEL_ORDER", "Crowd", "ModelParams", "Prepared", "RunOutput", "load_model_params",
           "make_crowd", "prepare", "run_model"]
