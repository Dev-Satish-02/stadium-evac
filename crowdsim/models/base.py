"""Common model inputs and outputs."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from crowdsim.crowd.agents import Agents
from crowdsim.metrics import EvacuationResult


@dataclass(frozen=True, eq=False)
class Crowd:
    """What every model needs to know about each person."""
    start_node: np.ndarray    # graph node where the person joins the network (aisle node at their row)
    lateral: np.ndarray       # walk along the row before reaching start_node [m]
    speed: np.ndarray         # desired free walking speed on the level [m/s]
    premovement: np.ndarray   # delay before moving [s]
    radius: np.ndarray        # body radius [m]
    herding: float = 0.0      # scenario herding / panic parameter (0..1)
    labels: dict = field(default_factory=dict)  # extra per-agent columns kept in the output (tier, class, ...)

    @property
    def n(self) -> int:
        return int(self.speed.size)

    @classmethod
    def from_agents(cls, agents: Agents, herding: float) -> "Crowd":
        labels = {"seat": agents.seat, "tier": agents.tier, "row": agents.row,
                  "cls": np.array(agents.class_names, dtype=object)[agents.cls],
                  "egress_distance": agents.egress_distance}
        return cls(start_node=agents.node, lateral=agents.lateral, speed=agents.speed,
                   premovement=agents.premovement, radius=agents.radius, herding=herding, labels=labels)


@dataclass(eq=False)
class RunOutput:
    """Everything one simulation run produces."""
    result: EvacuationResult
    crowd: Crowd
    timeseries: pd.DataFrame                  # t, evacuated, ... sampled every record_dt
    peak_density: float = float("nan")        # persons/m^2 (NaN if the model has no density)
    peak_density_time: float = float("nan")
    peak_snapshot: np.ndarray | None = None   # (N, 2) positions at peak density (micro models)
    density_note: str = ""
    info: dict = field(default_factory=dict)

    def agent_table(self) -> pd.DataFrame:
        df = pd.DataFrame({"exit_time_s": self.result.exit_time, "exit_id": self.result.exit_id,
                           "speed": self.crowd.speed, "premovement_s": self.crowd.premovement,
                           "lateral_m": self.crowd.lateral})
        for k, v in self.crowd.labels.items():
            df[k] = v
        return df

