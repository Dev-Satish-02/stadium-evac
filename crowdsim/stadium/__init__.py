"""Stadium presets, the parametric generator and static analysis."""
from crowdsim.stadium.analysis import Check, seat_egress_distance, summary, validate
from crowdsim.stadium.generator import build_stadium
from crowdsim.stadium.model import SINK, Exit, Level, Seats, Stadium
from crowdsim.stadium.spec import StadiumSpec, load_stadium_spec


def load_stadium(name_or_path: str) -> Stadium:
    """Load a preset by name (e.g. 'football_bowl') or YAML path and build it."""
    return build_stadium(load_stadium_spec(name_or_path))


__all__ = ["SINK", "Check", "Exit", "Level", "Seats", "Stadium", "StadiumSpec", "build_stadium",
           "load_stadium", "load_stadium_spec", "seat_egress_distance", "summary", "validate"]
