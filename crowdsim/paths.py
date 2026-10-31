"""Project paths, resolved relative to this file so commands work from any directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
STADIUM_DIR = CONFIG_DIR / "stadiums"
SCENARIO_DIR = CONFIG_DIR / "scenarios"
CROWD_DIR = CONFIG_DIR / "crowd"
RESULTS_DIR = PROJECT_ROOT / "results"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
