"""YAML loading plus small helpers for strict, readable config validation.

Every config section is read through a ``Section`` so that a missing key,
a wrong type or an unknown (misspelt) key fails loudly with the file and
key path in the message, instead of silently producing a wrong stadium.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml

from crowdsim.paths import CROWD_DIR, SCENARIO_DIR, STADIUM_DIR


class ConfigError(ValueError):
    """Raised when a configuration file is missing, malformed or inconsistent."""


_REQUIRED = object()


def load_yaml(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


class Section:
    """Typed, strict accessor for one mapping inside a config file."""

    def __init__(self, data: Any, where: str):
        if not isinstance(data, dict):
            raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
        self._data = data
        self._used: set[str] = set()
        self.where = where

    def has(self, key: str) -> bool:
        """True if the key is present and not null (also marks it as read)."""
        self._used.add(key)
        return key in self._data and self._data[key] is not None

    def keys(self) -> list:
        return list(self._data)

    def raw(self, key: str, default: Any = _REQUIRED) -> Any:
        self._used.add(key)
        if key not in self._data or self._data[key] is None:
            if default is _REQUIRED:
                raise ConfigError(f"{self.where}: missing required key '{key}'")
            return default
        return self._data[key]

    def str(self, key: str, default: Any = _REQUIRED) -> str:
        val = self.raw(key, default)
        if not isinstance(val, str):
            raise ConfigError(f"{self.where}.{key}: expected text, got {val!r}")
        return val

    def float(self, key: str, default: Any = _REQUIRED, *, min: float | None = None,
              max: float | None = None, strict_min: bool = False) -> float:
        val = self.raw(key, default)
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ConfigError(f"{self.where}.{key}: expected a number, got {val!r}")
        val = float(val)
        if min is not None and (val <= min if strict_min else val < min):
            op = ">" if strict_min else ">="
            raise ConfigError(f"{self.where}.{key}: must be {op} {min}, got {val}")
        if max is not None and val > max:
            raise ConfigError(f"{self.where}.{key}: must be <= {max}, got {val}")
        return val

    def int(self, key: str, default: Any = _REQUIRED, *, min: int | None = None,
            max: int | None = None) -> int:
        val = self.raw(key, default)
        if isinstance(val, bool) or not isinstance(val, int):
            raise ConfigError(f"{self.where}.{key}: expected an integer, got {val!r}")
        if min is not None and val < min:
            raise ConfigError(f"{self.where}.{key}: must be >= {min}, got {val}")
        if max is not None and val > max:
            raise ConfigError(f"{self.where}.{key}: must be <= {max}, got {val}")
        return val

    def bool(self, key: str, default: Any = _REQUIRED) -> bool:
        val = self.raw(key, default)
        if not isinstance(val, bool):
            raise ConfigError(f"{self.where}.{key}: expected true/false, got {val!r}")
        return val

    def section(self, key: str, default: Any = _REQUIRED) -> "Section | None":
        val = self.raw(key, default)
        if val is None:
            return None
        return Section(val, f"{self.where}.{key}")

    def list(self, key: str, default: Any = _REQUIRED) -> list:
        val = self.raw(key, default)
        if not isinstance(val, list):
            raise ConfigError(f"{self.where}.{key}: expected a list, got {val!r}")
        return val

    def finish(self) -> None:
        """Fail on keys that were never read (usually typos)."""
        unknown = sorted(set(self._data) - self._used)
        if unknown:
            raise ConfigError(f"{self.where}: unknown key(s) {unknown}")


def _names(folder: Path) -> list[str]:
    return sorted(p.stem for p in folder.glob("*.yaml"))


def list_stadium_presets() -> list[str]:
    return _names(STADIUM_DIR)


def list_scenarios() -> list[str]:
    return _names(SCENARIO_DIR)


def resolve(name_or_path: str, folder: Path, kind: str, choices: Iterable[str]) -> Path:
    """Accept a preset name (e.g. 'football_bowl') or a path to a YAML file."""
    candidate = Path(name_or_path)
    if candidate.suffix in (".yaml", ".yml") and candidate.is_file():
        return candidate
    path = folder / f"{name_or_path}.yaml"
    if not path.is_file():
        raise ConfigError(f"Unknown {kind} '{name_or_path}'. Available: {', '.join(choices)}")
    return path


def stadium_path(name: str) -> Path:
    return resolve(name, STADIUM_DIR, "stadium preset", list_stadium_presets())


def scenario_path(name: str) -> Path:
    return resolve(name, SCENARIO_DIR, "scenario", list_scenarios())


def population_path() -> Path:
    return CROWD_DIR / "population.yaml"
