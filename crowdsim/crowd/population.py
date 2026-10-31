"""Crowd composition: mobility classes with speed / body-size distributions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crowdsim.configio import ConfigError, Section, load_yaml, population_path


@dataclass(frozen=True)
class Distribution:
    mean: float
    sd: float
    min: float = -np.inf
    max: float = np.inf

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Truncated normal by rejection (falls back to clipping after 50 rounds)."""
        out = rng.normal(self.mean, self.sd, n)
        for _ in range(50):
            bad = (out < self.min) | (out > self.max)
            if not bad.any():
                break
            out[bad] = rng.normal(self.mean, self.sd, int(bad.sum()))
        return np.clip(out, self.min, self.max)


@dataclass(frozen=True)
class LogNormal:
    median: float
    sigma: float

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return self.median * np.exp(self.sigma * rng.standard_normal(n))


@dataclass(frozen=True)
class MobilityClass:
    name: str
    share: float
    speed: Distribution
    radius: Distribution


@dataclass(frozen=True)
class Population:
    classes: tuple[MobilityClass, ...]
    premovement: LogNormal

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.classes)

    def shares(self, override: dict | None = None) -> np.ndarray:
        if not override:
            return np.array([c.share for c in self.classes])
        unknown = set(override) - set(self.names)
        if unknown:
            raise ConfigError(f"crowd mix uses unknown class(es) {sorted(unknown)}; known: {list(self.names)}")
        shares = np.array([float(override.get(n, 0.0)) for n in self.names])
        _check_shares(shares, "scenario mix")
        return shares


def _check_shares(shares: np.ndarray, where: str) -> None:
    if np.any(shares < 0) or abs(shares.sum() - 1.0) > 1e-6:
        raise ConfigError(f"{where}: class shares must be >= 0 and sum to 1 (got {shares.sum():.4f})")


def _dist(sec: Section, bounded: bool) -> Distribution:
    mean = sec.float("mean", min=0.0, strict_min=True)
    sd = sec.float("sd", min=0.0)
    if bounded:
        lo, hi = sec.float("min", min=0.0, strict_min=True), sec.float("max", min=0.0, strict_min=True)
        if not lo <= mean <= hi:
            raise ConfigError(f"{sec.where}: need min <= mean <= max")
        d = Distribution(mean, sd, lo, hi)
    else:
        d = Distribution(mean, sd, max(0.10, mean - 4 * sd), mean + 4 * sd)
    sec.finish()
    return d


def parse_lognormal(sec: Section) -> LogNormal:
    ln = LogNormal(median=sec.float("median", min=0.0), sigma=sec.float("sigma", min=0.0))
    sec.finish()
    return ln


def parse_population(data: dict, source: str = "population.yaml") -> Population:
    root = Section(data, source)
    cls_sec = root.section("classes")
    classes = []
    for name in cls_sec.keys():
        cs = cls_sec.section(name)
        classes.append(MobilityClass(name=name, share=cs.float("share", min=0.0, max=1.0),
                                     speed=_dist(cs.section("speed"), bounded=True),
                                     radius=_dist(cs.section("radius"), bounded=False)))
        cs.finish()
    cls_sec.finish()
    if not classes:
        raise ConfigError(f"{source}: at least one mobility class is required")
    _check_shares(np.array([c.share for c in classes]), f"{source}.classes")
    pop = Population(classes=tuple(classes), premovement=parse_lognormal(root.section("premovement")))
    root.finish()
    return pop


def load_population(path=None) -> Population:
    path = path or population_path()
    return parse_population(load_yaml(path), source=str(getattr(path, "name", path)))
