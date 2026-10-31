import numpy as np
import pytest

from crowdsim.configio import ConfigError, list_scenarios
from crowdsim.crowd import apply_scenario, load_population, load_scenario, populate
from crowdsim.crowd.scenarios import parse_scenario
from crowdsim.stadium import load_stadium


@pytest.fixture(scope="module")
def arena():
    return load_stadium("indoor_arena")


@pytest.fixture(scope="module")
def pop():
    return load_population()


def test_all_scenarios_load():
    names = list_scenarios()
    assert len(names) >= 6
    for n in names:
        assert load_scenario(n).expand()


def test_sweep_expands():
    variants = load_scenario("exit_width_sweep").expand()
    assert [v.exit_width_factor for v in variants] == [0.5, 0.75, 1.0, 1.25, 1.5]
    assert all(v.base == "exit_width_sweep" and "@" in v.name for v in variants)


def test_populate_counts_and_reproducibility(arena, pop):
    sc = load_scenario("emergency")
    a = populate(arena, sc, pop, seed=1)
    b = populate(arena, sc, pop, seed=1)
    c = populate(arena, sc, pop, seed=2)
    assert a.n == round(0.95 * arena.capacity)
    assert np.array_equal(a.seat, b.seat) and np.array_equal(a.speed, b.speed)
    assert not np.array_equal(a.seat, c.seat)
    assert len(np.unique(a.seat)) == a.n


def test_speed_factor_applied(arena, pop):
    base = populate(arena, load_scenario("emergency"), pop, seed=3)
    panic = populate(arena, load_scenario("panic"), pop, seed=3)
    # same seed and occupancy -> same draws, so the ratio is exactly 1.6 / 1.2
    assert np.allclose(panic.speed / base.speed, 1.6 / 1.2)


def test_mixed_crowd_is_slower(arena, pop):
    normal = populate(arena, load_scenario("emergency"), pop, seed=4)
    mixed = populate(arena, load_scenario("mixed_crowd"), pop, seed=4)
    assert mixed.speed.mean() < normal.speed.mean()
    assert mixed.summary()["share_elderly"] == pytest.approx(0.20, abs=0.02)


def test_unknown_mix_class_rejected(pop):
    with pytest.raises(ConfigError, match="unknown class"):
        pop.shares({"robots": 1.0})


def test_blocked_gate_out_of_range(arena):
    sc = parse_scenario({"name": "x", "occupancy": 0.5, "blocked_exits": [99]}, "x.yaml")
    with pytest.raises(ConfigError, match="do not exist"):
        apply_scenario(arena, sc)


def test_scenario_typo_rejected():
    with pytest.raises(ConfigError, match="unknown key"):
        parse_scenario({"name": "x", "occupancy": 0.5, "speed_factr": 2}, "x.yaml")


def test_bodies_and_premovement_positive(arena, pop):
    a = populate(arena, load_scenario("normal_egress"), pop, seed=5)
    assert np.all(a.radius > 0) and np.all(a.premovement > 0)
    assert a.summary()["unreachable"] == 0
