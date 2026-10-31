import math
from dataclasses import replace

import numpy as np
import pytest

from crowdsim.configio import ConfigError
from crowdsim.crowd import load_population, load_scenario
from crowdsim.models import MODEL_ORDER, make_crowd, prepare, run_model
from crowdsim.models import network as network_model
from crowdsim.models import sfpe as sfpe_model
from crowdsim.models.domain import room_domain
from crowdsim.models.egressnet import EgressNet
from crowdsim.models.micro import lattice_cells, run_room
from crowdsim.models.params import load_model_params, parse_overrides
from crowdsim.models.sfpe import fifo_departures
from crowdsim.stadium import load_stadium
from crowdsim.validation.suite import chain_net, simple_crowd


@pytest.fixture(scope="module")
def P():
    return load_model_params()


def test_params_override_and_validation(P):
    q = P.with_overrides({"sf.tau": "0.8", "movement.k.stair": 1.0, "network.routing": "static"})
    assert q.sf.tau == 0.8 and q.movement.k["stair"] == 1.0 and q.network.routing == "static"
    with pytest.raises(ConfigError):
        P.with_overrides({"sf.nonsense": 1})
    with pytest.raises(ConfigError):
        P.with_overrides({"sf.dt": 0.2})
    assert parse_overrides(["a.b=1", "c.d=x"]) == {"a.b": "1", "c.d": "x"}


def test_fifo_departures():
    dep = fifo_departures(np.array([0.0, 0.0, 0.0, 10.0]), cap=2.0)
    assert dep.tolist() == [0.0, 0.5, 1.0, 10.0]


def test_sfpe_matches_hand_calculation(P):
    mp = P.movement
    net, nodes = chain_net(P, [("concourse", 20.0, 2.0), ("exit", 0.5, 1.0)], 1.0)
    o = sfpe_model.run(net, simple_crowd(100, nodes.index("n0"), 1.2), P)
    c_door = mp.k["level"] / (4 * mp.sfpe_a) * (1.0 - 2 * mp.boundary_layer["exit"])
    assert o.result.exit_time.max() == pytest.approx(20 / 1.2 + 99 / c_door + 0.5 / 1.2, rel=1e-12)


def test_network_free_walk_and_capacity(P):
    Pn = P.with_overrides({"movement.row_factor": 1.0})
    net, nodes = chain_net(Pn, [("concourse", 28.0, 3.0)], 3.0)
    o = network_model.run(net, simple_crowd(1, nodes.index("n0"), 1.34), Pn)
    assert o.result.exit_time[0] == pytest.approx(28.0 / 1.34, rel=1e-6)
    net, nodes = chain_net(P, [("aisle", 0.9, 1.3)] * 24, 1.3)
    o = network_model.run(net, simple_crowd(800, nodes.index("n0"), 1.34, lateral=0.5), P)
    t = np.sort(o.result.exit_time)
    assert (800 - 81) / (t[-1] - t[80]) == pytest.approx(net.capacity[0], rel=0.03)


def test_candidate_moves_always_get_closer(P):
    st = load_stadium("cricket_oval")
    net = EgressNet.from_stadium(st, P.movement)
    rows, cols = np.nonzero(net.cand_edge >= 0)
    nxt = net.cand_node[rows, cols]
    assert np.all(net.dist[nxt] < net.dist[rows])


def test_room_domain_fields(P):
    dom = room_domain(8.0, 6.0, 1.2, P)
    assert np.all(np.isfinite(dom.S[dom.walk]))
    assert np.all(dom.S[dom.exit_id >= 0] == 0)
    ix, iy = dom.nav_index(4.0, 3.0)
    wi, wj = int((4.0 - dom.wx0) / dom.wcell), int((3.0 - dom.wy0) / dom.wcell)
    assert dom.sd[wi, wj] == pytest.approx(3.0, abs=0.2)
    assert dom.dirx[ix, iy] > 0.8  # points (mainly) towards the door at x = 8


@pytest.mark.parametrize("kind", ["ca", "sf"])
def test_micro_room_evacuates_and_is_reproducible(P, kind):
    dom = room_domain(8.0, 8.0, 1.6, P)
    cells = lattice_cells(dom, 60, rng=np.random.default_rng(1))
    v, r = np.full(60, 1.34), np.full(60, 0.23)
    a, ia, _ = run_room(kind, dom, cells, v, r, P, seed=3)
    b, _, _ = run_room(kind, dom, cells, v, r, P, seed=3)
    assert np.all(np.isfinite(a)) and np.all(ia == 0)
    assert np.array_equal(a, b)


def test_all_models_on_small_stadium_crowd(P):
    st = load_stadium("indoor_arena")
    sc = replace(load_scenario("blocked_exit"), occupancy=0.05, name="tiny")
    prep = prepare(st, sc, P)
    crowd = make_crowd(prep, load_population(), 11)
    open_ids = {i for i, e in enumerate(prep.stadium.exits) if e.enabled}
    for m in MODEL_ORDER:
        r = run_model(m, prep, crowd, 11).result
        assert np.all(np.isfinite(r.exit_time)), m
        assert set(np.unique(r.exit_id).tolist()) <= open_ids, m
        assert 0 not in set(r.exit_id.tolist()) and 1 not in set(r.exit_id.tolist())  # G0, G1 blocked
