import numpy as np
import pytest

from crowdsim.configio import list_stadium_presets
from crowdsim.crowd import apply_scenario, load_scenario
from crowdsim.stadium import SINK, load_stadium, seat_egress_distance, summary, validate


@pytest.fixture(scope="module")
def stadiums():
    return {k: load_stadium(k) for k in list_stadium_presets()}


def test_three_presets_exist(stadiums):
    assert {"football_bowl", "cricket_oval", "indoor_arena"} <= set(stadiums)


@pytest.mark.parametrize("key", ["football_bowl", "cricket_oval", "indoor_arena"])
def test_preset_passes_all_integrity_checks(stadiums, key):
    failed = [c for c in validate(stadiums[key]) if not c.passed]
    assert not failed, failed


def test_build_is_deterministic():
    a, b = load_stadium("indoor_arena"), load_stadium("indoor_arena")
    assert a.capacity == b.capacity
    assert np.array_equal(a.seats.x, b.seats.x)


def test_upper_tier_is_further_from_exits(stadiums):
    st = stadiums["football_bowl"]
    d = seat_egress_distance(st)
    lower, upper = d[st.seats.tier == 0], d[st.seats.tier == 1]
    assert upper.mean() > lower.mean()


def test_height_counts_in_walking_distance(stadiums):
    """Top-corner seats in the football upper tier must walk further than 2-D plan distance suggests."""
    st = stadiums["football_bowl"]
    seats = st.seats
    i = int(np.argmax(np.where(seats.tier == 1, seats.z, -np.inf)))
    exits = np.array([[e.x, e.y] for e in st.gates()])
    plan = np.min(np.linalg.norm(exits - [seats.x[i], seats.y[i]], axis=1))
    assert seat_egress_distance(st)[i] > plan + seats.z[i]  # at least the vertical drop on top


def test_arena_is_sunken(stadiums):
    st = stadiums["indoor_arena"]
    assert st.seats.z.max() < 0.0
    assert all(e.z == 0.0 for e in st.gates())


def test_pitch_gates_closed_by_default(stadiums):
    st = stadiums["cricket_oval"]
    pitch = [e for e in st.exits if e.kind == "pitch"]
    assert pitch and not any(e.enabled for e in pitch)
    assert all(not st.graph.has_edge(e.node, SINK) for e in pitch)


def test_blocking_gates_lengthens_egress(stadiums):
    st = stadiums["football_bowl"]
    blocked = apply_scenario(st, load_scenario("blocked_exit"))
    base, worse = seat_egress_distance(st), seat_egress_distance(blocked)
    assert np.all(worse >= base - 1e-9) and worse.mean() > base.mean()
    assert not st.graph is blocked.graph
    assert all(e.enabled for e in st.gates())  # original untouched


def test_pitch_refuge_shortens_cricket_egress(stadiums):
    st = stadiums["cricket_oval"]
    open_ = apply_scenario(st, load_scenario("pitch_refuge"))
    assert seat_egress_distance(open_).mean() < seat_egress_distance(st).mean()


def test_exit_width_sweep_scales_widths(stadiums):
    st = stadiums["indoor_arena"]
    base_w = {e.node: e.width for e in st.gates()}
    for sc in load_scenario("exit_width_sweep").expand():
        s = apply_scenario(st, sc)
        assert summary(s)["exit_width_m"] == pytest.approx(summary(st)["exit_width_m"] * sc.exit_width_factor)
        # the walkway through each gate (used for capacity by the models) is scaled too
        for e in s.gates():
            w = [d["width"] for _, _, d in s.graph.edges(e.node, data=True) if d["kind"] == "exit"]
            assert w == [pytest.approx(base_w[e.node] * sc.exit_width_factor)]
    # the base stadium is never modified
    for e in st.gates():
        w = [d["width"] for _, _, d in st.graph.edges(e.node, data=True) if d["kind"] == "exit"]
        assert w == [pytest.approx(base_w[e.node])]
