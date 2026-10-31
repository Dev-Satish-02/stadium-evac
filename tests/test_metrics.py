import math

import numpy as np
import pytest

from crowdsim.metrics import (EvacuationResult, Timer, crush_risk, density_grid, evacuation_curve,
                              evacuation_time, exit_balance, exit_specific_flow, fruin_los, peak_density,
                              percentile_time, queue_length, summarise)


def test_percentiles_known_case():
    t = np.arange(1.0, 11.0)  # 10 people out at 1..10 s
    assert percentile_time(t, 50) == 5.0
    assert percentile_time(t, 90) == 9.0
    assert evacuation_time(t) == 10.0


def test_percentile_inf_when_not_reached():
    t = np.r_[np.arange(1.0, 9.0), [np.inf, np.inf]]
    assert percentile_time(t, 80) == 8.0
    assert math.isinf(percentile_time(t, 90))


def test_evacuation_curve():
    t = np.array([1.0, 2.0, 2.0, 5.0])
    assert evacuation_curve(t, [0, 1, 2, 4, 5]).tolist() == [0, 1, 3, 3, 4]


def test_specific_flow_uniform_stream():
    # two exits: 1 m wide at 1 p/s, 2 m wide at 3 p/s -> 1.0 and 1.5 p/(m s)
    t = np.r_[np.arange(100.0), np.arange(300) / 3.0]
    ids = np.r_[np.zeros(100, int), np.ones(300, int)]
    flow = exit_specific_flow(t, ids, [1.0, 2.0])
    assert flow == pytest.approx([1.0, 1.5], rel=1e-9)


def test_exit_balance():
    t = np.r_[np.arange(10.0), np.arange(10.0)]
    ids = np.r_[np.zeros(10, int), np.ones(10, int)]
    bal = exit_balance(t, ids, [1.0, 1.0, 1.0])
    assert bal["unused_open_exits"] == 1
    bal2 = exit_balance(t, ids, [1.0, 1.0, 1.0], open_exits=[True, True, False])
    assert bal2["load_cv"] == 0.0 and bal2["clearance_spread_s"] == 0.0


def test_result_validation():
    with pytest.raises(ValueError):
        EvacuationResult(exit_time=[1.0, np.inf], exit_id=[0, 0], exit_width=[1.0])
    with pytest.raises(ValueError):
        EvacuationResult(exit_time=[1.0], exit_id=[3], exit_width=[1.0])


def test_summarise_runs():
    res = EvacuationResult(exit_time=np.arange(50.0), exit_id=np.zeros(50, int), exit_width=[1.0])
    s = summarise(res)
    assert s["evacuated_fraction"] == 1.0 and s["T100_s"] == 49.0


def test_density_grid_and_peak():
    xy = np.array([[0.1, 0.1], [0.2, 0.5], [0.9, 0.9], [0.5, 0.5], [3.5, 3.5]])
    dens, xe, ye = density_grid(xy, cell=1.0)
    assert dens.max() == 4.0 and dens.sum() == 5.0
    assert peak_density([xy, xy[:2]]) == 4.0
    dens2, _, _ = density_grid(xy, cell=0.5)
    assert dens2.sum() * 0.25 == pytest.approx(5.0)


def test_fruin_and_crush():
    assert "".join(fruin_los([0.2, 0.4, 0.5, 1.0, 2.0, 3.0])) == "ABCDEF"
    assert crush_risk([4.9, 5.0, 6.0]).tolist() == [False, True, True]


def test_queue_length():
    xy = np.array([[0, 0], [1, 0], [3, 0]])
    assert queue_length(xy, (0, 0), 1.5) == 2


def test_timer():
    with Timer() as tm:
        sum(range(1000))
    assert tm.elapsed >= 0
