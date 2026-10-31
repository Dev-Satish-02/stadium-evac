import os

import numpy as np
import pytest

from crowdsim.experiments.live import live


def test_live_headless_writes_results(tmp_path):
    root = live("indoor_arena", "emergency", ["sfpe", "network"], seed=1, crowd_fraction=0.05, frame_dt=10.0,
                window=False, save_trajectories=True, out_root=tmp_path, hold=False)
    for f in ("summary.md", "summary.csv", "runs.csv", "evacuation_curves.png"):
        assert (root / f).exists(), f
    run = root / "runs" / "network" / "seed1"
    assert (run / "agents.csv").exists() and (run / "trajectories.csv.gz").exists()


jps = pytest.importorskip("jupedsim")


def test_jupedsim_room_and_geometry():
    from crowdsim.models.domain import room_domain
    from crowdsim.models.params import load_model_params
    from crowdsim.verification.jupedsim_check import domain_polygons, run_jupedsim

    P = load_model_params()
    dom = room_domain(6.0, 6.0, 1.2, P)
    walk, exits = domain_polygons(dom)
    assert walk.area == pytest.approx((dom.walk | (dom.exit_id >= 0)).sum() * dom.cell ** 2)
    assert list(exits) == [0]
    xy = np.array([[1.0, 1.0], [1.0, 3.0], [2.0, 2.0], [3.0, 4.0]])
    o = run_jupedsim("jps-cfsm", walk, exits, np.zeros(4), xy, np.zeros(4, int), np.full(4, 1.3), np.full(4, 0.2),
                     P, t_max=120)
    assert o["status"] == "completed" and np.isfinite(o["exit_time"]).all()
    assert o["exit_time"].min() > (6.0 - 3.0 - 0.5) / 1.3 * 0.8  # nobody faster than free walking
