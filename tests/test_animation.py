from dataclasses import replace

import numpy as np
import pytest

from crowdsim.animation import Tracer
from crowdsim.animation.render import render
from crowdsim.crowd import load_population, load_scenario
from crowdsim.models import make_crowd, prepare, run_model
from crowdsim.models.params import load_model_params
from crowdsim.stadium import load_stadium


@pytest.fixture(scope="module")
def setup():
    st = load_stadium("indoor_arena")
    sc = replace(load_scenario("emergency"), occupancy=0.05)
    prep = prepare(st, sc, load_model_params())
    crowd = make_crowd(prep, load_population(), 1)
    return prep, crowd


@pytest.mark.parametrize("model", ["network", "ca", "sf"])
def test_trace_is_consistent_with_result(setup, model):
    prep, crowd = setup
    tr = Tracer(prep.stadium, prep.net, crowd, frame_dt=5.0)
    out = run_model(model, prep, crowd, 1, tracer=tr)
    trace = tr.finish(out.result.exit_time, model)
    assert trace.xy.shape == (len(trace.t), crowd.n, 2)
    assert np.all(np.diff(trace.t) > 0)
    assert np.isfinite(trace.xy[0]).all()                       # everyone is seated / on the plan at t = 0
    assert np.isnan(trace.xy[-1]).sum() > 0.9 * crowd.n * 2     # nearly everyone has left at the end
    et = out.result.exit_time
    for i in (len(trace.t) // 3, len(trace.t) // 2):             # drawn people == people not yet out (+/- 1 s)
        n_in = int(np.isfinite(trace.xy[i][:, 0]).sum())
        assert np.sum(et > trace.t[i]) <= n_in <= np.sum(et > trace.t[i] - 1.0)


def test_render_writes_a_video(setup, tmp_path):
    prep, crowd = setup
    traces = {}
    for m in ("network", "ca"):
        tr = Tracer(prep.stadium, prep.net, crowd, frame_dt=10.0)
        out = run_model(m, prep, crowd, 1, tracer=tr)
        traces[m] = tr.finish(out.result.exit_time, m)
    p = render(traces, prep.stadium, tmp_path / "x", title="test", fps=5)
    assert p.exists() and p.stat().st_size > 1000
