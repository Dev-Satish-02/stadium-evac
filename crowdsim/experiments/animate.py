"""``animate``: run models on one identical crowd while recording positions, then render a replay video."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from crowdsim.animation import Tracer
from crowdsim.animation.render import render
from crowdsim.configio import ConfigError
from crowdsim.crowd import load_population, load_scenario
from crowdsim.models import make_crowd, prepare, run_model
from crowdsim.models.params import load_model_params
from crowdsim.paths import RESULTS_DIR
from crowdsim.stadium import load_stadium


def animate(stadium_key: str, scenario_name: str, models: list[str], seed: int, overrides: dict | None = None,
            crowd_fraction: float = 1.0, frame_dt: float = 2.0, fps: int = 20, marker: float = 5.0,
            out_root: Path | None = None) -> Path:
    params = load_model_params(overrides)
    stadium = load_stadium(stadium_key)
    sc_list = load_scenario(scenario_name).expand()
    if len(sc_list) != 1:
        raise ConfigError(f"'{scenario_name}' is a sweep; animate a single scenario such as emergency")
    sc = sc_list[0]
    if not 0.0 < crowd_fraction <= 1.0:
        raise ConfigError("--crowd-fraction must be in (0, 1]")
    if crowd_fraction < 1.0:
        sc = replace(sc, occupancy=sc.occupancy * crowd_fraction)
    prep = prepare(stadium, sc, params)
    crowd = make_crowd(prep, load_population(), seed)
    print(f"{stadium_key} / {sc.name}: {crowd.n:,} people, seed {seed}, models {', '.join(models)}")
    traces = {}
    for m in models:
        tracer = Tracer(prep.stadium, prep.net, crowd, frame_dt)
        out = run_model(m, prep, crowd, seed, tracer=tracer)
        traces[m] = tracer.finish(out.result.exit_time, m)
        print(f"  {m:8s} {len(traces[m].t)} frames, {out.result.compute_time_s:.0f} s compute")
    root = out_root or RESULTS_DIR / "animations" / stadium_key
    name = f"{sc.name}_{'-'.join(models)}_seed{seed}"
    title = f"{stadium.name} - {sc.name} - {crowd.n:,} people (same crowd for every model)"
    print("rendering video (a few minutes for large crowds) ...")
    return render(traces, prep.stadium, root / name, title=title, fps=fps, marker=marker)
