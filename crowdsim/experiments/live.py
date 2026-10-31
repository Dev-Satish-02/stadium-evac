"""``python run.py live``: watch a simulation while it runs, then get every result file.

Runs the chosen models one after another on the SAME crowd. Each spatial model
(network, ca, sf) gets a live plan panel; sfpe (an analytic calculation) only
adds its evacuation curve. When a model finishes its results are written exactly
like ``simulate`` (agents.csv, timeseries.csv, summary.json, figures); with
several models the comparison tables and figures of ``compare`` are written too.
"""
from __future__ import annotations

import gzip
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import crowdsim.analysis.plots as plots  # imported BEFORE the GUI backend is selected (it sets Agg on import)
import crowdsim.animation.render  # noqa: F401  (same reason)
from crowdsim.animation.live import LiveTracer, LiveView, pick_backend
from crowdsim.configio import ConfigError
from crowdsim.crowd import load_population, load_scenario
from crowdsim.experiments.runner import _print_row, _safe, _write_tables, run_summary, save_run
from crowdsim.models import MODEL_ORDER, make_crowd, prepare, run_model
from crowdsim.models.params import load_model_params
from crowdsim.paths import RESULTS_DIR
from crowdsim.stadium import load_stadium


def write_trajectories(path: Path, trace_t, trace_xy, trace_state) -> None:
    """Long-format trajectories (one row per person per frame, evacuated people omitted), gzip CSV."""
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("t_s,agent,x_m,y_m,state\n")
        for t, xy, st in zip(trace_t, trace_xy, trace_state):
            idx = np.flatnonzero(np.isfinite(xy[:, 0]))
            block = np.column_stack([np.full(idx.size, t), idx, xy[idx, 0], xy[idx, 1], st[idx]])
            np.savetxt(f, block, fmt=["%.1f", "%d", "%.2f", "%.2f", "%d"], delimiter=",")


def live(stadium_key: str, scenario_name: str, models: list[str], seed: int, overrides: dict | None = None,
         crowd_fraction: float = 1.0, frame_dt: float = 2.0, record: bool = False, window: bool = True,
         save_trajectories: bool = False, out_root: Path | None = None, hold: bool = True,
         marker: float = 4.0, fps: int = 20) -> Path:
    bad = [m for m in models if m not in MODEL_ORDER]
    if bad or not models:
        raise ConfigError(f"unknown model(s) {bad}; choose from {', '.join(MODEL_ORDER)}")
    if not 0.0 < crowd_fraction <= 1.0:
        raise ConfigError("--crowd-fraction must be in (0, 1]")
    if frame_dt <= 0:
        raise ConfigError("--frame-dt must be > 0")
    params = load_model_params(overrides)
    stadium = load_stadium(stadium_key)
    sc_list = load_scenario(scenario_name).expand()
    if len(sc_list) != 1:
        raise ConfigError(f"'{scenario_name}' is a sweep (several runs); use 'compare' for sweeps, "
                          "or pick a single scenario such as emergency")
    sc = sc_list[0]
    if crowd_fraction < 1.0:
        sc = replace(sc, occupancy=sc.occupancy * crowd_fraction,
                     name=f"{sc.name}_{int(round(crowd_fraction * 100))}pct")
    backend = pick_backend(window)
    prep = prepare(stadium, sc, params)
    crowd = make_crowd(prep, load_population(), seed)
    order = sorted(models, key=lambda m: (m != "sfpe", models.index(m)))  # sfpe (instant) first
    panels = [m for m in order if m != "sfpe"]
    root = (out_root or RESULTS_DIR / "live") / stadium_key / _safe(sc.name)
    tag = "-".join(order)
    print(f"{stadium.name} / {sc.name}: {crowd.n:,} people, seed {seed}, models {', '.join(order)} "
          f"(window: {backend if window else 'none'})")
    view = LiveView(prep.stadium, panels, crowd.n, f"{stadium.name} - {sc.name} - {crowd.n:,} people - seed {seed}",
                    frame_dt, window=window, record=(root / f"live_{tag}_seed{seed}") if record else None,
                    marker=marker, fps=fps)
    outputs, rows = {}, []
    for m in order:
        t0 = time.perf_counter()
        if m == "sfpe":
            out = run_model(m, prep, crowd, seed)
            view.add_curve(m, out.result.exit_time)
            tracer = None
        else:
            view.start(m)
            drawn0 = view.draw_seconds
            tracer = LiveTracer(view, prep.stadium, prep.net, crowd, frame_dt, keep=save_trajectories)
            out = run_model(m, prep, crowd, seed, tracer=tracer)
            # compute time without the time spent drawing the window (so it stays comparable with 'compare')
            out.result.compute_time_s = max(out.result.compute_time_s - (view.draw_seconds - drawn0), 0.0)
            view.finish(m, out.result.exit_time)
        row = run_summary(out, m, stadium_key, sc.name, seed, prep)
        row["wall_time_s"] = time.perf_counter() - t0
        _print_row(row)
        d = root / "runs" / m / f"seed{seed}"
        save_run(d, out, row, prep, make_plots=True)
        if tracer is not None and save_trajectories and tracer._t:
            write_trajectories(d / "trajectories.csv.gz", tracer._t, tracer._xy, tracer._state)
        outputs[m] = [out]
        rows.append(row)
    df = pd.DataFrame(rows)
    _write_tables(root, df)
    if len(order) > 1:
        plots.comparison(root, outputs, df, prep)
    print(f"\nresults folder: {root}")
    print((root / "summary.md").read_text(encoding="utf-8"))
    view.close(hold=hold)
    return root
