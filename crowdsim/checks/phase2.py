"""End-to-end smoke test of Phase 2: every model on every stadium (reduced crowd for speed)."""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from crowdsim.checks.phase1 import Report
from crowdsim.configio import list_stadium_presets
from crowdsim.crowd import load_population, load_scenario
from crowdsim.experiments.runner import run_summary
from crowdsim.models import MODEL_ORDER, make_crowd, prepare, run_model
from crowdsim.models.params import load_model_params
from crowdsim.stadium import load_stadium

OCCUPANCY = 0.20


def run(out_dir: Path, seed: int = 3) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rep = Report()
    P = load_model_params()
    rep.add("setup", "model parameters load and validate", True,
            f"network dt {P.network.dt} s, SF dt {P.sf.dt} s, CA cell {P.micro.nav_cell} m")
    pop = load_population()
    rows = []
    for key in list_stadium_presets():
        st = load_stadium(key)
        for scen in ("emergency", "blocked_exit"):
            sc = replace(load_scenario(scen), occupancy=OCCUPANCY, name=f"{scen}_{int(OCCUPANCY * 100)}pct")
            print(f"\n== {key} / {sc.name}")
            prep = prepare(st, sc, P)
            crowd = make_crowd(prep, pop, seed)
            open_ids = {i for i, e in enumerate(prep.stadium.exits) if e.enabled}
            for m in MODEL_ORDER:
                o = run_model(m, prep, crowd, seed)
                r = o.result
                row = run_summary(o, m, key, sc.name, seed, prep)
                rows.append(row)
                done = np.isfinite(r.exit_time)
                used = set(np.unique(r.exit_id[done]).tolist())
                order_ok = row["T50_s"] <= row["T90_s"] <= row["T100_s"]
                ok = bool(done.all()) and used <= open_ids and order_ok and o.info.get("escaped_numerically", 0) == 0
                rep.add(key, f"{m:8s} {sc.name}: all out via open exits, T50<=T90<=T100", ok,
                        f"{done.sum()}/{r.n} out, T90 {row['T90_s']:.0f} s, T100 {row['T100_s']:.0f} s, "
                        f"{r.compute_time_s:.1f} s compute")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "phase2_smoke.csv", index=False)
    # blocked gates must never make the evacuation faster (same crowd, same model)
    for (key, m), g in df.groupby(["stadium", "model"]):
        t_e = g[g.scenario.str.startswith("emergency")].T90_s.values
        t_b = g[g.scenario.str.startswith("blocked")].T90_s.values
        if len(t_e) and len(t_b):
            rep.add(key, f"{key}: {m} - blocking 2 gates does not speed things up (T90)", t_b[0] >= 0.95 * t_e[0],
                    f"{t_e[0]:.0f} s -> {t_b[0]:.0f} s")
    print(f"\nwrote {out_dir / 'phase2_smoke.csv'}")
    if rep.ok:
        print(f"\nPHASE 2 CHECK: ALL {len(rep.results)} CHECKS PASSED in {time.perf_counter() - t0:.0f} s")
        return 0
    print(f"\nPHASE 2 CHECK: {rep.n_failed} OF {len(rep.results)} CHECKS FAILED")
    return 1
