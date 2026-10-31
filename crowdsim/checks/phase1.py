"""End-to-end verification of Phase 0 + Phase 1.

Builds every stadium preset, validates its geometry and navigation graph,
applies every scenario (including sweep variants), populates the crowd,
self-tests the metrics on a known case, then writes figures, CSV tables and
a short markdown report. Exit status 0 only if every check passes.
"""
from __future__ import annotations

import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from crowdsim import DEFAULT_SEED, __version__
from crowdsim.configio import list_scenarios, list_stadium_presets
from crowdsim.crowd import apply_scenario, load_population, load_scenario, populate
from crowdsim.metrics import EvacuationResult, fruin_los, percentile_time, summarise
from crowdsim.stadium import load_stadium, summary, validate
from crowdsim.stadium.plotting import plot_all


@dataclass
class Result:
    group: str
    name: str
    passed: bool
    detail: str = ""


class Report:
    def __init__(self) -> None:
        self.results: list[Result] = []

    def add(self, group: str, name: str, passed: bool, detail: str = "") -> None:
        r = Result(group, name, bool(passed), detail)
        self.results.append(r)
        mark = "PASS" if r.passed else "FAIL"
        print(f"  [{mark}] {name}" + (f"  ({detail})" if detail else ""))

    @property
    def ok(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def n_failed(self) -> int:
        return sum(not r.passed for r in self.results)


def _metrics_selftest(rep: Report) -> None:
    print("\n== metrics self-test (known answers)")
    # 100 people through one 1 m exit at 1 person per second -> flow exactly 1 p/(m s)
    t = np.arange(100, dtype=float)
    res = EvacuationResult(exit_time=t, exit_id=np.zeros(100, int), exit_width=[1.0], model="selftest")
    s = summarise(res)
    rep.add("metrics", "T50 = 49 s, T90 = 89 s, T100 = 99 s",
            (s["T50_s"], s["T90_s"], s["T100_s"]) == (49.0, 89.0, 99.0),
            f"got {s['T50_s']}, {s['T90_s']}, {s['T100_s']}")
    rep.add("metrics", "specific flow = 1.00 p/(m s)", math.isclose(s["mean_exit_flow_pms"], 1.0),
            f"got {s['mean_exit_flow_pms']:.3f}")
    rep.add("metrics", "T90 = inf when only 80 % evacuate",
            math.isinf(percentile_time(np.r_[np.arange(8.0), [np.inf, np.inf]], 90)))
    rep.add("metrics", "Fruin LOS A..F thresholds", "".join(fruin_los([0.2, 0.4, 0.5, 1.0, 2.0, 3.0])) == "ABCDEF")


def run(out_dir: Path, seed: int = DEFAULT_SEED, make_plots: bool = True) -> int:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = Report()
    print(f"crowdsim {__version__} | Python {platform.python_version()} | numpy {np.__version__} | "
          f"pandas {pd.__version__}")
    print(f"output folder: {out_dir}")

    presets = list_stadium_presets()
    scenario_names = list_scenarios()
    rep.add("setup", "stadium presets found", len(presets) >= 3, ", ".join(presets))
    rep.add("setup", "scenario files found", len(scenario_names) >= 6, ", ".join(scenario_names))
    population = load_population()
    rep.add("setup", "population config loads", True, ", ".join(population.names))
    scenarios = [v for n in scenario_names for v in load_scenario(n).expand()]
    rep.add("setup", "scenarios expand (sweeps included)", len(scenarios) > len(scenario_names),
            f"{len(scenarios)} concrete scenarios")

    stadium_rows, crowd_rows = [], []
    for key in presets:
        print(f"\n== stadium: {key}")
        tb = time.perf_counter()
        st = load_stadium(key)
        build_s = time.perf_counter() - tb
        for c in validate(st):
            rep.add(key, c.name, c.passed, c.detail)
        row = summary(st)
        row["build_time_s"] = round(build_s, 2)
        stadium_rows.append(row)
        print(f"  capacity {row['capacity']:,} | {row['levels']} levels | {row['gates']} gates "
              f"({row['exit_width_per_1000_m']} m per 1000) | egress distance mean {row['egress_mean_m']} m, "
              f"max {row['egress_max_m']} m | built in {build_s:.2f} s")
        if make_plots:
            files = plot_all(st, out_dir)
            rep.add(key, "figures written", all(f.is_file() and f.stat().st_size > 10_000 for f in files),
                    ", ".join(f.name for f in files))

        for sc in scenarios:
            sst = apply_scenario(st, sc)
            ag = populate(sst, sc, population, seed)
            again = populate(sst, sc, population, seed)
            want_n = int(round(sc.occupancy * st.capacity))
            shares = population.shares(sc.mix)
            got = np.bincount(ag.cls, minlength=len(shares)) / max(ag.n, 1)
            lo = np.array([c.speed.min for c in population.classes])[ag.cls] * sc.speed_factor
            hi = np.array([c.speed.max for c in population.classes])[ag.cls] * sc.speed_factor
            closed = [e.id for e in sst.gates() if not e.enabled]
            checks = {
                "agent count": ag.n == want_n,
                "class shares within 1.5 %": bool(np.all(np.abs(got - shares) <= 0.015)),
                "speeds within class bounds": bool(np.all((ag.speed >= lo - 1e-9) & (ag.speed <= hi + 1e-9))),
                "every agent can reach an open exit": int(np.sum(~np.isfinite(ag.egress_distance))) == 0,
                "blocked gates closed": closed == [f"G{i}" for i in sc.blocked_exits],
                "same seed -> identical crowd": bool(np.array_equal(ag.seat, again.seat)
                                                     and np.array_equal(ag.speed, again.speed)),
            }
            failed = [k for k, v in checks.items() if not v]
            rep.add(key, f"scenario {sc.name}", not failed,
                    f"{ag.n:,} agents, mean speed {ag.speed.mean():.2f} m/s"
                    + (f"; FAILED: {', '.join(failed)}" if failed else ""))
            crow = {"preset": key, **ag.summary(),
                    "open_exits": len(sst.enabled_exits()),
                    "open_exit_width_m": round(sum(e.width for e in sst.enabled_exits()), 2)}
            crowd_rows.append(crow)

    _metrics_selftest(rep)

    sdf = pd.DataFrame(stadium_rows)
    sdf["seats_per_tier"] = sdf["seats_per_tier"].astype(str)
    cdf = pd.DataFrame(crowd_rows)
    sdf.to_csv(out_dir / "stadium_summary.csv", index=False)
    cdf.to_csv(out_dir / "scenario_summary.csv", index=False)
    rdf = pd.DataFrame([r.__dict__ for r in rep.results])
    rdf.to_csv(out_dir / "phase1_checks.csv", index=False)
    _write_markdown(out_dir / "phase1_report.md", sdf, cdf, rep, seed)

    elapsed = time.perf_counter() - t0
    print(f"\nwrote: {out_dir / 'stadium_summary.csv'}\n       {out_dir / 'scenario_summary.csv'}\n"
          f"       {out_dir / 'phase1_checks.csv'}\n       {out_dir / 'phase1_report.md'}")
    total = len(rep.results)
    if rep.ok:
        print(f"\nPHASE 1 CHECK: ALL {total} CHECKS PASSED in {elapsed:.1f} s")
        return 0
    print(f"\nPHASE 1 CHECK: {rep.n_failed} OF {total} CHECKS FAILED (see lines marked FAIL above)",
          file=sys.stderr)
    return 1


def _write_markdown(path: Path, sdf: pd.DataFrame, cdf: pd.DataFrame, rep: Report, seed: int) -> None:
    cols = ["preset", "capacity", "levels", "gates", "pitch_gates", "exit_width_m", "exit_width_per_1000_m",
            "vomitories", "stairs", "ramps", "max_seat_height_m", "egress_mean_m", "egress_p95_m",
            "egress_max_m", "graph_nodes", "graph_edges"]
    ccols = ["preset", "scenario", "agents", "open_exits", "open_exit_width_m", "speed_mean",
             "premove_median_s", "egress_mean_m", "unreachable"]
    lines = [
        "# Phase 1 verification report", "",
        f"Seed: {seed}. Checks passed: {sum(r.passed for r in rep.results)} / {len(rep.results)}.", "",
        "## Stadium presets", "", _md_table(sdf[cols]), "",
        "Egress distances are 3-D walking distances (row, aisle steps, cross-aisle, vomitory, "
        "concourse, stairs/ramps, gate) to the nearest open exit, from the static shortest path.", "",
        "## Scenario crowds", "", _md_table(cdf[ccols]), "",
    ]
    failed = [r for r in rep.results if not r.passed]
    if failed:
        lines += ["## Failed checks", ""] + [f"- {r.group}: {r.name} ({r.detail})" for r in failed]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])
