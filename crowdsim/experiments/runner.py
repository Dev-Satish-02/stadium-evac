"""Running models on stadium scenarios and writing results (simulate / compare).

Fairness rules used everywhere:
* every model gets the SAME crowd for a given seed (same seats, speeds,
  body sizes, pre-movement times) - common random numbers;
* every model is scored by the same metric functions (crowdsim.metrics).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from crowdsim.crowd import load_population, load_scenario
from crowdsim.metrics import CRUSH_DENSITY, fruin_los, percentile_time, summarise
from crowdsim.models import MODEL_LABELS, MODEL_ORDER, make_crowd, prepare, run_model
from crowdsim.models.base import RunOutput
from crowdsim.models.params import load_model_params
from crowdsim.paths import RESULTS_DIR
from crowdsim.stadium import load_stadium


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else round(float(v), 4)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def run_summary(out: RunOutput, model: str, stadium_key: str, scenario: str, seed: int, prep) -> dict:
    """One results-table row: shared metrics + density + breakdown by tier and mobility class."""
    res = out.result
    open_mask = np.array([e.enabled for e in prep.stadium.exits], dtype=bool)
    s = summarise(res, open_exits=open_mask)
    row = {"model": model, "model_label": MODEL_LABELS[model], "stadium": stadium_key, "scenario": scenario,
           "seed": seed}
    row.update({k: v for k, v in s.items() if k not in ("model", "scenario", "seed")})
    pk = out.peak_density
    row["peak_density_pm2"] = pk
    row["peak_density_time_s"] = out.peak_density_time
    row["peak_los"] = str(fruin_los([pk])[0]) if np.isfinite(pk) else "n/a"
    row["crush_risk"] = bool(np.isfinite(pk) and pk >= CRUSH_DENSITY)
    ts = out.timeseries
    if "max_density" in ts and len(ts) > 1:
        dt = float(np.median(np.diff(ts["t"])))
        row["time_density_ge_4_s"] = float((ts["max_density"] >= 4.0).sum() * dt)
    else:
        row["time_density_ge_4_s"] = float("nan")
    labels = out.crowd.labels
    tier_names = [t.id for t in prep.stadium.spec.tiers]
    for i, name in enumerate(tier_names):
        m = labels["tier"] == i
        row[f"T90_tier_{name}_s"] = percentile_time(res.exit_time[m], 90) if m.any() else float("nan")
    for c in np.unique(labels["cls"]):
        m = labels["cls"] == c
        row[f"T90_class_{c}_s"] = percentile_time(res.exit_time[m], 90)
    for k, v in out.info.items():
        if isinstance(v, (int, float, str)):
            row[f"info_{k}"] = v
    return row


def save_run(out_dir: Path, out: RunOutput, row: dict, prep, make_plots: bool = True) -> None:
    from crowdsim.analysis import plots

    out_dir.mkdir(parents=True, exist_ok=True)
    out.agent_table().to_csv(out_dir / "agents.csv", index=False)
    out.timeseries.to_csv(out_dir / "timeseries.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(_clean({**row, "info": out.info,
                                                             "density_note": out.density_note}), indent=2))
    if make_plots:
        plots.single_run(out_dir, out, row, prep)


def simulate(stadium_key: str, scenario_name: str, model: str, seed: int, overrides: dict | None = None,
             out_root: Path | None = None, make_plots: bool = True) -> list[dict]:
    """Run one model (all variants of a sweep scenario) and write results."""
    if model not in MODEL_ORDER:
        raise ValueError(f"unknown model '{model}'. Choose from {', '.join(MODEL_ORDER)}")
    params = load_model_params(overrides)
    stadium = load_stadium(stadium_key)
    population = load_population()
    root = (out_root or RESULTS_DIR / "runs") / stadium_key
    rows = []
    for sc in load_scenario(scenario_name).expand():
        prep = prepare(stadium, sc, params)
        crowd = make_crowd(prep, population, seed)
        t0 = time.perf_counter()
        out = run_model(model, prep, crowd, seed)
        row = run_summary(out, model, stadium_key, sc.name, seed, prep)
        row["wall_time_s"] = time.perf_counter() - t0
        d = root / _safe(sc.name) / model / f"seed{seed}"
        save_run(d, out, row, prep, make_plots)
        _print_row(row)
        print(f"   results: {d}")
        rows.append(row)
    return rows


def compare(stadium_key: str, scenario_name: str, models: list[str], seeds: int, overrides: dict | None = None,
            out_root: Path | None = None, first_seed: int = 1, make_plots: bool = True) -> Path:
    """Run several models x seeds on the same crowds; write tables and comparison figures."""
    from crowdsim.analysis import plots

    params = load_model_params(overrides)
    stadium = load_stadium(stadium_key)
    population = load_population()
    base = load_scenario(scenario_name)
    root = (out_root or RESULTS_DIR / "compare") / stadium_key / _safe(scenario_name)
    root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for sc in base.expand():
        prep = prepare(stadium, sc, params)
        vdir = root / _safe(sc.name) if base.sweep_param else root
        outputs = {m: [] for m in models}
        for k in range(seeds):
            seed = first_seed + k
            crowd = make_crowd(prep, population, seed)
            for m in models:
                t0 = time.perf_counter()
                out = run_model(m, prep, crowd, seed)
                row = run_summary(out, m, stadium_key, sc.name, seed, prep)
                row["wall_time_s"] = time.perf_counter() - t0
                if base.sweep_param:
                    row["sweep_param"] = base.sweep_param
                    row["sweep_value"] = getattr(sc, base.sweep_param)
                _print_row(row)
                all_rows.append(row)
                outputs[m].append(out)
                save_run(vdir / "runs" / m / f"seed{seed}", out, row, prep, make_plots=False)
        rows_v = pd.DataFrame([r for r in all_rows if r["scenario"] == sc.name])
        _write_tables(vdir, rows_v)
        if make_plots:
            plots.comparison(vdir, outputs, rows_v, prep)
    df = pd.DataFrame(all_rows)
    df.to_csv(root / "all_runs.csv", index=False)
    if base.sweep_param:
        agg = _aggregate(df, ["sweep_value", "model"])
        agg.to_csv(root / "sweep_summary.csv", index=False)
        if make_plots:
            plots.sweep(root, df, base.sweep_param)
    print(f"\nresults folder: {root}")
    return root


# ---------------------------------------------------------------- helpers
HEADLINE = ["T50_s", "T90_s", "T95_s", "T100_s", "evacuated_fraction", "mean_exit_flow_pms", "load_cv",
            "clearance_spread_s", "peak_density_pm2", "time_density_ge_4_s", "compute_time_s"]


def _aggregate(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    cols = [c for c in df.columns if c in HEADLINE or c.startswith("T90_tier_") or c.startswith("T90_class_")]
    num = df[keys + cols].replace([np.inf, -np.inf], np.nan)
    g = num.groupby(keys)
    mean = g.mean().add_suffix("_mean")
    sd = g.std().add_suffix("_sd")
    out = pd.concat([mean, sd], axis=1).reset_index()
    out.insert(len(keys), "runs", g.size().values)
    return out


def _write_tables(vdir: Path, rows: pd.DataFrame) -> None:
    vdir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(vdir / "runs.csv", index=False)
    agg = _aggregate(rows, ["model"])
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    agg = agg.sort_values("model", key=lambda s: s.map(order))
    agg.to_csv(vdir / "summary.csv", index=False)
    lines = [f"# {rows['stadium'].iloc[0]} - {rows['scenario'].iloc[0]}", "",
             f"{rows['seed'].nunique()} seed(s) per model; mean +/- sd. Same crowd for every model.", "",
             "| model | T50 [s] | T90 [s] | T100 [s] | evacuated | mean gate flow [p/(m s)] | exit load CV | "
             "peak density [p/m2] | compute [s] |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in agg.iterrows():
        f = lambda c, d=0: _pm(r.get(f"{c}_mean"), r.get(f"{c}_sd"), d)  # noqa: E731
        lines.append(f"| {MODEL_LABELS[r['model']]} | {f('T50_s')} | {f('T90_s')} | {f('T100_s')} | "
                     f"{f('evacuated_fraction', 3)} | {f('mean_exit_flow_pms', 2)} | {f('load_cv', 2)} | "
                     f"{f('peak_density_pm2', 2)} | {f('compute_time_s', 1)} |")
    (vdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pm(m, s, d) -> str:
    if m is None or not np.isfinite(m):
        return "n/a"
    if s is None or not np.isfinite(s):
        return f"{m:.{d}f}"
    return f"{m:.{d}f} +/- {s:.{d}f}"


def _safe(name: str) -> str:
    return name.replace("@", "__").replace("=", "-").replace("/", "_")


def _print_row(r: dict) -> None:
    def f(v, d=0):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "n/a"
        return "inf" if not np.isfinite(v) else f"{v:.{d}f}"
    print(f"  {r['model']:8s} seed {r['seed']:<3d} {r['scenario']:38s} T50 {f(r['T50_s']):>5s}  T90 {f(r['T90_s']):>5s}  "
          f"T100 {f(r['T100_s']):>5s} s  out {100 * r['evacuated_fraction']:5.1f} %  peak {f(r['peak_density_pm2'], 2):>5s} p/m2  "
          f"[{r['wall_time_s']:.1f} s]")
