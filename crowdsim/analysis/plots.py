"""Figures for single runs, model comparisons and parameter sweeps."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from crowdsim.metrics import density_grid, exit_counts  # noqa: E402
from crowdsim.models import MODEL_LABELS, MODEL_ORDER  # noqa: E402

COLOURS = {"sfpe": "#7f7f7f", "ca": "#1f77b4", "sf": "#d62728", "network": "#2ca02c"}


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _curve(ax, times: np.ndarray, n: int, **kw) -> None:
    t = np.sort(times[np.isfinite(times)])
    ax.step(np.r_[0.0, t], np.r_[0.0, np.arange(1, t.size + 1) / n], where="post", **kw)


def _gate_labels(prep) -> list[str]:
    return [e.id for e in prep.stadium.exits]


def _density_map(ax, xy: np.ndarray, prep, title: str, cell: float = 1.2) -> None:
    curve = prep.stadium.curve
    for c in prep.stadium.spec.concourses:
        if any(e.level == c.id and e.kind == "gate" for e in prep.stadium.exits):
            inner, outer = curve.polyline(c.d_inner), curve.polyline(c.d_outer)
            poly = np.vstack([outer, inner[::-1]])
            ax.fill(poly[:, 0], poly[:, 1], color="#eeeeee", lw=0)
            ax.plot(inner[:, 0], inner[:, 1], color="#999", lw=0.6)
            ax.plot(outer[:, 0], outer[:, 1], color="#999", lw=0.6)
    for e in prep.stadium.exits:
        if e.kind == "gate":
            ax.scatter([e.x], [e.y], marker="v", s=40, color="#2ca02c" if e.enabled else "#000000", zorder=5)
    if xy is not None and len(xy):
        dens, xe, ye = density_grid(xy, cell)
        ix = np.clip(np.searchsorted(xe, xy[:, 0], side="right") - 1, 0, dens.shape[0] - 1)
        iy = np.clip(np.searchsorted(ye, xy[:, 1], side="right") - 1, 0, dens.shape[1] - 1)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=dens[ix, iy], s=2, cmap="inferno_r", vmin=0, vmax=6, zorder=4)
        plt.colorbar(sc, ax=ax, shrink=0.8, label="local density [persons/m2]")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


# ---------------------------------------------------------------- one run
def single_run(out_dir: Path, out, row: dict, prep) -> None:
    res, n = out.result, out.result.n
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    ax = axes[0]
    _curve(ax, res.exit_time, n, color=COLOURS.get(row["model"], "k"))
    for q, ls in ((50, ":"), (90, "--")):
        v = row[f"T{q}_s"]
        if np.isfinite(v):
            ax.axvline(v, color="k", ls=ls, lw=0.8, label=f"T{q} = {v:.0f} s")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("fraction evacuated")
    ax.set_title("evacuation curve")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    labels = _gate_labels(prep)
    counts = exit_counts(res.exit_id, len(labels))
    colours = ["#2ca02c" if e.enabled else "#bbbbbb" for e in prep.stadium.exits]
    ax.bar(range(len(labels)), counts, color=colours)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("people")
    ax.set_title("people per exit (grey = closed)")
    ax = axes[2]
    ts = out.timeseries
    for col in ("waiting_to_move", "in_rows", "queued", "walking", "on_concourse", "waiting_to_enter"):
        if col in ts and ts[col].max() > 0:
            ax.plot(ts["t"], ts[col], label=col.replace("_", " "))
    ax.set_xlabel("time [s]")
    ax.set_ylabel("people")
    ax.set_title("where people are")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.suptitle(f"{MODEL_LABELS[row['model']]} | {row['stadium']} | {row['scenario']} | seed {row['seed']}")
    _save(fig, out_dir / "run_overview.png")
    if "max_density" in ts:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.plot(ts["t"], ts["max_density"], color=COLOURS.get(row["model"], "k"))
        for y, lab in ((2.17, "Fruin LOS F"), (5.0, "crush risk")):
            ax.axhline(y, color="#999", ls="--", lw=0.8)
            ax.annotate(lab, (ts["t"].iloc[0], y), fontsize=7, xytext=(2, 2), textcoords="offset points")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("max density [p/m2]")
        ax.set_title(out.density_note, fontsize=9)
        ax.grid(alpha=0.3)
        _save(fig, out_dir / "density_timeseries.png")
    if out.peak_snapshot is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        _density_map(ax, out.peak_snapshot, prep, f"exit concourse at peak density ({out.peak_density:.2f} p/m2, "
                                                   f"t = {out.peak_density_time:.0f} s)")
        _save(fig, out_dir / "density_map_peak.png")


# ---------------------------------------------------------------- comparison
def comparison(vdir: Path, outputs: dict, rows: pd.DataFrame, prep) -> None:
    models = [m for m in MODEL_ORDER if m in outputs and outputs[m]]
    n = outputs[models[0]][0].result.n
    # 1 evacuation curves
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in models:
        for i, o in enumerate(outputs[m]):
            _curve(ax, o.result.exit_time, n, color=COLOURS[m], alpha=1.0 if i == 0 else 0.35, lw=1.4 if i == 0 else 0.8,
                   label=MODEL_LABELS[m] if i == 0 else None)
    ax.set_xlabel("time since alarm [s]")
    ax.set_ylabel("fraction evacuated")
    ax.set_title("evacuation curves (each line = one seed)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, vdir / "evacuation_curves.png")
    # 2 headline times
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.2
    for j, q in enumerate(("T50_s", "T90_s", "T100_s")):
        g = rows.replace([np.inf], np.nan).groupby("model")[q]
        mean, sd = g.mean().reindex(models), g.std().reindex(models).fillna(0)
        ax.bar(np.arange(len(models)) + (j - 1) * w, mean.values, w, yerr=sd.values, capsize=3, label=q[:-2])
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=8)
    ax.set_ylabel("time [s]")
    ax.set_title("evacuation times (mean +/- sd over seeds)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, vdir / "evacuation_times.png")
    # 3 exit usage
    labels = _gate_labels(prep)
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(labels) * len(models) / 2), 4.5))
    w = 0.8 / len(models)
    for j, m in enumerate(models):
        c = np.mean([exit_counts(o.result.exit_id, len(labels)) for o in outputs[m]], axis=0)
        ax.bar(np.arange(len(labels)) + (j - (len(models) - 1) / 2) * w, c, w, color=COLOURS[m], label=MODEL_LABELS[m])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([lab + ("" if e.enabled else "\n(closed)") for lab, e in zip(labels, prep.stadium.exits)],
                       fontsize=7, rotation=90)
    ax.set_ylabel("people (mean over seeds)")
    ax.set_title("exit usage")
    ax.legend(fontsize=8)
    _save(fig, vdir / "exit_usage.png")
    # 4 density over time
    dm = [m for m in models if "max_density" in outputs[m][0].timeseries]
    if dm:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for m in dm:
            ts = outputs[m][0].timeseries
            ax.plot(ts["t"], ts["max_density"], color=COLOURS[m], label=f"{MODEL_LABELS[m]} ({outputs[m][0].density_note})")
        for y in (2.17, 5.0):
            ax.axhline(y, color="#999", ls="--", lw=0.8)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("max density [p/m2]")
        ax.set_title("highest crowd density over time (first seed); dashed: Fruin LOS F, crush risk")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        _save(fig, vdir / "density_over_time.png")
    # 5 peak density maps (micro models)
    snaps = [(m, outputs[m][0]) for m in models if outputs[m][0].peak_snapshot is not None]
    if snaps:
        fig, axes = plt.subplots(1, len(snaps), figsize=(8 * len(snaps), 6), squeeze=False)
        for ax, (m, o) in zip(axes.flat, snaps):
            _density_map(ax, o.peak_snapshot, prep, f"{MODEL_LABELS[m]}: peak {o.peak_density:.2f} p/m2 at "
                                                    f"t = {o.peak_density_time:.0f} s")
        _save(fig, vdir / "density_maps_peak.png")
    # 6 T90 by group (tier / mobility class)
    gcols = [c for c in rows.columns if c.startswith("T90_tier_") or c.startswith("T90_class_")]
    if gcols:
        fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(gcols)), 4.5))
        w = 0.8 / len(models)
        for j, m in enumerate(models):
            vals = rows[rows.model == m][gcols].replace([np.inf], np.nan).mean()
            ax.bar(np.arange(len(gcols)) + (j - (len(models) - 1) / 2) * w, vals.values, w, color=COLOURS[m],
                   label=MODEL_LABELS[m])
        ax.set_xticks(range(len(gcols)))
        ax.set_xticklabels([c.replace("T90_", "").replace("_s", "").replace("_", " ") for c in gcols], fontsize=8,
                           rotation=20)
        ax.set_ylabel("T90 [s]")
        ax.set_title("time for 90 % of each group to get out")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        _save(fig, vdir / "t90_by_group.png")
    # 7 compute cost
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ct = rows.groupby("model").compute_time_s.mean().reindex(models)
    ax.bar([MODEL_LABELS[m] for m in models], ct.values, color=[COLOURS[m] for m in models])
    ax.set_yscale("log")
    ax.set_ylabel("compute time per run [s] (log)")
    ax.set_title("computational cost")
    ax.tick_params(axis="x", labelsize=7)
    _save(fig, vdir / "compute_time.png")


def sweep(root: Path, df: pd.DataFrame, param: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for m in [m for m in MODEL_ORDER if m in set(df.model)]:
        d = df[df.model == m].replace([np.inf], np.nan).groupby("sweep_value")
        for ax, q in zip(axes, ("T90_s", "peak_density_pm2")):
            mean, sd = d[q].mean(), d[q].std().fillna(0)
            if mean.notna().any():
                ax.errorbar(mean.index, mean.values, yerr=sd.values, marker="o", capsize=3, color=COLOURS[m],
                            label=MODEL_LABELS[m])
    axes[0].set_ylabel("T90 [s]")
    axes[1].set_ylabel("peak density [p/m2]")
    for ax in axes:
        ax.set_xlabel(param)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"sensitivity to {param}")
    _save(fig, root / "sweep.png")
