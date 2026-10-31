"""Figures for a built stadium: level plans, seat egress-distance map, radial section."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; works without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from crowdsim.stadium.analysis import seat_egress_distance  # noqa: E402
from crowdsim.stadium.model import Stadium  # noqa: E402

TIER_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]


def _band(ax, stadium: Stadium, d0: float, d1: float, colour: str, alpha: float = 0.15) -> None:
    inner = stadium.curve.polyline(d0)
    outer = stadium.curve.polyline(d1)
    poly = np.vstack([outer, inner[::-1]])
    ax.fill(poly[:, 0], poly[:, 1], color=colour, alpha=alpha, lw=0)
    ax.plot(inner[:, 0], inner[:, 1], color=colour, lw=0.6)
    ax.plot(outer[:, 0], outer[:, 1], color=colour, lw=0.6)


def _field(ax, stadium: Stadium) -> None:
    f = stadium.curve.polyline(0.0)
    ax.fill(f[:, 0], f[:, 1], color="#7fbf7f", alpha=0.35, lw=0)
    ax.plot(f[:, 0], f[:, 1], color="#3b7d3b", lw=0.8)


def _finish(ax, title: str) -> None:
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=0.2)


def plot_levels(stadium: Stadium, path: Path) -> Path:
    """One plan view per level: seats, aisles, cross-aisles, vomitories, stairs/ramps, exits."""
    levels = stadium.levels
    ncol = min(3, len(levels))
    nrow = int(np.ceil(len(levels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 5.4 * nrow), squeeze=False)
    seats, curve = stadium.seats, stadium.curve
    tier_ids = [t.id for t in stadium.spec.tiers]
    for ax, lv in zip(axes.flat, levels):
        _field(ax, stadium)
        if lv.kind == "seating":
            ti = tier_ids.index(lv.id)
            tg = stadium.tiers[ti]
            col = TIER_COLOURS[ti % len(TIER_COLOURS)]
            _band(ax, stadium, tg.d_start, tg.d_end, col)
            m = seats.tier == ti
            ax.scatter(seats.x[m], seats.y[m], s=0.15, color=col, rasterized=True)
            for k, s in enumerate(tg.s_aisles):
                p = curve.xy(np.array([s, s]), np.array([tg.d_start, tg.d_end]))
                ax.plot(p[:, 0], p[:, 1], color="k", lw=0.5)
            cross = curve.polyline(tg.d_cross)
            ax.plot(cross[:, 0], cross[:, 1], color="k", lw=0.9, ls="--", label="cross-aisle")
            vp = curve.xy(tg.s_aisles[list(tg.vomitory_aisles)], tg.d_cross)
            ax.scatter(vp[:, 0], vp[:, 1], marker="s", s=22, color="#d62728", zorder=5, label="vomitory")
            if tg.pitch_gate_aisles:
                pp = curve.xy(tg.s_aisles[list(tg.pitch_gate_aisles)], tg.d_start)
                ax.scatter(pp[:, 0], pp[:, 1], marker="^", s=26, color="#17becf", zorder=5,
                           label="pitch gate")
            title = (f"{lv.id} tier - {int(m.sum()):,} seats, z {lv.z_min:.1f}-{lv.z_max:.1f} m")
        else:
            _band(ax, stadium, lv.d_inner, lv.d_outer, "#7f7f7f", alpha=0.25)
            ex = [e for e in stadium.exits if e.level == lv.id and e.kind == "gate"]
            if ex:
                ax.scatter([e.x for e in ex], [e.y for e in ex], marker="v", s=55, color="#d62728",
                           zorder=6, label="exit gate")
                for e in ex:
                    ax.annotate(e.id, (e.x, e.y), fontsize=7, xytext=(3, 3), textcoords="offset points")
            d_mid = 0.5 * (lv.d_inner + lv.d_outer)
            for kind, marker, colour in (("stair", "D", "#9467bd"), ("ramp", "o", "#e377c2")):
                cs = [c for c in stadium.connectors if c.kind == kind and lv.id in (c.from_level, c.to_level)]
                if cs:
                    p = curve.xy(np.array([c.s for c in cs]), d_mid)
                    ax.scatter(p[:, 0], p[:, 1], marker=marker, s=30, color=colour, zorder=5, label=kind)
            vs = [c for c in stadium.connectors if c.kind == "vomitory" and c.to_level == lv.id]
            if vs:
                p = curve.xy(np.array([c.s for c in vs]), d_mid)
                ax.scatter(p[:, 0], p[:, 1], marker="s", s=14, color="#ff9896", zorder=4,
                           label="vomitory landing")
            title = f"{lv.id} concourse - z {lv.z_min:.1f} m"
        ax.legend(loc="upper right", fontsize=7, framealpha=0.8)
        _finish(ax, title)
    for ax in list(axes.flat)[len(levels):]:
        ax.axis("off")
    fig.suptitle(f"{stadium.name}: level plans", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_egress_distance(stadium: Stadium, path: Path) -> Path:
    """Seats coloured by the shortest walking distance (3-D, incl. steps and stairs) to an exit."""
    seats = stadium.seats
    dist = seat_egress_distance(stadium)
    n = len(stadium.tiers)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 5.4), squeeze=False)
    finite = dist[np.isfinite(dist)]
    vmin, vmax = float(finite.min()), float(finite.max())
    sc = None
    for i, ax in enumerate(axes.flat):
        _field(ax, stadium)
        m = seats.tier == i
        sc = ax.scatter(seats.x[m], seats.y[m], c=dist[m], s=0.3, cmap="viridis", vmin=vmin, vmax=vmax,
                        rasterized=True)
        ex = stadium.enabled_exits()
        ax.scatter([e.x for e in ex], [e.y for e in ex], marker="v", s=40, color="#d62728", zorder=5)
        _finish(ax, f"{stadium.spec.tiers[i].id} tier: mean {dist[m].mean():.0f} m, max {dist[m].max():.0f} m")
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.85, label="walking distance to nearest exit [m]")
    fig.suptitle(f"{stadium.name}: egress distance per seat", fontsize=12)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_section(stadium: Stadium, path: Path) -> Path:
    """Radial cross-section (offset d vs height z) showing raked tiers, concourses and stairs."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, (tg, ts) in enumerate(zip(stadium.tiers, stadium.spec.tiers)):
        col = TIER_COLOURS[i % len(TIER_COLOURS)]
        half = ts.row_depth / 2.0
        for d, z in zip(tg.d_rows, tg.z_rows):
            ax.plot([d - half, d + half], [z, z], color=col, lw=2)
        ax.plot([tg.d_cross - tg.cross_width / 2, tg.d_cross + tg.cross_width / 2], [tg.z_cross] * 2,
                color="k", lw=3, solid_capstyle="butt")
        ax.annotate(f"{ts.id} tier", (tg.d_rows[0], tg.z_rows[0]), fontsize=8, xytext=(-4, 8),
                    textcoords="offset points", color=col, ha="right")
    for c in stadium.spec.concourses:
        ax.plot([c.d_inner, c.d_outer], [c.z, c.z], color="#7f7f7f", lw=5, alpha=0.6)
        ax.annotate(f"{c.id} concourse (z={c.z:g} m)", (c.d_inner, c.z), fontsize=8, xytext=(0, -12),
                    textcoords="offset points", color="#555")
        for ci, cn in enumerate(c.connectors):  # schematic: straight line between concourse centres
            to = stadium.spec.concourse(cn.to)
            length = (c.z - to.z) / np.sin(np.radians(cn.angle_deg))
            d0 = 0.5 * (c.d_inner + c.d_outer) + 1.5 * ci
            d1 = 0.5 * (to.d_inner + to.d_outer) + 1.5 * ci
            ax.plot([d0, d1], [c.z, to.z], color="#9467bd" if cn.kind == "stair" else "#e377c2",
                    lw=1.5, ls="-.", label=f"{cn.kind} ({cn.angle_deg:g} deg, walk {length:.0f} m)")
    ax.axhline(0, color="#3b7d3b", lw=0.8)
    ax.set_xlabel("distance from field boundary d [m]")
    ax.set_ylabel("height z [m]")
    ax.set_title(f"{stadium.name}: radial section (rows; cross-aisles black; concourses grey; links schematic)")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_all(stadium: Stadium, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    key = stadium.spec.key
    return [plot_levels(stadium, out_dir / f"{key}_levels.png"),
            plot_egress_distance(stadium, out_dir / f"{key}_egress_distance.png"),
            plot_section(stadium, out_dir / f"{key}_section.png")]
