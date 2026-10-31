"""Render recorded traces as a replay video (MP4 if ffmpeg is installed, otherwise GIF).

Layout: one plan-view panel per model (people coloured by speed: red = stopped,
blue = free walking; grey = still seated) and an evacuation-curve panel below
with a moving time marker.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm  # noqa: E402,F401
import matplotlib.colors  # noqa: E402,F401
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402

from crowdsim.animation.tracer import SEATED, Trace  # noqa: E402
from crowdsim.models import MODEL_LABELS  # noqa: E402
from crowdsim.stadium.model import Stadium  # noqa: E402
from crowdsim.stadium.plotting import TIER_COLOURS, _band, _field  # noqa: E402

CURVE_COLOURS = {"ca": "#1f77b4", "sf": "#d62728", "network": "#2ca02c", "sfpe": "#7f7f7f",
                 "jps-cfsm": "#9467bd", "jps-sf": "#ff7f0e"}
V_MAX = 1.5  # speed that maps to the "fast" end of the colour scale [m/s]


def _backdrop(ax, st: Stadium) -> None:
    _field(ax, st)
    for i, tg in enumerate(st.tiers):
        _band(ax, st, tg.d_start, tg.d_end, TIER_COLOURS[i % len(TIER_COLOURS)], alpha=0.08)
    for c in st.spec.concourses:
        _band(ax, st, c.d_inner, c.d_outer, "#7f7f7f", alpha=0.12)
    ok = [e for e in st.exits if e.enabled]
    bad = [e for e in st.exits if not e.enabled]
    if ok:
        ax.scatter([e.x for e in ok], [e.y for e in ok], marker="v", s=40, color="#2ca02c", zorder=3)
    if bad:
        ax.scatter([e.x for e in bad], [e.y for e in bad], marker="X", s=45, color="#d62728", zorder=3)
    d_far = max([c.d_outer for c in st.spec.concourses] + [tg.d_end for tg in st.tiers])
    pts = st.curve.polyline(d_far)
    mx, my = 0.04 * np.ptp(pts[:, 0]), 0.04 * np.ptp(pts[:, 1])
    ax.set_xlim(pts[:, 0].min() - mx, pts[:, 0].max() + mx)
    ax.set_ylim(pts[:, 1].min() - my, pts[:, 1].max() + my)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _speeds(trace: Trace) -> np.ndarray:
    """(F, N) speed [m/s] from the displacement between consecutive frames (0 where unknown)."""
    xy = trace.xy.astype(float)
    sp = np.zeros(xy.shape[:2])
    dt = np.diff(trace.t)[:, None]
    d = np.hypot(*(np.diff(xy, axis=0).transpose(2, 0, 1)))
    ok = np.isfinite(d)
    sp[1:][ok] = (d / dt)[ok]
    return sp


def render(traces: dict, stadium: Stadium, out_path: Path, title: str = "", fps: int = 20,
           marker: float = 5.0, size=(16, 9), dpi: int = 100) -> Path:
    """traces: {model_key: Trace}. Writes out_path (.mp4 or .gif depending on ffmpeg). Returns the real path."""
    models = list(traces)
    n_m = len(models)
    use_mp4 = shutil.which("ffmpeg") is not None
    out_path = out_path.with_suffix(".mp4" if use_mp4 else ".gif")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not use_mp4:
        dpi, size = 60, (12, 7)
        print("ffmpeg not found -> writing a GIF instead (install ffmpeg for MP4: sudo apt install ffmpeg)")

    fig = plt.figure(figsize=size, dpi=dpi)
    gs = fig.add_gridspec(2, n_m, height_ratios=[3.2, 1.0], hspace=0.18, wspace=0.04, left=0.04, right=0.98,
                          top=0.83, bottom=0.07)
    t_end = max(float(np.nanmax(tr.t)) for tr in traces.values())
    n = next(iter(traces.values())).xy.shape[1]
    axes, scat, clocks, speeds = {}, {}, {}, {}
    cmap = plt.get_cmap("RdYlBu")
    for j, m in enumerate(models):
        ax = fig.add_subplot(gs[0, j])
        _backdrop(ax, stadium)
        ax.set_title(MODEL_LABELS.get(m, m), fontsize=12)
        sc = ax.scatter(np.empty(0), np.empty(0), s=marker, color="#888888", linewidths=0, zorder=4)
        axes[m], scat[m] = ax, sc
        clocks[m] = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=10, family="monospace",
                            bbox=dict(facecolor="white", alpha=0.8, lw=0))
        speeds[m] = _speeds(traces[m])
    cax = fig.add_axes([0.30, 0.915, 0.40, 0.012])
    mappable = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0.0, V_MAX), cmap=cmap)
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.set_label("speed [m/s]   (red = stopped, blue = free walking; grey = still seated)", fontsize=9)

    axc = fig.add_subplot(gs[1, :])
    for m in models:
        tt = np.sort(traces[m].exit_time[np.isfinite(traces[m].exit_time)])
        axc.step(np.r_[0.0, tt], np.r_[0.0, np.arange(1, tt.size + 1) / n], where="post",
                 color=CURVE_COLOURS.get(m, "k"), label=MODEL_LABELS.get(m, m), lw=1.6)
    axc.set_xlim(0, t_end)
    axc.set_ylim(0, 1.02)
    axc.set_xlabel("time [s]")
    axc.set_ylabel("evacuated")
    axc.grid(alpha=0.3)
    if n_m > 1:
        axc.legend(loc="lower right", fontsize=8)
    vline = axc.axvline(0, color="k", lw=1)
    fig.suptitle(title, fontsize=13, y=0.995)

    F = {m: len(traces[m].t) for m in models}
    n_frames = max(F.values())
    ex = {m: np.sort(traces[m].exit_time[np.isfinite(traces[m].exit_time)]) for m in models}
    writer = (animation.FFMpegWriter(fps=fps, bitrate=4000) if use_mp4 else animation.PillowWriter(fps=min(fps, 12)))
    with writer.saving(fig, str(out_path), dpi):
        for k in range(n_frames):
            for m in models:
                tr = traces[m]
                i = min(k, F[m] - 1)
                xy = tr.xy[i]
                vis = np.isfinite(xy[:, 0])
                col = speeds[m][i][vis]
                seated = tr.state[i][vis] == SEATED
                sc = scat[m]
                sc.set_offsets(xy[vis])
                rgba = cmap(np.clip(col / V_MAX, 0, 1))
                rgba[seated] = (0.6, 0.6, 0.6, 1.0)
                sc.set_facecolor(rgba)
                t = float(tr.t[i])
                out_n = int(np.searchsorted(ex[m], t, side="right"))
                clocks[m].set_text(f"t = {int(t // 60):02d}:{int(t % 60):02d}   out {100 * out_n / n:5.1f} %")
            vline.set_xdata([float(traces[models[0]].t[min(k, F[models[0]] - 1)])] * 2)
            writer.grab_frame()
    plt.close(fig)
    return out_path
