"""Live simulation window: the plan view updates while the model is computing.

Layout: one plan panel per spatial model (people coloured by speed) and, underneath,
the evacuation curve, people out per gate, and live counters. Keys: SPACE pauses /
resumes, Q closes the window (the simulation keeps running and its results are
still saved). With ``record`` the same frames are written to an MP4 with ffmpeg.

Used by ``python run.py live`` (crowdsim/experiments/live.py) and by the JuPedSim
cross-check (crowdsim/verification/jupedsim_check.py).
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from crowdsim.animation.render import CURVE_COLOURS, V_MAX, _backdrop
from crowdsim.animation.tracer import JAMMED, SEATED, Tracer
from crowdsim.configio import ConfigError
from crowdsim.metrics import density_grid, fruin_los
from crowdsim.models import MODEL_LABELS

GREY = (0.6, 0.6, 0.6, 1.0)


def pick_backend(window: bool) -> str:
    """Select an interactive matplotlib backend (TkAgg, then QtAgg) or Agg when no window is wanted."""
    if not window:
        plt.switch_backend("Agg")
        return "Agg"
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise ConfigError("no display found. Run this from a desktop terminal on your laptop, or add --no-window "
                          "(with --record to still get the video)")
    errors = []
    for b in ("TkAgg", "QtAgg"):
        try:
            plt.switch_backend(b)
            return b
        except Exception as exc:  # noqa: BLE001 - try the next backend
            errors.append(f"{b}: {exc}")
    raise ConfigError("no GUI backend for matplotlib. Install Tk once with:  sudo apt install -y python3-tk\n"
                      "  (or inside the venv: pip install PyQt6)\n  details: " + "; ".join(errors))


class LiveView:
    def __init__(self, stadium, panels: list[str], n_people: int, title: str, frame_dt: float,
                 window: bool = True, record: Path | None = None, marker: float = 4.0, fps: int = 20,
                 labels: dict | None = None):
        self.stadium, self.n, self.frame_dt = stadium, n_people, frame_dt
        self.window, self.paused = window, False
        self.labels = {**MODEL_LABELS, **(labels or {})}
        self.cmap = plt.get_cmap("RdYlBu")
        n_p = max(1, len(panels))
        self.fig = plt.figure(figsize=(15, 8.6), dpi=100)
        try:
            self.fig.canvas.manager.set_window_title("crowdsim live - " + title)
        except Exception:  # noqa: BLE001 - not every backend has a window title
            pass
        gs = self.fig.add_gridspec(2, 3 * n_p, height_ratios=[3.0, 1.15], hspace=0.22, wspace=0.6,
                                   left=0.04, right=0.98, top=0.83, bottom=0.07)
        self.ax, self.scat, self.clock = {}, {}, {}
        for j, m in enumerate(panels):
            ax = self.fig.add_subplot(gs[0, 3 * j:3 * j + 3])
            _backdrop(ax, stadium)
            ax.set_title(self.labels.get(m, m), fontsize=12)
            self.ax[m] = ax
            self.scat[m] = ax.scatter(np.empty(0), np.empty(0), s=marker, color="#888888", linewidths=0, zorder=4)
            self.clock[m] = ax.text(0.02, 0.02, "waiting ...", transform=ax.transAxes, fontsize=10,
                                    family="monospace", bbox=dict(facecolor="white", alpha=0.85, lw=0))
        cax = self.fig.add_axes([0.30, 0.935, 0.40, 0.012])  # labels sit below it, above the panel titles
        cb = self.fig.colorbar(ScalarMappable(norm=Normalize(0.0, V_MAX), cmap=self.cmap), cax=cax,
                               orientation="horizontal")
        cb.set_label("speed [m/s]  (red = stopped / queueing, blue = free walking, grey = still seated)", fontsize=9)
        self.fig.suptitle(title + "      [SPACE pause/resume, Q close window]", fontsize=12, y=0.995)
        third = n_p  # bottom row split into three equal parts
        self.ax_curve = self.fig.add_subplot(gs[1, 0:third])
        self.ax_gate = self.fig.add_subplot(gs[1, third:2 * third])
        self.ax_txt = self.fig.add_subplot(gs[1, 2 * third:3 * n_p])
        self.ax_curve.set_xlabel("time [s]")
        self.ax_curve.set_ylabel("evacuated")
        self.ax_curve.set_ylim(0, 1.02)
        self.ax_curve.set_xlim(0, 300)
        self.ax_curve.grid(alpha=0.3)
        self.curves = {}
        self.gate_ids = [e.id for e in stadium.exits]
        self.gate_open = np.array([e.enabled for e in stadium.exits])
        x = np.arange(len(self.gate_ids))
        self.bars = self.ax_gate.bar(x, np.zeros(len(x)), color=np.where(self.gate_open, "#4c72b0", "#dddddd"))
        self.ax_gate.set_xticks(x)
        self.ax_gate.set_xticklabels(self.gate_ids, rotation=90, fontsize=6)
        self.ax_gate.set_ylabel("people out")
        self.ax_gate.set_title("people out per gate (active model)", fontsize=9)
        self.ax_txt.axis("off")
        self.txt = self.ax_txt.text(0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=9,
                                    transform=self.ax_txt.transAxes)
        pts = stadium.curve.polyline(max([c.d_outer for c in stadium.spec.concourses] +
                                         [t.d_end for t in stadium.tiers]) + 3.0)
        self.dens_bounds = (pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max())
        self.active, self.prev_xy, self.prev_t, self.t_max_seen = None, None, None, 300.0
        self.fig.canvas.mpl_connect("key_press_event", self._key)
        self.fig.canvas.mpl_connect("close_event", self._closed)
        self.writer = None
        self.draw_seconds = 0.0
        if record is not None:
            if shutil.which("ffmpeg") is None:
                raise ConfigError("--record needs ffmpeg:  sudo apt install -y ffmpeg")
            record.parent.mkdir(parents=True, exist_ok=True)
            self.record_path = record.with_suffix(".mp4")
            self.writer = animation.FFMpegWriter(fps=fps, bitrate=5000)
            self.writer.setup(self.fig, str(self.record_path), dpi=100)
        if self.window:
            plt.show(block=False)
            plt.pause(0.05)

    # ------------------------------------------------------------ events
    def _key(self, ev) -> None:
        if ev.key == " ":
            self.paused = not self.paused
            print("paused - press SPACE in the window to resume" if self.paused else "resumed")

    def _closed(self, _ev) -> None:
        if self.window:
            print("window closed - the simulation continues and results will still be saved")
        self.window = False
        self.paused = False

    # ------------------------------------------------------------ model lifecycle
    def start(self, model: str) -> None:
        self.active, self.prev_xy, self.prev_t = model, None, None
        (line,) = self.ax_curve.plot([], [], color=CURVE_COLOURS.get(model, "k"), lw=1.8,
                                     label=self.labels.get(model, model))
        self.curves[model] = line
        self.ax_curve.legend(loc="lower right", fontsize=7)
        if model in self.clock:
            self.ax[model].set_title(self.labels.get(model, model) + "  (running)", fontsize=12, color="#c03030")

    def add_curve(self, model: str, exit_time: np.ndarray) -> None:
        """A model without movement (SFPE): only its evacuation curve."""
        self.start(model)
        self._set_curve(model, exit_time)
        self._draw()

    def finish(self, model: str, exit_time: np.ndarray) -> None:
        self._set_curve(model, exit_time)
        if model in self.ax:
            self.ax[model].set_title(self.labels.get(model, model) + "  (finished)", fontsize=12, color="k")
        self._draw()

    def _set_curve(self, model: str, exit_time: np.ndarray) -> None:
        t = np.sort(exit_time[np.isfinite(exit_time)])
        self.curves[model].set_data(np.r_[0.0, t], np.r_[0.0, np.arange(1, t.size + 1) / self.n])
        if t.size and t[-1] * 1.05 > self.t_max_seen:
            self.t_max_seen = float(t[-1] * 1.05)
            self.ax_curve.set_xlim(0, self.t_max_seen)

    # ------------------------------------------------------------ one frame
    def update(self, t: float, xy: np.ndarray, state: np.ndarray, exit_time: np.ndarray, exit_id: np.ndarray,
               counters: dict, density: float = float("nan"), density_note: str = "") -> None:
        m = self.active
        vis = np.isfinite(xy[:, 0])
        sp = np.zeros(len(xy))
        if self.prev_xy is not None and t > self.prev_t:
            d = np.hypot(*(xy - self.prev_xy).T)
            ok = np.isfinite(d)
            sp[ok] = d[ok] / (t - self.prev_t)
        self.prev_xy, self.prev_t = xy.copy(), t
        rgba = self.cmap(np.clip(sp[vis] / V_MAX, 0, 1))
        rgba[state[vis] == JAMMED] = self.cmap(0.0)  # waiting in a queue (drawn queue slots shift, so no speed)
        rgba[state[vis] == SEATED] = GREY
        sc = self.scat[m]
        sc.set_offsets(xy[vis])
        sc.set_facecolor(rgba)
        done = np.isfinite(exit_time) & (exit_time <= t + 1e-9)
        n_out = int(done.sum())
        self.clock[m].set_text(f"t = {int(t // 60):02d}:{int(t % 60):02d}   out {100 * n_out / self.n:5.1f} %")
        self._set_curve(m, np.where(done, exit_time, np.inf))
        if t * 1.05 > self.t_max_seen:
            self.t_max_seen = t * 1.3
            self.ax_curve.set_xlim(0, self.t_max_seen)
        cnt = np.bincount(exit_id[done & (exit_id >= 0)], minlength=len(self.bars))
        for b, h in zip(self.bars, cnt):
            b.set_height(h)
        self.ax_gate.set_ylim(0, max(10, cnt.max() * 1.15))
        lines = [f"model            {self.labels.get(m, m)[:30]}", f"simulated time   {t:7.0f} s",
                 f"evacuated        {n_out:7d} / {self.n} ({100 * n_out / self.n:.1f} %)"]
        lines += [f"{k:16s} {v:7d}" for k, v in counters.items()]
        if np.isfinite(density):
            lines.append(f"max density      {density:7.2f} p/m2 (LOS {fruin_los([density])[0]})")
            lines.append(f"                 {density_note}")
        self.txt.set_text("\n".join(lines))
        self._draw()

    def _draw(self) -> None:
        t0 = time.perf_counter()
        self._draw_inner()
        self.draw_seconds += time.perf_counter() - t0

    def _draw_inner(self) -> None:
        if self.window:
            self.fig.canvas.draw_idle()
            plt.pause(0.001)
            while self.paused and self.window:
                plt.pause(0.1)
        if self.writer is not None:
            self.writer.grab_frame()

    def close(self, hold: bool) -> None:
        if self.writer is not None:
            self.writer.finish()
            print(f"video: {self.record_path}")
            self.writer = None
        if self.window and hold:
            print("close the window to finish")
            plt.show()
        self.window = False
        plt.close(self.fig)


class LiveTracer(Tracer):
    """Tracer that also drives a LiveView (and optionally keeps frames for trajectories.csv)."""

    def __init__(self, view: LiveView, stadium, net, crowd, frame_dt: float = 2.0, keep: bool = False):
        super().__init__(stadium, net, crowd, frame_dt)
        self.view, self.keep = view, keep

    def snap(self, t: float, eng, micro: dict | None = None) -> None:
        if not self.due(t):
            return
        t0, d0 = time.perf_counter(), self.view.draw_seconds
        self._frame(t, eng, micro)
        self.view.draw_seconds = d0 + (time.perf_counter() - t0)  # everything spent here is display overhead

    def _frame(self, t: float, eng, micro) -> None:
        xy, state = self.positions(t, eng, micro)
        c = eng.counts()
        on_conc = int(sum(mic.n for mic in micro.values())) if micro else 0
        counters = {"still seated": c["waiting_to_move"], "walking rows": c["in_rows"],
                    "walking/aisles": c["walking"], "queueing": c["queued"]}
        if micro:
            counters["on concourse"] = on_conc
            counters["wait for conc."] = c["handed_over"] - on_conc
        if micro:
            pos = [mic.positions() for mic in micro.values() if mic.n]
            dens = 0.0
            if pos:
                g, _, _ = density_grid(np.vstack(pos), 1.2, bounds=self.view.dens_bounds)
                dens = float(g.max())
            note = "exit concourse, 1.2 m cells"
        else:
            dens, note = eng.max_density(), "busiest network element"
        self.view.update(t, xy, state, eng.exit_time, eng.exit_id, counters, dens, note)
        if self.keep:
            self._t.append(float(t))
            self._xy.append(xy.astype(np.float32))
            self._state.append(state)


__all__ = ["LiveTracer", "LiveView", "matplotlib", "pick_backend"]
