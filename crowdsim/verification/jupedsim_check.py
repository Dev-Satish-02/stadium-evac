"""Cross-check with JuPedSim (independent, peer-reviewed pedestrian simulator, https://www.jupedsim.org).

JuPedSim simulates one 2-D floor, so it cannot run a whole multi-level stadium.
It is used here on exactly the parts our CA / social-force models simulate:

J1  room bottleneck  - the 12 x 12 m room of validation V2/V4 (180 people, doors
                       1.2-3.6 m). Same start positions, speeds and body sizes for
                       every model. Compares specific flow and T50/T90/T100 of our
                       CA, SF and SFPE with JuPedSim's collision-free speed model
                       (CFSM, Tordeux et al. 2016 - JuPedSim's default model) and
                       JuPedSim's social force model (Helbing 2000, same parameters
                       as ours).
J2  stadium concourse - the exit concourse(s) of a stadium. Our hybrid SF run is
                       done first; every person's arrival on the concourse (time,
                       place) is recorded and replayed in JuPedSim, each person
                       heading for the gate our floor field sends them to (nearest
                       gate by walking distance). Compares exit curves, T50/T90/T100,
                       time spent on the concourse and use of each gate.

Geometry is converted from our rasterised domains, so every simulator walks on
exactly the same floor. JuPedSim trajectories are saved as .sqlite (JuPedSim's own
format) and plotted with PedPy (JuPedSim's analysis library).

Limit (state it in the report): in J2 the arrivals are fixed from our run (one-way
coupling) - JuPedSim cannot hold people back in the stands, so if its concourse is
slower the queue builds up on the concourse instead.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

import crowdsim.analysis.plots  # noqa: F401  (sets the Agg backend before a GUI backend may be chosen)
import crowdsim.animation.render  # noqa: F401
from crowdsim.configio import ConfigError
from crowdsim.metrics import exit_specific_flow, percentile_time

JPS_MODELS = {"jps-cfsm": "JuPedSim CFSM (collision-free speed)", "jps-sf": "JuPedSim social force"}
OUR_LABELS = {"ca": "ours: M2 cellular automaton", "sf": "ours: M3 social force", "sfpe": "ours: M1 SFPE capacity"}
COLOURS = {"ca": "#1f77b4", "sf": "#d62728", "sfpe": "#7f7f7f", "jps-cfsm": "#9467bd", "jps-sf": "#ff7f0e"}
MAX_PLOTTED = 1500  # people drawn in the PedPy trajectory plots
CHECK_DT = 0.1      # how often agents are spawned / exits detected [s]


def _require_jupedsim():
    try:
        import jupedsim  # noqa: F401
        import shapely  # noqa: F401
    except ImportError as exc:
        raise ConfigError("JuPedSim is not installed. Inside the venv run:  pip install -r requirements-jupedsim.txt"
                          ) from exc


# ---------------------------------------------------------------- geometry
def domain_polygons(dom):
    """(walkable polygon, {exit_id: exit polygon}) from a rasterised Domain, built from cell boxes."""
    from shapely.geometry import box
    from shapely.ops import unary_union

    c = dom.cell
    passable = dom.walk | (dom.exit_id >= 0)

    def boxes(mask):
        out = []
        for ix in range(mask.shape[0]):
            col = mask[ix]
            if not col.any():
                continue
            d = np.diff(np.r_[0, col.astype(np.int8), 0])
            for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
                out.append(box(dom.x0 + ix * c, dom.y0 + a * c, dom.x0 + (ix + 1) * c, dom.y0 + b * c))
        return out

    walk = unary_union(boxes(passable)).buffer(0)
    if walk.geom_type == "MultiPolygon":
        walk = max(walk.geoms, key=lambda g: g.area)
    exits = {}
    for k in np.unique(dom.exit_id[dom.exit_id >= 0]):
        p = unary_union(boxes(dom.exit_id == k)).intersection(walk)
        if p.area > 0:
            exits[int(k)] = p.convex_hull  # JuPedSim needs convex exit areas
    return walk, exits


def nearest_exit_index(dom, exit_ids, xy: np.ndarray) -> np.ndarray:
    """Exit (from exit_ids) with the shortest walking distance from each point (our floor-field rule)."""
    from crowdsim.models.domain import floor_field

    passable = dom.walk | (dom.exit_id >= 0)
    ix, iy = dom.nav_index(xy[:, 0], xy[:, 1])
    dist = np.full((len(exit_ids), len(xy)), np.inf)
    for j, k in enumerate(exit_ids):
        S = floor_field(passable, dom.exit_id == k, dom.cell, walls=~passable)
        dist[j] = S[ix, iy]
    return np.asarray(exit_ids)[np.argmin(dist, axis=0)]


# ---------------------------------------------------------------- one JuPedSim run
def run_jupedsim(kind: str, walk, exits: dict, spawn_t, spawn_xy, spawn_exit, speed, radius, P,
                 dt: float = 0.01, t_max: float = 3600.0, sqlite: Path | None = None, on_frame=None,
                 frame_dt: float = 2.0, spawn_alt: list | None = None,
                 spawn_group: np.ndarray | None = None) -> dict:
    """Spawn person i at spawn_xy[i] at time spawn_t[i], walking to exit spawn_exit[i]. If that spot is
    occupied the other spots in spawn_alt[i] (same arrival point) are tried, otherwise the person waits.
    Returns exit times, the actual spawn times and how the run ended."""
    import jupedsim as jps

    n = len(spawn_t)
    if kind == "jps-cfsm":
        model = jps.CollisionFreeSpeedModel()
    elif kind == "jps-sf":
        model = jps.SocialForceModel(body_force=P.sf.k_body, friction=P.sf.kappa)
    else:
        raise ValueError(kind)
    if sqlite is not None and sqlite.exists():
        sqlite.unlink()
    writer = jps.SqliteTrajectoryWriter(output_file=sqlite, every_nth_frame=max(1, int(round(0.5 / dt)))) \
        if sqlite is not None else None
    sim = jps.Simulation(model=model, geometry=walk, dt=dt, trajectory_writer=writer)
    stage, journey = {}, {}
    for k, poly in exits.items():
        stage[k] = sim.add_exit_stage(poly)
        journey[k] = sim.add_journey(jps.JourneyDescription([stage[k]]))
    r_used = np.asarray(radius, float) + (P.sf.space_margin if kind == "jps-sf" else 0.0)
    order = np.argsort(spawn_t, kind="stable")
    nxt = 0                      # next person (in time order) not yet due
    waiting: list[int] = []      # due but no free spot yet
    id2i, active = {}, set()
    exit_time = np.full(n, np.inf)
    added_at = np.full(n, np.nan)
    chunk = max(1, int(round(CHECK_DT / dt)))
    status, next_frame = "completed", 0.0
    r_max = float(r_used.max())
    t_wall = time.perf_counter()

    n_rejected = 0
    fresh: list = []  # (x, y, r) added in the current pass - not yet in JuPedSim's neighbour index

    def free_spot(i):
        cands = spawn_alt[i] if spawn_alt is not None else spawn_xy[i:i + 1]
        for c in cands:
            pos = (float(c[0]), float(c[1]))
            ok = all(np.hypot(fx - pos[0], fy - pos[1]) >= r_used[i] + fr + 0.02 for fx, fy, fr in fresh)
            if not ok:
                continue
            for aid in sim.agents_in_range(pos, float(r_used[i]) + r_max + 0.02):  # returns agent ids
                j = id2i[aid]
                q = sim.agent(aid).position
                if np.hypot(q[0] - pos[0], q[1] - pos[1]) < r_used[i] + r_used[j] + 0.02:
                    ok = False
                    break
            if ok:
                return pos
        return None

    try:
        while (nxt < n or waiting or sim.agent_count() > 0) and sim.elapsed_time() < t_max:
            now = sim.elapsed_time()
            while nxt < n and spawn_t[order[nxt]] <= now + 1e-9:
                waiting.append(int(order[nxt]))
                nxt += 1
            still = []
            fresh.clear()
            full = set()  # arrival points with no free slot right now (later arrivals there wait too)
            for i in waiting:
                if spawn_group is not None and spawn_group[i] in full:
                    still.append(i)
                    continue
                pos = free_spot(i)
                if pos is None:
                    still.append(i)
                    if spawn_group is not None:
                        full.add(spawn_group[i])
                    continue
                k = int(spawn_exit[i])
                if kind == "jps-cfsm":
                    par = jps.CollisionFreeSpeedModelAgentParameters(
                        journey_id=journey[k], stage_id=stage[k], position=pos, desired_speed=float(speed[i]),
                        radius=float(r_used[i]))
                else:
                    par = jps.SocialForceModelAgentParameters(
                        journey_id=journey[k], stage_id=stage[k], position=pos, desired_speed=float(speed[i]),
                        radius=float(r_used[i]), reaction_time=P.sf.tau, mass=P.sf.mass,
                        agent_scale=P.sf.A, obstacle_scale=P.sf.A, force_distance=P.sf.B)
                try:
                    aid = sim.add_agent(par)
                except RuntimeError:  # JuPedSim rejected the spot (too close to someone): try again later
                    n_rejected += 1
                    still.append(i)
                    continue
                fresh.append((pos[0], pos[1], float(r_used[i])))
                id2i[aid] = i
                active.add(aid)
                added_at[i] = now
            waiting = still
            if nxt >= n and not waiting and sim.agent_count() == 0:
                break
            sim.iterate(chunk)
            present = {a.id for a in sim.agents()}
            t = sim.elapsed_time()
            for aid in active - present:
                exit_time[id2i[aid]] = t
            active = present
            if on_frame is not None and t >= next_frame:
                next_frame = t + frame_dt
                xy = np.full((n, 2), np.nan)
                for a in sim.agents():
                    xy[id2i[a.id]] = a.position
                on_frame(t, xy, exit_time, (n - nxt) + len(waiting))
    except RuntimeError as exc:  # JuPedSim aborts when a person is pushed out of the walkable area
        status = f"aborted at t = {sim.elapsed_time():.1f} s: {exc}"
    if status == "completed" and (nxt < n or waiting or sim.agent_count() > 0):
        status = f"stopped at t_max = {t_max:.0f} s"
    return {"exit_time": exit_time, "added_at": added_at, "status": status, "rejected_spawns": n_rejected,
            "wall_time_s": time.perf_counter() - t_wall}


# ---------------------------------------------------------------- J1 room bottleneck
def j1_room(out: Path, P, seeds: int, models: list[str]) -> pd.DataFrame:
    from crowdsim.models.domain import room_domain
    from crowdsim.models.micro import lattice_cells, run_room

    print("\n== J1 room bottleneck (12 x 12 m, 180 people) - ours vs JuPedSim")
    n, rows, curves = 180, [], {}
    for w in (1.2, 1.6, 2.4, 3.6):
        dom = room_domain(12.0, 12.0, w, P)
        walk, exits = domain_polygons(dom)
        for s in range(seeds):
            rng = np.random.default_rng(100 + s)          # identical crowd generation to validation V2
            cells = lattice_cells(dom, n, rng=rng)
            v = np.clip(rng.normal(1.34, 0.26, n), 0.6, 2.2)
            r = np.clip(rng.normal(0.23, 0.015, n), 0.15, 0.35)
            x, y = dom.cell_center(cells[:, 0], cells[:, 1])
            xy = np.column_stack([x, y])
            res = {}
            for kind in ("ca", "sf"):
                t0 = time.perf_counter()
                t, _, _ = run_room(kind, dom, cells, v, r, P, seed=100 + s, t_max=600)
                res[kind] = (t, "completed" if np.isfinite(t).all() else "incomplete", time.perf_counter() - t0)
            for kind in models:
                sq = out / f"j1_door{w:g}m_{kind}.sqlite" if s == 0 and w == 1.6 else None
                o = run_jupedsim(kind, walk, exits, np.zeros(n), xy, np.zeros(n, int), v, r, P, t_max=600,
                                 sqlite=sq)
                res[kind] = (o["exit_time"], o["status"], o["wall_time_s"])
            for kind, (t, status, wall) in res.items():
                ids = np.where(np.isfinite(t), 0, -1)
                rows.append({"door_m": w, "seed": s, "model": kind, "status": status,
                             "evacuated": float(np.isfinite(t).mean()),
                             "specific_flow": float(exit_specific_flow(t, ids, [w])[0]),
                             "T50_s": percentile_time(t, 50), "T90_s": percentile_time(t, 90),
                             "T100_s": percentile_time(t, 100), "wall_time_s": wall})
                if s == 0:
                    curves[(w, kind)] = (np.sort(t), status)
            print(f"  door {w:g} m seed {s}: " + ", ".join(
                f"{k} {rows[-len(res) + j]['specific_flow']:.2f}" for j, k in enumerate(res)) + "  p/(m s)")
        sfpe = P.movement.k["level"] / (4 * P.movement.sfpe_a) * (w - 2 * P.movement.boundary_layer["exit"]) / w
        rows.append({"door_m": w, "seed": 0, "model": "sfpe", "status": "analytic", "evacuated": 1.0,
                     "specific_flow": sfpe, "T50_s": np.nan, "T90_s": np.nan, "T100_s": np.nan, "wall_time_s": 0})
    df = pd.DataFrame(rows)
    df.to_csv(out / "j1_room_bottleneck.csv", index=False)
    _plot_j1(out, df, curves)
    return df


def _plot_j1(out: Path, df: pd.DataFrame, curves: dict) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axes[0]
    ax.axhspan(1.2, 2.6, color="#ddd", alpha=0.6, label="experimental range (validation band)")
    for kind in ["sfpe", "ca", "sf", *JPS_MODELS]:
        g = df[(df.model == kind) & (df.evacuated == 1.0)].groupby("door_m").specific_flow
        if g.ngroups == 0:
            continue
        m, sd = g.mean(), g.std().fillna(0)
        ax.errorbar(m.index, m.values, yerr=sd.values, marker="o", capsize=3, color=COLOURS[kind],
                    label=OUR_LABELS.get(kind, JPS_MODELS.get(kind, kind)))
    ax.set_xlabel("door width [m]")
    ax.set_ylabel("specific flow [persons/(m s)]")
    ax.set_title("J1 bottleneck flow (completed runs only)")
    ax.set_ylim(0, 3.2)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax = axes[1]
    for (w, kind), (t, status) in curves.items():
        if w != 1.6:
            continue
        tt = t[np.isfinite(t)]
        lab = OUR_LABELS.get(kind, JPS_MODELS.get(kind, kind))
        if status.startswith("aborted"):
            lab += " (aborted: " + status.split(":")[0].replace("aborted at ", "") + ")"
        ax.step(np.r_[0, tt], np.r_[0, np.arange(1, tt.size + 1) / t.size], where="post", color=COLOURS[kind],
                label=lab)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("fraction evacuated")
    ax.set_title("J1 same room, 1.6 m door, first crowd")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "j1_room_bottleneck.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- J2 stadium concourse
def j2_concourse(out: Path, P, stadium_key: str, scenario: str, seed: int, crowd_fraction: float,
                 models: list[str], live: bool, record: bool = False) -> pd.DataFrame:
    from dataclasses import replace

    from crowdsim.crowd import load_population, load_scenario
    from crowdsim.models import make_crowd, prepare
    from crowdsim.models.hybrid import run_hybrid
    from crowdsim.stadium import load_stadium

    st = load_stadium(stadium_key)
    sc = load_scenario(scenario).expand()[0]
    if crowd_fraction < 1.0:
        sc = replace(sc, occupancy=sc.occupancy * crowd_fraction)
    prep = prepare(st, sc, P)
    crowd = make_crowd(prep, load_population(), seed)
    doms, _ = prep.domains()
    if len(doms) != 1:
        raise ConfigError(f"{stadium_key}: J2 needs gates on one concourse level (found {list(doms)})")
    (level, dom), = doms.items()
    print(f"\n== J2 {st.name} / {sc.name}: exit concourse '{level}', {crowd.n:,} people, seed {seed}")
    log: list = []
    t0 = time.perf_counter()
    ours = run_hybrid("sf", prep.stadium, prep.net, crowd, P, seed=seed, domains=prep.domains(), inject_log=log)
    print(f"  ours (hybrid SF): T90 {percentile_time(ours.result.exit_time, 90):.0f} s "
          f"[{time.perf_counter() - t0:.0f} s]")
    gid = np.concatenate([g for g, _, _ in log])
    t_in = np.concatenate([np.full(g.size, t) for g, t, _ in log])
    xy_in = np.vstack([xy for _, _, xy in log])
    walk, exits = domain_polygons(dom)
    from shapely.geometry import Point

    # JuPedSim needs every spawn point at least one body radius away from walls
    inner = walk.buffer(-(float(crowd.radius.max()) + P.sf.space_margin + 0.05))
    from shapely.ops import nearest_points

    for i in range(len(xy_in)):
        p = Point(xy_in[i])
        if not inner.contains(p):
            q = nearest_points(inner, p)[0]
            xy_in[i] = (q.x, q.y)
    # every arrival point (vomitory / stair landing) has several spawn slots across its width; JuPedSim may
    # use any free slot of the person's own arrival point (as our models do)
    from scipy.spatial import cKDTree

    slot_xy = np.vstack(list(dom.inject.values()))
    slot_node = np.concatenate([np.full(len(v), k) for k, v in dom.inject.items()])
    for i in range(len(slot_xy)):
        p = Point(slot_xy[i])
        if not inner.contains(p):
            q = nearest_points(inner, p)[0]
            slot_xy[i] = (q.x, q.y)
    _, near = cKDTree(slot_xy).query(xy_in)
    by_node = {k: slot_xy[slot_node == k] for k in dom.inject}
    alt = []
    for i in range(len(xy_in)):
        c = by_node[slot_node[near[i]]]
        alt.append(np.vstack([xy_in[i], c[np.argsort(np.hypot(*(c - xy_in[i]).T))]]))
    exit_ids = sorted(exits)
    target = nearest_exit_index(dom, exit_ids, xy_in)
    base = {"gid": gid, "t_in_s": t_in, "x_in": xy_in[:, 0], "y_in": xy_in[:, 1], "target_gate": target,
            "sf_exit_s": ours.result.exit_time[gid], "sf_gate": ours.result.exit_id[gid]}
    summary = [_j2_row("sf", t_in, ours.result.exit_time[gid], "completed", float("nan"))]
    view = None
    for kind in models:
        cb = None
        if live or record:
            from crowdsim.animation.live import LiveView, pick_backend

            pick_backend(live)
            view = LiveView(prep.stadium, [kind], gid.size, f"{JPS_MODELS[kind]} on the "
                            f"{st.name} exit concourse - {gid.size:,} arrivals replayed", 2.0,
                            window=live, record=(out / f"j2_{stadium_key}_{kind}_live") if record else None,
                            labels=JPS_MODELS)
            view.start(kind)
            ids_all = np.full(gid.size, -1)

            def cb(t, xy, et, n_wait, _v=view, _ids=ids_all, _gates=target):
                ids = np.where(np.isfinite(et), _gates, -1)
                _v.update(t, xy, np.ones(len(xy), np.uint8), et, ids,
                          {"on concourse": int(np.isfinite(xy[:, 0]).sum()), "not yet arrived": int(n_wait)})
        o = run_jupedsim(kind, walk, exits, t_in, xy_in, target, crowd.speed[gid], crowd.radius[gid], P,
                         sqlite=out / f"j2_{stadium_key}_{kind}.sqlite", on_frame=cb, spawn_alt=alt,
                         spawn_group=slot_node[near])
        if view is not None:
            view.finish(kind, o["exit_time"])
            view.close(hold=False)
            view = None
        base[f"{kind}_exit_s"] = o["exit_time"]
        base[f"{kind}_spawn_delay_s"] = o["added_at"] - t_in
        summary.append(_j2_row(kind, o["added_at"], o["exit_time"], o["status"], o["wall_time_s"],
                               np.nanmean(o["added_at"] - t_in)))
        print(f"  {kind:9s} {o['status']:10.10s} T90 {summary[-1]['T90_s']:.0f} s, "
              f"concourse time {summary[-1]['mean_concourse_time_s']:.1f} s [{o['wall_time_s']:.0f} s]")
    df = pd.DataFrame(base)
    df.to_csv(out / f"j2_{stadium_key}_concourse_agents.csv", index=False)
    sm = pd.DataFrame(summary)
    sm.insert(0, "stadium", stadium_key)
    sm.to_csv(out / f"j2_{stadium_key}_summary.csv", index=False)
    _plot_j2(out, stadium_key, st, df, models, exit_ids)
    return sm


def _j2_row(kind, t_in, t_out, status, wall, delay=0.0) -> dict:
    """t_in = when the person actually appeared on the concourse (JuPedSim: after any spawn delay)."""
    ok = np.isfinite(t_out) & np.isfinite(t_in)
    ct = (t_out - t_in)[ok]
    return {"model": kind, "status": status, "evacuated": float(ok.mean()), "T50_s": percentile_time(t_out, 50),
            "T90_s": percentile_time(t_out, 90), "T100_s": percentile_time(t_out, 100),
            "mean_concourse_time_s": float(ct.mean()) if ct.size else np.nan,
            "p90_concourse_time_s": float(np.percentile(ct, 90)) if ct.size else np.nan,
            "mean_spawn_delay_s": float(delay) if np.isfinite(delay) else np.nan, "wall_time_s": wall}


def _plot_j2(out: Path, key: str, st, df: pd.DataFrame, models: list[str], exit_ids) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    n = len(df)
    for kind in ["sf", *models]:
        t = np.sort(df[f"{kind}_exit_s"].values)
        tt = t[np.isfinite(t)]
        axes[0].step(np.r_[0, tt], np.r_[0, np.arange(1, tt.size + 1) / n], where="post", color=COLOURS[kind],
                     label=OUR_LABELS.get(kind, JPS_MODELS.get(kind, kind)))
        t_start = df.t_in_s + (df[f"{kind}_spawn_delay_s"].fillna(0) if f"{kind}_spawn_delay_s" in df else 0.0)
        ct = (df[f"{kind}_exit_s"] - t_start).values
        ct = ct[np.isfinite(ct)]
        axes[1].hist(ct, bins=60, range=(0, max(60, np.percentile(ct, 99) if ct.size else 60)), histtype="step",
                     lw=1.5, color=COLOURS[kind], label=OUR_LABELS.get(kind, JPS_MODELS.get(kind, kind)))
    tt = np.sort(df.t_in_s.values)
    axes[0].step(np.r_[0, tt], np.r_[0, np.arange(1, tt.size + 1) / n], where="post", color="k", ls=":",
                 label="arrivals on the concourse (same for all)")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("fraction")
    axes[0].set_title("J2 people out of the gates")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("time from arriving on the concourse to leaving by a gate [s]")
    axes[1].set_ylabel("people")
    axes[1].set_title("J2 time spent on the concourse")
    axes[1].legend(fontsize=7)
    names = [st.exits[k].id for k in exit_ids]
    x = np.arange(len(exit_ids))
    width = 0.8 / (1 + len(models))
    for j, kind in enumerate(["sf", *models]):
        done = np.isfinite(df[f"{kind}_exit_s"].values)
        gates = df.sf_gate.values if kind == "sf" else df.target_gate.values
        cnt = [int(np.sum(done & (gates == k))) for k in exit_ids]
        axes[2].bar(x + j * width, cnt, width, color=COLOURS[kind], label=OUR_LABELS.get(kind, JPS_MODELS.get(kind, kind)))
    axes[2].set_xticks(x + 0.4 - width / 2)
    axes[2].set_xticklabels(names, rotation=90, fontsize=7)
    axes[2].set_ylabel("people out")
    axes[2].set_title("J2 use of each gate\n(gate = nearest by walking distance, as in our floor field)", fontsize=10)
    axes[2].set_ylim(0, axes[2].get_ylim()[1] * 1.35)
    axes[2].legend(fontsize=7, loc="upper right", ncol=1)
    fig.suptitle(f"{st.name}: our social-force concourse vs JuPedSim on the same arrivals", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"j2_{key}_concourse.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- PedPy plots of the JuPedSim trajectories
def pedpy_plots(out: Path) -> list[Path]:
    """Plot every JuPedSim .sqlite with PedPy (JuPedSim's own analysis library), if it is installed."""
    try:
        import matplotlib.pyplot as plt
        import pedpy
    except ImportError:
        print("  (pedpy not installed - skipping trajectory plots)")
        return []
    made = []
    for sq in sorted(out.glob("*.sqlite")):
        try:
            traj = pedpy.load_trajectory_from_jupedsim_sqlite(trajectory_file=sq)
            area = pedpy.load_walkable_area_from_jupedsim_sqlite(trajectory_file=sq)
            ids = traj.data["id"].unique()
            if ids.size > MAX_PLOTTED:  # drawing tens of thousands of paths takes very long: plot a sample
                keep = np.random.default_rng(0).choice(ids, MAX_PLOTTED, replace=False)
                traj = pedpy.TrajectoryData(data=traj.data[traj.data["id"].isin(keep)].copy(),
                                            frame_rate=traj.frame_rate)
            fig, ax = plt.subplots(figsize=(8, 8))
            pedpy.plot_trajectories(traj=traj, walkable_area=area, axes=ax, traj_alpha=0.3, traj_width=0.3)
            n_shown = traj.data["id"].nunique()
            ax.set_title(f"PedPy: JuPedSim trajectories - {sq.stem}"
                         + (f" ({n_shown:,} of {ids.size:,} people)" if n_shown < ids.size else ""))
            p = sq.with_name(sq.stem + "_pedpy_trajectories.png")
            fig.savefig(p, dpi=130)
            plt.close(fig)
            made.append(p)
        except Exception as exc:  # noqa: BLE001 - a plot failure must not lose the numbers
            print(f"  pedpy could not plot {sq.name}: {exc}")
    return made


# ---------------------------------------------------------------- entry point
def run(out: Path, stadiums: list[str], scenario: str = "emergency", seed: int = 1, quick: bool = False,
        live: bool = False, crowd_fraction: float = 1.0, skip_room: bool = False, jps_sf: bool = True,
        record: bool = False) -> int:
    _require_jupedsim()
    import jupedsim

    from crowdsim.models.params import load_model_params

    out.mkdir(parents=True, exist_ok=True)
    P = load_model_params()
    models = ["jps-cfsm"] + (["jps-sf"] if jps_sf else [])
    print(f"JuPedSim {jupedsim.__version__}; models {', '.join(models)}")
    lines = [f"# JuPedSim cross-check (JuPedSim {jupedsim.__version__})", ""]
    if not skip_room:
        df = j1_room(out, P, 1 if quick else 3, models)
        agg = df.groupby(["door_m", "model"]).agg(flow=("specific_flow", "mean"), T90=("T90_s", "mean"),
                                                  evacuated=("evacuated", "min")).reset_index()
        lines += ["## J1 room bottleneck (180 people, 12 x 12 m room)", "",
                  "| door [m] | model | specific flow [p/(m s)] | T90 [s] | evacuated (min) |", "|---|---|---|---|---|"]
        for _, r in agg.iterrows():
            lines.append(f"| {r.door_m:g} | {r.model} | {r.flow:.2f} | {r.T90:.1f} | {r.evacuated:.2f} |")
        bad = df[df.status.str.startswith("aborted")]
        if len(bad):
            lines += ["", f"JuPedSim runs that aborted: {len(bad)} (" + "; ".join(
                sorted({f'{m} door {d:g} m' for m, d in zip(bad.model, bad.door_m)})) + ")",
                "JuPedSim stops a run when a person is pushed outside the walkable area (crowd pressure).",
                "Their flow is left out of the plot."]
        lines.append("")
    for key in stadiums:
        sm = j2_concourse(out, P, key, scenario, seed, 0.3 if quick and crowd_fraction == 1.0 else crowd_fraction,
                          models, live, record)
        lines += [f"## J2 {key} exit concourse", "", "| model | status | evacuated | T50 [s] | T90 [s] | T100 [s] | "
                  "mean time on concourse [s] | mean spawn delay [s] |", "|---|---|---|---|---|---|---|---|"]
        for _, r in sm.iterrows():
            lines.append(f"| {r.model} | {r.status[:40]} | {r.evacuated:.3f} | {r.T50_s:.0f} | {r.T90_s:.0f} | "
                         f"{r.T100_s:.0f} | {r.mean_concourse_time_s:.1f} | {r.mean_spawn_delay_s:.1f} |")
        lines.append("")
    made = pedpy_plots(out)
    lines += ["Figures: j1_room_bottleneck.png, j2_<stadium>_concourse.png" +
              (", *_pedpy_trajectories.png (PedPy)" if made else ""),
              "JuPedSim trajectories: *.sqlite (open with PedPy or the JuPedSim visualizer)."]
    (out / "jupedsim_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nresults folder: {out}")
    return 0
