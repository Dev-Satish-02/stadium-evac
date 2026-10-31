"""Phase 3 - Validation suite (step 10).

Hard checks (PASS/FAIL, exit status 1 if any fails):
  V1  free walking: a lone walker keeps the desired speed (all four models)
  V2  bottleneck: specific flow through a door is inside the literature band
  V5  network: an element discharges at its SFPE capacity (no capacity loss)
  V6  SFPE: model output equals an independent hand calculation
  V7  hybrid: everyone in a stadium run is accounted for (conservation)
  V8  reproducibility: same seed -> identical result
Observations (reported, never fail):
  V3  "faster-is-slower" in the social force model
  V4  the four models evacuating the same room

Literature band used in V2: 1.2 - 2.6 persons/(m s). Lower end ~ SFPE design
value (1.32 p/(m s)); upper end covers controlled bottleneck experiments
(about 1.6 - 2.5 p/(m s), e.g. Seyfried et al. 2009; Kretz et al. 2006).
"""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from crowdsim.checks.phase1 import Report  # noqa: E402
from crowdsim.crowd import load_population, load_scenario  # noqa: E402
from crowdsim.metrics import EvacuationResult, exit_specific_flow, percentile_time, summarise  # noqa: E402
from crowdsim.models import make_crowd, prepare, run_model  # noqa: E402
from crowdsim.models import network as network_model  # noqa: E402
from crowdsim.models import sfpe as sfpe_model  # noqa: E402
from crowdsim.models.base import Crowd  # noqa: E402
from crowdsim.models.domain import corridor_domain, room_domain  # noqa: E402
from crowdsim.models.egressnet import EgressNet  # noqa: E402
from crowdsim.models.micro import lattice_cells, run_room  # noqa: E402
from crowdsim.models.params import ModelParams, load_model_params  # noqa: E402
from crowdsim.stadium import load_stadium  # noqa: E402
from crowdsim.stadium.model import SINK  # noqa: E402

FLOW_BAND = (1.2, 2.6)
MODEL_COLOURS = {"sfpe": "#7f7f7f", "ca": "#1f77b4", "sf": "#d62728", "network": "#2ca02c"}


# ---------------------------------------------------------------- small graphs for network / SFPE
def chain_net(params: ModelParams, kinds_lengths_widths, exit_width: float):
    """Straight chain n0 - n1 - ... - nk -> exit. Returns (EgressNet, nodes)."""
    g = nx.Graph()
    g.add_node(SINK, x=np.nan, y=np.nan, z=0.0, level="", kind="sink")
    x = 0.0
    g.add_node("n0", x=0.0, y=0.0, z=0.0, level="L", kind="node")
    for i, (kind, length, width) in enumerate(kinds_lengths_widths):
        x += length
        g.add_node(f"n{i + 1}", x=x, y=0.0, z=0.0, level="L", kind="node")
        g.add_edge(f"n{i}", f"n{i + 1}", length=length, width=width, kind=kind, dz=0.0)
    last = f"n{len(kinds_lengths_widths)}"
    g.add_edge(last, SINK, length=0.0, width=exit_width, kind="sink", dz=0.0)
    nodes = tuple(g.nodes)
    return EgressNet.from_graph(g, nodes, {last: 0}, [exit_width], [True], params.movement), nodes


def simple_crowd(n: int, start: int, speed, lateral=0.0, premovement=0.0, radius=0.23) -> Crowd:
    f = lambda v: np.broadcast_to(np.asarray(v, dtype=float), (n,)).copy()  # noqa: E731
    return Crowd(start_node=np.full(n, start, dtype=int), lateral=f(lateral), speed=f(speed),
                 premovement=f(premovement), radius=f(radius))


# ---------------------------------------------------------------- V1
def v1_free_walk(rep: Report, P: ModelParams, out: Path) -> pd.DataFrame:
    print("\n== V1 free walking (lone walker, 30 m corridor)")
    rows = []
    Pn = P.with_overrides({"movement.row_factor": 1.0})
    dom = corridor_domain(30.0, 3.0, P)
    start = np.array(dom.nav_index(1.0, 1.5)).reshape(1, 2)
    x0, _ = dom.cell_center(start[0, 0], start[0, 1])
    dist = 30.0 - float(x0)
    net, nodes = chain_net(Pn, [("concourse", dist, 3.0)], 3.0)
    for v in (0.8, 1.34, 1.8):
        for kind in ("ca", "sf"):
            t, _, _ = run_room(kind, dom, start, np.array([v]), np.array([0.23]), P, seed=1, t_max=200)
            rows.append({"model": kind, "desired_speed": v, "measured_speed": dist / t[0]})
        for kind, mod in (("network", network_model), ("sfpe", sfpe_model)):
            o = mod.run(net, simple_crowd(1, nodes.index("n0"), v), Pn, seed=1)
            rows.append({"model": kind, "desired_speed": v, "measured_speed": dist / o.result.exit_time[0]})
    df = pd.DataFrame(rows)
    df["error_pct"] = 100 * (df.measured_speed / df.desired_speed - 1)
    for kind, g in df.groupby("model"):
        worst = float(g.error_pct.abs().max())
        rep.add("V1", f"{kind}: lone walker speed within 5 % of desired", worst <= 5.0, f"worst {worst:.1f} %")
    df.to_csv(out / "v1_free_walk.csv", index=False)
    return df


# ---------------------------------------------------------------- V2
def v2_bottleneck(rep: Report, P: ModelParams, out: Path, seeds: int) -> pd.DataFrame:
    print("\n== V2 bottleneck flow vs door width (12 x 12 m room, 180 people)")
    rows = []
    n = 180
    for w in (1.2, 1.6, 2.4, 3.6):
        dom = room_domain(12.0, 12.0, w, P)
        for kind in ("ca", "sf"):
            for s in range(seeds):
                rng = np.random.default_rng(100 + s)
                cells = lattice_cells(dom, n, rng=rng)
                v = np.clip(rng.normal(1.34, 0.26, n), 0.6, 2.2)
                r = np.clip(rng.normal(0.23, 0.015, n), 0.15, 0.35)
                t, ids, _ = run_room(kind, dom, cells, v, r, P, seed=100 + s, t_max=600)
                rows.append({"model": kind, "door_m": w, "seed": s,
                             "specific_flow": float(exit_specific_flow(t, ids, [w])[0]),
                             "evacuated": float(np.isfinite(t).mean())})
        sfpe_js = P.movement.k["level"] / (4 * P.movement.sfpe_a) * (w - 2 * P.movement.boundary_layer["exit"]) / w
        rows.append({"model": "sfpe/network", "door_m": w, "seed": 0, "specific_flow": sfpe_js, "evacuated": 1.0})
    df = pd.DataFrame(rows)
    agg = df.groupby(["model", "door_m"]).specific_flow.mean().unstack(0)
    for kind in ("ca", "sf"):
        vals = agg[kind]
        ok = bool(((vals >= FLOW_BAND[0]) & (vals <= FLOW_BAND[1])).all())
        rep.add("V2", f"{kind}: door specific flow in {FLOW_BAND[0]}-{FLOW_BAND[1]} p/(m s)", ok,
                ", ".join(f"{w:g} m: {v:.2f}" for w, v in vals.items()))
        ev = df[df.model == kind].evacuated.min()
        rep.add("V2", f"{kind}: every bottleneck run fully evacuated", ev == 1.0, f"min evacuated {ev:.2%}")
    df.to_csv(out / "v2_bottleneck.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.axhspan(*FLOW_BAND, color="#ddd", alpha=0.6, label="literature band (used as pass range)")
    for kind, lab in (("ca", "M2 cellular automaton"), ("sf", "M3 social force"), ("sfpe/network", "M1/M4 SFPE capacity")):
        g = df[df.model == kind].groupby("door_m").specific_flow
        m, sd = g.mean(), g.std().fillna(0)
        ax.errorbar(m.index, m.values, yerr=sd.values, marker="o", capsize=3, label=lab,
                    color={"ca": MODEL_COLOURS["ca"], "sf": MODEL_COLOURS["sf"]}.get(kind, "#555"))
    ax.set_xlabel("door width [m]")
    ax.set_ylabel("specific flow [persons/(m s)]")
    ax.set_title("V2 bottleneck flow")
    ax.set_ylim(0, 3.2)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "v2_bottleneck.png", dpi=130)
    plt.close(fig)
    return df


# ---------------------------------------------------------------- V3
def v3_faster_is_slower(rep: Report, P: ModelParams, out: Path, seeds: int) -> pd.DataFrame:
    print("\n== V3 faster-is-slower (social force, 1.0 m door, 180 people) [observation]")
    rows = []
    n = 180
    dom = room_domain(12.0, 12.0, 1.0, P)
    for v0 in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0):
        for s in range(seeds):
            rng = np.random.default_rng(200 + s)
            cells = lattice_cells(dom, n, rng=rng)
            r = np.clip(rng.normal(0.23, 0.015, n), 0.15, 0.35)
            t, _, _ = run_room("sf", dom, cells, np.full(n, v0), r, P, seed=200 + s, t_max=300)
            rows.append({"v0": v0, "seed": s, "T90_s": percentile_time(t, 90),
                         "evacuated": float(np.isfinite(t).mean())})
    df = pd.DataFrame(rows)
    df.to_csv(out / "v3_faster_is_slower.csv", index=False)
    med = df.replace(np.inf, np.nan).groupby("v0").T90_s.median()
    clog = df.groupby("v0").evacuated.apply(lambda e: int((e < 1).sum()))
    best = med.idxmin()
    rise = float(med.loc[med.index > best].max() / med.min() - 1) if (med.index > best).any() else 0.0
    verdict = "clearly reproduced" if rise > 0.15 else "weakly reproduced" if rise > 0.05 else "not reproduced"
    rep.add("V3 (observation)", "faster-is-slower: T90 rises again above an optimum desired speed", True,
            f"{verdict}: fastest median T90 at v0 = {best:g} m/s, rises {100 * rise:.0f} % at higher v0; "
            + "clogged runs per v0: " + ", ".join(f"{k:g}:{v}" for k, v in clog.items()))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    fin = df[np.isfinite(df.T90_s)]
    ax.scatter(fin.v0, fin.T90_s, color="#d62728", alpha=0.5, label="runs")
    ax.plot(med.index, med.values, "k-o", label="median T90")
    for v, c in clog.items():
        if c:
            ax.annotate(f"{c} clogged", (v, med.get(v, np.nan) if np.isfinite(med.get(v, np.nan)) else 0),
                        fontsize=8, xytext=(3, 8), textcoords="offset points")
    ax.set_xlabel("desired speed v0 [m/s]")
    ax.set_ylabel("time for 90 % to leave [s]")
    ax.set_title("V3 social force: faster-is-slower test (1.0 m door)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "v3_faster_is_slower.png", dpi=130)
    plt.close(fig)
    return df


# ---------------------------------------------------------------- V4
def v4_cross_model(rep: Report, P: ModelParams, out: Path, seeds: int) -> pd.DataFrame:
    print("\n== V4 all four models evacuating the same room (12 x 12 m, 1.6 m door, 180 people) [observation]")
    n, w = 180, 1.6
    dom = room_domain(12.0, 12.0, w, P)
    Pn = P.with_overrides({"movement.row_factor": 1.0})
    rows, curves = [], {}
    for s in range(seeds):
        rng = np.random.default_rng(300 + s)
        cells = lattice_cells(dom, n, rng=rng)
        v = np.clip(rng.normal(1.34, 0.26, n), 0.6, 2.2)
        r = np.clip(rng.normal(0.23, 0.015, n), 0.15, 0.35)
        x, y = dom.cell_center(cells[:, 0], cells[:, 1])
        lateral = np.hypot(12.0 - x, 6.0 - y)
        net, nodes = chain_net(Pn, [("exit", 0.3, w)], w)
        crowd = simple_crowd(n, nodes.index("n0"), v, lateral=lateral, radius=r)
        res = {"sfpe": sfpe_model.run(net, crowd, Pn, seed=s).result,
               "network": network_model.run(net, crowd, Pn, seed=s).result}
        for kind in ("ca", "sf"):
            t, ids, _ = run_room(kind, dom, cells, v, r, P, seed=300 + s)
            res[kind] = EvacuationResult(exit_time=t, exit_id=ids, exit_width=[w], model=kind)
        for kind, rr in res.items():
            sm = summarise(rr)
            rows.append({"model": kind, "seed": s, "T50_s": sm["T50_s"], "T90_s": sm["T90_s"],
                         "T100_s": sm["T100_s"], "specific_flow": sm["mean_exit_flow_pms"],
                         "evacuated": sm["evacuated_fraction"]})
            curves.setdefault(kind, []).append(np.sort(rr.exit_time))
    df = pd.DataFrame(rows)
    df.to_csv(out / "v4_cross_model_room.csv", index=False)
    agg = df.groupby("model")[["T50_s", "T90_s", "T100_s", "specific_flow"]].mean()
    rep.add("V4 (observation)", "room evacuation by all four models", bool(df.evacuated.min() == 1.0),
            "; ".join(f"{k}: T90 {r.T90_s:.0f} s" for k, r in agg.iterrows()))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for kind, cs in curves.items():
        c = cs[0]
        ax.plot(c, np.arange(1, n + 1) / n, color=MODEL_COLOURS[kind], label=kind)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("fraction evacuated")
    ax.set_title("V4 same room, four models (first seed)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "v4_cross_model_room.png", dpi=130)
    plt.close(fig)
    return df


# ---------------------------------------------------------------- V5, V6
def v5_network_capacity(rep: Report, P: ModelParams) -> None:
    print("\n== V5 network element discharges at SFPE capacity")
    for kind, nseg, length, width in (("exit", 1, 5.0, 2.0), ("aisle", 24, 0.9, 1.3), ("concourse", 3, 10.0, 6.0),
                                      ("stair", 4, 4.0, 3.0)):
        net, nodes = chain_net(P, [(kind, length, width)] * nseg, width)
        n = 2000  # long stream, so the start-up transient does not bias the measured rate
        o = network_model.run(net, simple_crowd(n, nodes.index("n0"), 1.34, lateral=0.5), P, seed=1)
        # steady discharge between the 10th and 90th percentile passage (start-up and tail excluded)
        rate = float(exit_specific_flow(o.result.exit_time, o.result.exit_id, [1.0])[0])
        cap = net.capacity[0]
        err = abs(rate / cap - 1)
        rep.add("V5", f"{kind} chain discharges at capacity", err <= 0.03,
                f"{rate:.3f} vs capacity {cap:.3f} p/s ({100 * err:.1f} %)")


def v6_sfpe_hand_calc(rep: Report, P: ModelParams) -> None:
    print("\n== V6 SFPE model = hand calculation")
    mp = P.movement
    n, L, Wc, Wd, v = 100, 20.0, 2.0, 1.0, 1.2
    net, nodes = chain_net(P, [("concourse", L, Wc), ("exit", 0.5, Wd)], Wd)
    o = sfpe_model.run(net, simple_crowd(n, nodes.index("n0"), v), P)
    c_corr = mp.k["level"] / (4 * mp.sfpe_a) * (Wc - 2 * mp.boundary_layer["concourse"])
    c_door = mp.k["level"] / (4 * mp.sfpe_a) * (Wd - 2 * mp.boundary_layer["exit"])
    ctrl = c_door if n / c_door >= n / c_corr else c_corr
    # everyone walks L to the door (arrives together), the door serves 1 person every 1/C s, then 0.5 m
    expect_last = L / v + (n - 1) / ctrl + 0.5 / v
    got = float(o.result.exit_time.max())
    rep.add("V6", "SFPE last exit time equals hand calculation", math.isclose(got, expect_last, rel_tol=1e-9),
            f"model {got:.3f} s, hand {expect_last:.3f} s (controlling: {'door' if ctrl == c_door else 'corridor'})")


# ---------------------------------------------------------------- V7, V8
def v7_hybrid_conservation(rep: Report, P: ModelParams) -> None:
    print("\n== V7 hybrid stadium runs account for everyone (indoor arena, 25 % occupancy)")
    st = load_stadium("indoor_arena")
    sc = replace(load_scenario("emergency"), occupancy=0.25, name="emergency_25pct")
    prep = prepare(st, sc, P)
    crowd = make_crowd(prep, load_population(), 7)
    open_ids = {i for i, e in enumerate(prep.stadium.exits) if e.enabled}
    for kind in ("network", "ca", "sf"):
        o = run_model(kind, prep, crowd, 7)
        r = o.result
        done = np.isfinite(r.exit_time)
        ok = bool(done.all()) and set(np.unique(r.exit_id[done]).tolist()) <= open_ids
        esc = o.info.get("escaped_numerically", 0)
        rep.add("V7", f"{kind}: all {crowd.n} people evacuated through open gates", ok and esc == 0,
                f"evacuated {done.sum()} / {crowd.n}, T100 {np.max(r.exit_time[done]):.0f} s, numerical escapes {esc}")


def v8_reproducible(rep: Report, P: ModelParams) -> None:
    print("\n== V8 reproducibility")
    dom = room_domain(8.0, 8.0, 1.6, P)
    rng = np.random.default_rng(5)
    cells = lattice_cells(dom, 60, rng=rng)
    v = np.full(60, 1.34)
    r = np.full(60, 0.23)
    for kind in ("ca", "sf"):
        a, _, _ = run_room(kind, dom, cells, v, r, P, seed=9)
        b, _, _ = run_room(kind, dom, cells, v, r, P, seed=9)
        rep.add("V8", f"{kind}: same seed gives identical exit times", bool(np.array_equal(a, b)))


# ---------------------------------------------------------------- driver
def run(out_dir: Path, quick: bool = False) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    P = load_model_params()
    rep = Report()
    seeds = 1 if quick else 3
    print(f"validation suite ({'quick' if quick else 'full'}), output: {out_dir}")
    v1_free_walk(rep, P, out_dir)
    v2_bottleneck(rep, P, out_dir, seeds)
    v3_faster_is_slower(rep, P, out_dir, 2 if quick else 4)
    v4_cross_model(rep, P, out_dir, seeds)
    v5_network_capacity(rep, P)
    v6_sfpe_hand_calc(rep, P)
    v7_hybrid_conservation(rep, P)
    v8_reproducible(rep, P)
    df = pd.DataFrame([r.__dict__ for r in rep.results])
    df.to_csv(out_dir / "validation_checks.csv", index=False)
    lines = ["# Validation report (Phase 3)", "",
             f"Hard checks passed: {sum(r.passed for r in rep.results if 'observation' not in r.group)} / "
             f"{sum(1 for r in rep.results if 'observation' not in r.group)}", "",
             "| group | check | result | detail |", "|---|---|---|---|"]
    lines += [f"| {r.group} | {r.name} | {'PASS' if r.passed else 'FAIL'} | {r.detail} |" for r in rep.results]
    (out_dir / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    hard = [r for r in rep.results if "observation" not in r.group]
    failed = [r for r in hard if not r.passed]
    print(f"\nwrote {out_dir / 'validation_report.md'} and CSV/PNG files")
    if failed:
        print(f"\nVALIDATION: {len(failed)} OF {len(hard)} HARD CHECKS FAILED")
        return 1
    print(f"\nVALIDATION: ALL {len(hard)} HARD CHECKS PASSED (+ {len(rep.results) - len(hard)} observations)")
    return 0
