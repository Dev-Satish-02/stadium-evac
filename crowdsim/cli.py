"""Command-line interface (``python run.py <command>``)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from crowdsim import DEFAULT_SEED, __version__
from crowdsim.configio import ConfigError, list_scenarios, list_stadium_presets
from crowdsim.paths import RESULTS_DIR

MODELS = ("sfpe", "ca", "sf", "network")


def _cmd_list(_args) -> int:
    print("stadium presets:", ", ".join(list_stadium_presets()))
    print("scenarios      :", ", ".join(list_scenarios()))
    print("models         : sfpe (M1 hydraulic), ca (M2 cellular automaton), sf (M3 social force), "
          "network (M4 queueing network)")
    return 0


def _cmd_stadium(args) -> int:
    from crowdsim.stadium import load_stadium, summary, validate
    from crowdsim.stadium.plotting import plot_all

    st = load_stadium(args.preset)
    print(json.dumps(summary(st), indent=2))
    checks = validate(st)
    for c in checks:
        print(f"[{'PASS' if c.passed else 'FAIL'}] {c.name}" + (f"  ({c.detail})" if c.detail else ""))
    if not args.no_plots:
        for f in plot_all(st, Path(args.out)):
            print("figure:", f)
    return 0 if all(c.passed for c in checks) else 1


def _cmd_crowd(args) -> int:
    from crowdsim.crowd import apply_scenario, load_population, load_scenario, populate
    from crowdsim.stadium import load_stadium

    st = load_stadium(args.preset)
    pop = load_population()
    for sc in load_scenario(args.scenario).expand():
        ag = populate(apply_scenario(st, sc), sc, pop, args.seed)
        print(json.dumps(ag.summary(), indent=2))
    return 0


def _cmd_check1(args) -> int:
    from crowdsim.checks.phase1 import run

    return run(Path(args.out), seed=args.seed, make_plots=not args.no_plots)


def _cmd_check2(args) -> int:
    from crowdsim.checks.phase2 import run

    return run(Path(args.out))


def _cmd_validate(args) -> int:
    from crowdsim.validation.suite import run

    return run(Path(args.out), quick=args.quick)


def _overrides(args) -> dict:
    from crowdsim.models.params import parse_overrides

    return parse_overrides(args.set)


def _cmd_simulate(args) -> int:
    from crowdsim.experiments.runner import simulate

    simulate(args.stadium, args.scenario, args.model, args.seed, _overrides(args),
             Path(args.out) if args.out else None, make_plots=not args.no_plots)
    return 0


def _cmd_compare(args) -> int:
    from crowdsim.experiments.runner import compare

    models = list(MODELS) if args.models == "all" else [m.strip() for m in args.models.split(",")]
    bad = [m for m in models if m not in MODELS]
    if bad:
        raise ConfigError(f"unknown model(s) {bad}; choose from {', '.join(MODELS)} or 'all'")
    compare(args.stadium, args.scenario, models, args.seeds, _overrides(args), Path(args.out) if args.out else None,
            first_seed=args.first_seed, make_plots=not args.no_plots)
    return 0


def _cmd_animate(args) -> int:
    from crowdsim.experiments.animate import animate

    models = [m.strip() for m in args.models.split(",")]
    bad = [m for m in models if m not in ("ca", "sf", "network")]
    if bad:
        raise ConfigError(f"cannot animate {bad}: choose from ca, sf, network "
                          "(sfpe is an analytic calculation without movement)")
    path = animate(args.stadium, args.scenario, models, args.seed, _overrides(args), args.crowd_fraction,
                   args.frame_dt, args.fps, args.marker_size, Path(args.out) if args.out else None)
    print("video:", path)
    return 0


def _cmd_live(args) -> int:
    from crowdsim.experiments.live import live

    models = list(MODELS) if args.models == "all" else [m.strip() for m in args.models.split(",")]
    live(args.stadium, args.scenario, models, args.seed, _overrides(args), args.crowd_fraction, args.frame_dt,
         record=args.record, window=not args.no_window, save_trajectories=args.trajectories,
         out_root=Path(args.out) if args.out else None, hold=not args.no_hold, marker=args.marker_size, fps=args.fps)
    return 0


def _cmd_jps(args) -> int:
    from crowdsim.verification.jupedsim_check import run

    stadiums = [] if args.stadium == "none" else (list_stadium_presets() if args.stadium == "all"
                                                    else [x.strip() for x in args.stadium.split(",")])
    return run(Path(args.out), stadiums, args.scenario, args.seed, quick=args.quick, live=args.live,
               crowd_fraction=args.crowd_fraction, skip_room=args.skip_room, jps_sf=not args.no_jps_sf,
               record=args.record)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run.py", description="Stadium crowd-flow and emergency-evacuation "
                                "models (ME F340 project).")
    p.add_argument("--version", action="version", version=f"crowdsim {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("list", help="list stadium presets, scenarios and models")
    s.set_defaults(func=_cmd_list)

    s = sub.add_parser("stadium", help="build one stadium preset, print its summary and checks, save figures")
    s.add_argument("preset", help="preset name (see 'list') or path to a YAML file")
    s.add_argument("--out", default=str(RESULTS_DIR / "phase1"), help="figure folder")
    s.add_argument("--no-plots", action="store_true")
    s.set_defaults(func=_cmd_stadium)

    s = sub.add_parser("crowd", help="populate a stadium for a scenario and print the crowd summary")
    s.add_argument("preset")
    s.add_argument("scenario")
    s.add_argument("--seed", type=int, default=DEFAULT_SEED)
    s.set_defaults(func=_cmd_crowd)

    s = sub.add_parser("check-phase1", help="verify Phase 0 + Phase 1 end to end")
    s.add_argument("--out", default=str(RESULTS_DIR / "phase1"), help="output folder")
    s.add_argument("--seed", type=int, default=DEFAULT_SEED)
    s.add_argument("--no-plots", action="store_true")
    s.set_defaults(func=_cmd_check1)

    s = sub.add_parser("check-phase2", help="smoke-test all four models on every stadium (20 %% crowd)")
    s.add_argument("--out", default=str(RESULTS_DIR / "phase2"), help="output folder")
    s.set_defaults(func=_cmd_check2)

    s = sub.add_parser("validate", help="Phase 3 validation suite (free walking, bottleneck flow, ...)")
    s.add_argument("--out", default=str(RESULTS_DIR / "validation"), help="output folder")
    s.add_argument("--quick", action="store_true", help="fewer seeds (about 3x faster)")
    s.set_defaults(func=_cmd_validate)

    for name, helptext in (("simulate", "run ONE model on a stadium scenario and save results + figures"),
                           ("compare", "run several models x seeds on identical crowds and compare them")):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--stadium", required=True, help="stadium preset (see 'list')")
        s.add_argument("--scenario", required=True, help="scenario (see 'list'); sweeps run every value")
        if name == "simulate":
            s.add_argument("--model", required=True, choices=MODELS)
            s.add_argument("--seed", type=int, default=DEFAULT_SEED)
        else:
            s.add_argument("--models", default="all", help="'all' or a comma list, e.g. sfpe,network")
            s.add_argument("--seeds", type=int, default=3, help="number of random crowds (default 3)")
            s.add_argument("--first-seed", type=int, default=1)
        s.add_argument("--set", nargs="*", metavar="KEY=VALUE",
                       help="override model parameters, e.g. --set sf.tau=0.8 network.routing=static")
        s.add_argument("--out", default=None, help="results root folder (default results/runs or results/compare)")
        s.add_argument("--no-plots", action="store_true")
        s.set_defaults(func=_cmd_simulate if name == "simulate" else _cmd_compare)

    s = sub.add_parser("live", help="open a window and WATCH the simulation while it runs; saves all results")
    s.add_argument("--stadium", required=True)
    s.add_argument("--scenario", required=True, help="a non-sweep scenario, e.g. emergency")
    s.add_argument("--models", default="sf", help="'all' or a comma list, run one after another on the same crowd "
                   "(default sf; sfpe only adds its curve)")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--crowd-fraction", type=float, default=1.0, help="share of the scenario occupancy (0.2 = quick test)")
    s.add_argument("--frame-dt", type=float, default=2.0, help="simulated seconds between screen updates (default 2)")
    s.add_argument("--record", action="store_true", help="also write the window to an MP4 (needs ffmpeg)")
    s.add_argument("--no-window", action="store_true", help="no window (e.g. over SSH); combine with --record")
    s.add_argument("--no-hold", action="store_true", help="close the window automatically at the end")
    s.add_argument("--trajectories", action="store_true", help="also save trajectories.csv.gz (positions every frame)")
    s.add_argument("--fps", type=int, default=20, help="frame rate of the recorded video")
    s.add_argument("--marker-size", type=float, default=4.0)
    s.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="override model parameters")
    s.add_argument("--out", default=None, help="results root (default results/live)")
    s.set_defaults(func=_cmd_live)

    s = sub.add_parser("jupedsim-verify", help="cross-check our micro models against the JuPedSim simulator")
    s.add_argument("--stadium", default="indoor_arena", help="stadium(s) for the concourse test: a name, a comma "
                   "list, 'all' or 'none' (default indoor_arena)")
    s.add_argument("--scenario", default="emergency")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--crowd-fraction", type=float, default=1.0)
    s.add_argument("--quick", action="store_true", help="1 crowd in the room test, 30 %% crowd in the stadium")
    s.add_argument("--live", action="store_true", help="watch the JuPedSim concourse run in a window")
    s.add_argument("--record", action="store_true", help="save the JuPedSim concourse run as MP4 (needs ffmpeg)")
    s.add_argument("--skip-room", action="store_true", help="skip the J1 room test")
    s.add_argument("--no-jps-sf", action="store_true", help="skip JuPedSim's social force model")
    s.add_argument("--out", default=str(RESULTS_DIR / "jupedsim"))
    s.set_defaults(func=_cmd_jps)

    s = sub.add_parser("animate", help="replay video (MP4/GIF) of one or more models on the same crowd")
    s.add_argument("--stadium", required=True)
    s.add_argument("--scenario", required=True, help="a non-sweep scenario, e.g. emergency")
    s.add_argument("--models", default="network,ca,sf", help="comma list from ca, sf, network (default all three)")
    s.add_argument("--seed", type=int, default=DEFAULT_SEED)
    s.add_argument("--crowd-fraction", type=float, default=1.0, help="share of the scenario occupancy (e.g. 0.2 for a quick test)")
    s.add_argument("--frame-dt", type=float, default=2.0, help="simulated seconds between video frames (default 2)")
    s.add_argument("--fps", type=int, default=20)
    s.add_argument("--marker-size", type=float, default=5.0)
    s.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="override model parameters")
    s.add_argument("--out", default=None, help="output folder (default results/animations)")
    s.set_defaults(func=_cmd_animate)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"config error: {exc}")
        return 2
