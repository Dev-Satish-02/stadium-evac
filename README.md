# Stadium crowd-flow and emergency-evacuation models (ME F340 project)

Code-based framework that conceptualises four crowd-flow / evacuation models for
stadiums and compares them under simulated conditions.

**Status: Phases 0-3 complete + live simulation window + JuPedSim cross-check.**
**Start with `RUN_GUIDE.md`** (every command from unzipping to report results). Contents: project skeleton, stadium generator and presets,
crowd and scenarios, metrics (Phase 1); four models plus hybrid linkage (Phase 2);
validation suite (Phase 3); `simulate` / `compare` commands to produce results.

| Model | Type | Idea |
|---|---|---|
| **M1 `sfpe`** Hydraulic (SFPE) | macroscopic, analytic | everyone takes the nearest exit; each route's *controlling component* (largest people / capacity) is a first-come-first-served queue at the SFPE capacity |
| **M2 `ca`** Cellular automaton | microscopic, discrete | floor-field CA (0.4 m cells), density-dependent speed, friction, herding via a dynamic field |
| **M3 `sf`** Social force | microscopic, continuous | Helbing (2000) forces, body contact and friction, herding (panic) parameter |
| **M4 `network`** Queueing network | mesoscopic | element-by-element movement with SFPE capacities and storage, spill-back, congestion-aware route choice |

M2 and M3 are **hybrid** in the stadium: the network (M4) carries people from their
seats through aisles, vomitories, upper concourses and stairs; the CA / social force
model simulates the **exit concourse and gates** in 2-D. People wait on the element
they arrived on until the concourse has room, so congestion backs up into the stands.

---

## 1. Setup (Ubuntu, once)

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip   # only if venv is missing
cd stadium-evac
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Tested on Python 3.10, 3.12 and 3.13. The first social-force run compiles its
numba kernels (about 10-30 s once); later runs reuse the cache.

## 2. Verify everything (in this order)

```bash
source .venv/bin/activate
python -m pytest -q            # unit tests              -> "66 passed"
python run.py check-phase1     # stadiums, crowds, metrics -> "PHASE 1 CHECK: ALL 74 CHECKS PASSED"
python run.py check-phase2     # all models x stadiums     -> "PHASE 2 CHECK: ALL 37 CHECKS PASSED" (~3-5 min)
python run.py validate         # Phase 3 validation        -> "VALIDATION: ALL 18 HARD CHECKS PASSED (+ 2 observations)" (~1-2 min)
```

Each command exits with status 0 only if everything passed.

## 3. Run simulations

```bash
python run.py list                                     # presets, scenarios, models

# one model, one scenario (sweep scenarios run every value)
python run.py simulate --stadium football_bowl --scenario emergency --model sf --seed 1

# all four models on identical crowds, 3 random crowds each  (main command)
python run.py compare --stadium football_bowl --scenario emergency --seeds 3

# a subset of models / parameter overrides / custom output folder
python run.py compare --stadium indoor_arena --scenario blocked_exit --models sfpe,network --seeds 5
python run.py compare --stadium football_bowl --scenario emergency --models sf --seeds 3 \
    --set sf.tau=0.8 --out results/sensitivity/sf_tau_0.8
```

Batch scripts (run from this folder with the venv active):

```bash
bash scripts/part1_model_comparison.sh     # Part 1: 4 models on the football bowl, 6 scenarios (~45-90 min)
bash scripts/part2_stadium_comparison.sh   # Part 2 preview: 3 stadiums x scenarios (~60-120 min)
bash scripts/sensitivity_examples.sh       # parameter sensitivity examples (~15-30 min)
```

Times are for a typical laptop; one `compare` with all models and one seed takes
about 1.5-2.5 min on the large stadiums (CA is the slowest model, SFPE the fastest).

## 3a. Watch a simulation live (window) - see RUN_GUIDE.md

```bash
sudo apt install -y python3-tk ffmpeg      # once: window + video support
python run.py live --stadium football_bowl --scenario emergency --models all --seed 1 --record
bash scripts/live_three_stadiums.sh        # the three stadiums one after another
```

The window draws the run while it is computed (SPACE pause, Q close); afterwards all CSVs and
figures are in `results/live/<stadium>/<scenario>/` and the recorded MP4 next to them.

## 3c. Cross-check with JuPedSim

```bash
pip install -r requirements-jupedsim.txt
python run.py jupedsim-verify --live       # room bottleneck + indoor arena concourse in JuPedSim
bash scripts/jupedsim_verify.sh            # all three stadiums
```

Results and explanation: `results/jupedsim/jupedsim_report.md`, RUN_GUIDE.md section 5.

## 3b. Replay videos (watch the simulation run)

```bash
# one video with the three spatial models side by side on the SAME crowd (takes ~10-20 min for the football bowl)
python run.py animate --stadium football_bowl --scenario emergency --models network,ca,sf --seed 1

# quick preview with 20 % of the crowd (~1-2 min)
python run.py animate --stadium indoor_arena --scenario blocked_exit --crowd-fraction 0.2 --frame-dt 3
```

Four ready-made videos: `bash scripts/make_videos.sh`.

Output: `results/animations/<stadium>/<scenario>_<models>_seed<k>.mp4` (a GIF if `ffmpeg` is not installed:
`sudo apt install ffmpeg`). Plan view, one panel per model, people coloured by speed (red = stopped,
blue = free walking, grey = still seated), closed gates marked X, a live clock and an evacuation curve
underneath. Options: `--frame-dt` (simulated seconds per video frame, default 2), `--fps`, `--marker-size`,
`--set KEY=VALUE`. SFPE (M1) is an analytic calculation with no movement, so it is not animated.
Dots on the exit concourse are the true CA / social-force positions; in the stands (rows, aisles,
vomitories, stairs) they are placed along their route from the network model's progress. Levels overlap in
plan view (stands, concourse and stairs are drawn on top of each other).

## 4. Where results go

`results/compare/<stadium>/<scenario>/` (sweeps: one sub-folder per value plus `sweep.png`, `sweep_summary.csv`):

| File | Content |
|---|---|
| `summary.md`, `summary.csv` | mean +/- sd per model: T50, T90, T95, T100, evacuated fraction, gate flow, exit-load CV, peak density, time at >= 4 p/m2, T90 per tier and per mobility class, compute time |
| `runs.csv` | one row per (model, seed) |
| `evacuation_curves.png` | fraction evacuated vs time, every model and seed |
| `evacuation_times.png` | T50 / T90 / T100 bars with error bars |
| `exit_usage.png` | people per gate per model (closed gates marked) |
| `density_over_time.png` | highest density over time (network, CA, SF) |
| `density_maps_peak.png` | exit concourse at the moment of peak density (CA, SF) |
| `t90_by_group.png` | T90 per tier and mobility class |
| `compute_time.png` | computational cost (log scale) |
| `runs/<model>/seed<k>/` | `agents.csv` (every person: exit time, gate, tier, class, speed, ...), `timeseries.csv`, `summary.json` |

`results/runs/...` (from `simulate`) has the same files for one run plus
`run_overview.png`, `density_timeseries.png`, `density_map_peak.png`.
`results/validation/` holds the validation CSVs, figures and `validation_report.md`.

See `report/OBSERVATIONS_GUIDE.md` for what to record from each output.

## 5. Project layout

```
run.py                         single entry point
config/stadiums/*.yaml         football_bowl, cricket_oval, indoor_arena
config/scenarios/*.yaml        normal_egress, emergency, blocked_exit, exit_width_sweep, mixed_crowd, panic, pitch_refuge
config/crowd/population.yaml   mobility classes (speeds, body sizes), pre-movement
config/models.yaml             ALL model parameters (with sources / calibration notes)
crowdsim/geometry/             field curve and offset curves (bowl coordinates)
crowdsim/stadium/              preset schema, generator, navigation graph, checks, figures
crowdsim/crowd/                population, scenarios, agents
crowdsim/metrics/              shared metrics (times, flows, exit balance, density, Fruin LOS)
crowdsim/models/movement.py    shared SFPE movement rules (speed-density, capacity, widths)
crowdsim/models/egressnet.py   array form of the navigation graph
crowdsim/models/sfpe.py        M1 hydraulic / SFPE
crowdsim/models/ca.py          M2 cellular automaton
crowdsim/models/sf.py          M3 social force (numba)
crowdsim/models/network.py     M4 queueing network
crowdsim/models/domain.py      2-D domains for CA/SF (floor field, wall distance)
crowdsim/models/hybrid.py      network upstream + CA/SF on the exit concourse
crowdsim/validation/suite.py   Phase 3 validation
crowdsim/experiments/runner.py simulate / compare
crowdsim/experiments/animate.py animate (replay video)
crowdsim/animation/            position recorder, live window, video renderer
crowdsim/experiments/live.py   live (watch a run, then save results)
crowdsim/verification/         JuPedSim cross-check
crowdsim/analysis/plots.py     figures
crowdsim/checks/               phase-1 and phase-2 end-to-end checks
scripts/                       batch runs for Part 1, Part 2, sensitivity
tests/                         pytest unit tests
```

## 6. Modelling notes (for the methodology section)

* **Height without 3-D:** each tier and concourse is a level; stepped aisles, vomitories,
  stairs and ramps are network elements with 3-D length, width, vertical drop and type.
  Speeds on stairs/aisles use the SFPE k-values (and a reduction when climbing).
* **Same inputs for every model:** one crowd per seed (seats, speeds, body size,
  pre-movement) is given to all four models; all are scored by `crowdsim/metrics`.
* **Calibration (see `results/validation/`):**
  CA - density-dependent speed (SFPE ratio, floor 0.3) and k_s = 8, otherwise a 0.4 m CA
  passes ~4-5 p/(m s) and walks ~20 % slow in corridors (lattice artefact).
  SF - Helbing parameters with a 0.07 m personal-space margin (adult radius ~0.30 m),
  speed cap = desired speed, semi-implicit friction for numerical stability, a small
  random force so arches at narrow doors can break.
* **Known limitations to discuss:** no counter-flow, micro models only on the exit
  concourse, static (nearest-exit) floor field in CA/SF, SFPE values are design values
  (conservative), all geometry and crowd values are assumptions to be cited.

## 7. Troubleshooting

* `python3 -m venv` fails: install `python3-venv` (step 1).
* `ModuleNotFoundError`: activate the venv and run commands from the `stadium-evac` folder.
* Any FAIL line: send the full terminal output.
