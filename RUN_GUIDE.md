# Run guide: from unzipping to report results (Ubuntu)

Copy the commands exactly. Each block starts from the `stadium-evac` folder with the venv active
(after step 1, every new terminal needs: `cd ~/Downloads/stadium-evac && source .venv/bin/activate`).

## 1. One-time setup (about 5 minutes)

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-tk ffmpeg   # python3-tk = the live window, ffmpeg = videos
cd ~/Downloads
unzip stadium-evac_live_jupedsim.zip
cd stadium-evac
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-jupedsim.txt     # JuPedSim + PedPy for the cross-check
```

## 2. Check that everything works (about 6 minutes)

```bash
python -m pytest -q            # -> "66 passed"
python run.py check-phase1     # -> "PHASE 1 CHECK: ALL 74 CHECKS PASSED"
python run.py check-phase2     # -> "PHASE 2 CHECK: ALL 37 CHECKS PASSED"   (3-5 min)
python run.py validate         # -> "VALIDATION: ALL 18 HARD CHECKS PASSED (+ 2 observations)"
```

Quick test of the live window (small crowd, about 1-2 min). A window must open and people must move:

```bash
python run.py live --stadium indoor_arena --scenario emergency --models sf --crowd-fraction 0.2
```

## 3. Watch a simulation live (the main demo)

```bash
python run.py live --stadium football_bowl --scenario emergency --models all --seed 1 --record
```

* A window opens and the simulation is drawn **while it is being computed**:
  people (dots) leave their seats (grey), walk down aisles and vomitories, queue (red) and leave
  through the gates (green triangles; closed gates are red X). Colour = walking speed.
* The models run one after another on the **same crowd**: SFPE first (instant, curve only), then
  the cellular automaton, the social force model and the network model, each in its own panel.
* Bottom row: evacuation curve of every model, people out per gate, live counters
  (seated, walking, queueing, on the concourse, highest density with its Fruin level of service).
* Keys in the window: **SPACE** pause / resume, **Q** close the window (the run continues, results are still saved).
* When all models are done the results table is printed in the terminal; close the window to finish.
* `--record` writes exactly what the window shows to an MP4 on your laptop (needs ffmpeg).
  Without `--record` you can screen-record instead (Ubuntu: press the Print Screen key and switch the capture tool to video).

Useful options: `--models sf` (one model), `--models network,sf`, `--seed 2` (another random crowd),
`--crowd-fraction 0.3` (faster), `--frame-dt 1` (smoother, slower), `--trajectories`
(also save every person's position each frame), `--no-window --record` (no window, video only).

Approximate times on a laptop (all four models, full crowd, with `--record`): football bowl and cricket
oval 10-20 min, indoor arena 3-6 min. One model is about a third of that.

### The three stadiums, one after another

```bash
bash scripts/live_three_stadiums.sh             # emergency in football_bowl, cricket_oval, indoor_arena
bash scripts/live_three_stadiums.sh blocked_exit # same with two gates closed
```

Close each window when it has finished to start the next stadium.

### Every live run produces (results/live/<stadium>/<scenario>/)

| File | Content |
|---|---|
| `summary.md`, `summary.csv` | T50 / T90 / T100, gate flow, exit balance, peak density, compute time per model |
| `runs.csv` | the same, one row per model, all columns |
| `evacuation_curves.png`, `evacuation_times.png`, `exit_usage.png`, `density_over_time.png`, `density_maps_peak.png`, `t90_by_group.png`, `compute_time.png` | comparison figures |
| `live_<models>_seed<k>.mp4` | the recorded window (with `--record`) |
| `runs/<model>/seed<k>/agents.csv` | every person: exit time, gate, tier, mobility class, speed, pre-movement |
| `runs/<model>/seed<k>/timeseries.csv` | every 5 s: people seated / walking / queueing / on the concourse / out, max density |
| `runs/<model>/seed<k>/summary.json` | all numbers of that run |
| `runs/<model>/seed<k>/trajectories.csv.gz` | positions every frame (only with `--trajectories`) |

Compute times in live runs exclude the time spent drawing, so they stay comparable with `compare`.

## 4. Scenarios to run for the report

Scenarios: `normal_egress`, `emergency`, `blocked_exit`, `mixed_crowd`, `panic`, `pitch_refuge`
(cricket oval only; pitch gates opened). Watch any of them live:

```bash
python run.py live --stadium football_bowl --scenario blocked_exit --models all --record
python run.py live --stadium football_bowl --scenario panic        --models all --record
python run.py live --stadium cricket_oval  --scenario pitch_refuge --models all --record
```

For the averaged numbers in the report use several crowds without a window (faster):

```bash
bash scripts/part1_model_comparison.sh      # 4 models x 6 scenarios on the football bowl, 3 crowds each
bash scripts/part2_stadium_comparison.sh    # 3 stadiums
bash scripts/sensitivity_examples.sh        # parameter sensitivity
```

Results: `results/compare/<stadium>/<scenario>/` (same files as above, mean +/- sd over the crowds).
Report numbers from these averaged tables; use the live runs for figures, videos and the demo.

## 5. Cross-check with JuPedSim

JuPedSim (https://www.jupedsim.org, Forschungszentrum Juelich) is an independent pedestrian
simulator. It only simulates one 2-D floor, so it is applied to the parts our CA / social force
models simulate in 2-D:

```bash
python run.py jupedsim-verify --live              # room test + indoor arena concourse, JuPedSim drawn live (~10-15 min)
python run.py jupedsim-verify --quick --live      # quick version (~5 min)
bash scripts/jupedsim_verify.sh                   # room + all three stadiums, MP4s of the JuPedSim runs (~1.5-2 h)
```

Single stadium: `python run.py jupedsim-verify --stadium football_bowl` (about 25-30 min: the
football bowl and cricket oval each have about 41,000 people; JuPedSim's CFSM is the slow part).

* **J1 room bottleneck**: the 12 x 12 m room of validation V2/V4, 180 people, doors 1.2-3.6 m, the
  same people (start positions, speeds, sizes) in our CA, our SF, JuPedSim's collision-free speed
  model (CFSM, its default model) and JuPedSim's social force model (same Helbing parameters as ours).
* **J2 stadium concourse**: our hybrid social force run is done first; each person's arrival on the
  exit concourse (time and place) is replayed in JuPedSim on exactly the same floor plan, heading
  for the same gate. With `--live` you watch JuPedSim moving the people on the concourse.

Results in `results/jupedsim/`: `jupedsim_report.md` (tables), `j1_room_bottleneck.csv/.png`,
`j2_<stadium>_concourse.png`, `j2_<stadium>_summary.csv`, `j2_<stadium>_concourse_agents.csv`
(per person: arrival, exit time in each simulator), `*.sqlite` (JuPedSim's own trajectory files),
`*_pedpy_trajectories.png` (trajectories plotted with PedPy, JuPedSim's analysis library),
`j2_<stadium>_jps-cfsm_live.mp4` (with `--record`).

What to expect (from my test runs, seed 1 - write down your own numbers):
* J1: JuPedSim CFSM gives about 1.5-1.8 p/(m s), close to our CA (1.7-2.2); our SF is higher
  (about 2.0-2.3); SFPE is lowest (1.0-1.2, a design value). All except SFPE are inside the
  experimental range 1.2-2.6.
* JuPedSim's social force model **aborts** in the 1.2-2.4 m door tests: crowd pressure pushes a person
  through a wall and JuPedSim stops the run. This is a known weakness of the plain social force model
  at high density and the reason our SF uses an implicit friction step and a speed cap
  (config/models.yaml). Report it as a finding.
* J2 concourse T90 (ours SF / JuPedSim SF / JuPedSim CFSM): indoor arena (30 % crowd) 174 / 173 / 175 s;
  football bowl (full crowd) 497 / 497 / 556 s. With the same arrivals JuPedSim's social force model
  reproduces ours almost exactly; CFSM is slower in the big stadium because its door flow is lower
  (as in J1). The arrival curve and the exit curve almost coincide in the football bowl: the concourse
  and gates are not what limits the evacuation there - the stands (aisles, vomitories, stairs) are.
* "mean spawn delay": JuPedSim only adds a person when the spot is completely free (about 1 s on average).

## 6. Showing it to your professor

1. `python run.py live --stadium football_bowl --scenario emergency --models sf --crowd-fraction 0.5`
   (about 3 min, window opens immediately).
2. Open `results/live/football_bowl/emergency_50pct/summary.md` and the CSVs it wrote.
3. `python run.py jupedsim-verify --quick --live` (JuPedSim running on the same concourse).
4. Change a parameter and show that the result changes, e.g. `--set sf.tau=0.8` or a different `--seed`.

## 7. What to record for the report

See `report/OBSERVATIONS_GUIDE.md` (tables to fill, observations per scenario, questions for the
critical comparison, assumptions table). Add a short "external verification" subsection with
J1 (flow vs door width for ours vs JuPedSim) and J2 (concourse curves) and the JuPedSim social force aborts.

## Troubleshooting

* `no GUI backend for matplotlib`: `sudo apt install -y python3-tk`, then run the command again.
  (Alternative inside the venv: `pip install PyQt6`.)
* `no display found`: you are in a terminal without a desktop (e.g. SSH). Use `--no-window --record`.
* `--record needs ffmpeg`: `sudo apt install -y ffmpeg`.
* `JuPedSim is not installed`: `pip install -r requirements-jupedsim.txt` with the venv active.
* The window is slow with the full crowd: add `--frame-dt 4` or `--crowd-fraction 0.3`.
* Any traceback: copy the whole terminal output.
