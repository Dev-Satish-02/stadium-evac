# What to observe and record

Record numbers from `summary.md` / `summary.csv` (mean +/- sd over seeds), not from one run.
Say which seed count you used. Every claim in the report should point to a figure or table.

## A. Validation (results/validation/) - "Model verification and calibration" section

| Test | Record | What it tells you |
|---|---|---|
| V1 free walking | measured vs desired speed per model (`v1_free_walk.csv`) | models reproduce free walking; CA lattice error (~3-4 %) |
| V2 bottleneck | specific flow vs door width (`v2_bottleneck.png`) | CA/SF give ~1.6-2.2 p/(m s), inside the experimental range; SFPE/network use the design value ~1.0-1.2 per clear width - SFPE is deliberately conservative |
| V3 faster-is-slower | median T90 vs desired speed, clogged runs (`v3_faster_is_slower.png`) | whether the SF reproduces the panic effect, and how strongly (state the % rise honestly) |
| V4 same room | T50/T90/T100 for all four models (`v4_cross_model_room.png`) | micro models evacuate the room about 2x faster than SFPE/network - the gap is the SFPE capacity vs measured flows |
| V5 / V6 / V7 / V8 | PASS lines | network keeps capacity; SFPE equals hand calculation; nobody lost in hybrid runs; reproducible |

## B. Part 1 - model comparison on the football bowl (results/compare/football_bowl/...)

For every scenario fill one table:

| Model | T50 | T90 | T100 | gate flow | exit load CV | peak density | time >= 4 p/m2 | compute time |
|---|---|---|---|---|---|---|---|---|

Then note:

1. **Agreement and disagreement** (`evacuation_curves.png`, `evacuation_times.png`):
   which models agree on T90? Where do curves split - start (pre-movement, walking),
   middle (bottleneck flow) or tail (stragglers, far seats)? T100 is driven by single
   people; T90 is the more robust comparison metric.
2. **Bottleneck behaviour**: in `emergency`, the gates are the designed bottleneck.
   Compare the micro models (gate flows emerge from interactions) with SFPE/network
   (gate flow fixed by the SFPE capacity).
3. **Exit balance** (`exit_usage.png`, load CV): static routing (SFPE, CA/SF on the concourse)
   vs congestion-aware routing (network). Which gates are over/under used and why
   (look at the level plans from Phase 1)?
4. **Density / safety** (`density_over_time.png`, `density_maps_peak.png`): where and when
   the peak occurs (which gates/vomitory landings), Fruin level of service, time above
   4 p/m2. Note the CA peak is quantised (max 6.25 p/m2 = one person per 0.4 m cell) and
   the network peak is capped at its jam density (3.5 p/m2): only the SF density is
   continuous - this is itself a model-comparison point.
5. **Scenario effects** across `normal_egress`, `emergency`, `blocked_exit`, `mixed_crowd`,
   `panic`: how much does each change T90 per model? Does the ranking of models change?
   In `blocked_exit`, how is the load redistributed? In `mixed_crowd`, look at
   `t90_by_group.png` (elderly / mobility-impaired T90). In `panic`, does any model show
   faster-is-slower at the gates?
6. **Sensitivity** (`exit_width_sweep/sweep.png`): T90 vs gate width per model. Where does the
   curve flatten (the bottleneck moves upstream to vomitories / stairs / aisles)? Which
   models are most sensitive?
7. **Tiers** (`t90_by_group.png`): upper vs lower tier T90 - the cost of height and stairs.
8. **Cost vs fidelity** (`compute_time.png`): seconds per run vs what each model can
   represent (density, interactions, route choice).

## C. Part 2 - stadium designs (results/compare/<stadium>/...)

Normalise before comparing: capacity and exit width per 1000 seats are in
`results/phase1/stadium_summary.csv`. Record T90 per 1000 people or T90 vs exit width
per 1000. Note the design features that explain differences: two tiers + stairs
(football), pitch gates and long ramps (cricket, compare `emergency` vs `pitch_refuge`),
sunken bowl with climbing exits and unused corner gates (indoor arena).

## D. Critical comparison - questions to answer in the discussion

* Which model would you use for (a) a quick design check, (b) gate/concourse crowd safety,
  (c) route-choice / exit-closure planning? Why?
* Where do the models' assumptions drive the result rather than physics (SFPE design
  capacity, CA lattice, SF calibration, network jam density)?
* What would real data (video counts at gates) be needed to validate the full-stadium results?
* Limitations: no counter-flow; micro models only on the exit concourse; static floor field
  in CA/SF; all geometry and crowd values are assumptions with cited sources.

## E. Assumptions table (fill with sources)

| Quantity | Value used | Where set | Source |
|---|---|---|---|
| free walking speed (adult) | 1.34 +/- 0.26 m/s | config/crowd/population.yaml | |
| SFPE k values, a = 0.266 | level 1.40, stair 1.08, aisle 1.00 | config/models.yaml | SFPE Handbook |
| boundary layers | stairs 0.15, doors 0.15, concourse 0.46 m ... | config/models.yaml | SFPE Handbook |
| Fruin LOS thresholds | 3.25 / 2.32 / 1.39 / 0.93 / 0.46 m2 per person | crowdsim/metrics/density.py | Fruin (1971) |
| crush density | 5 p/m2 | crowdsim/metrics/density.py | |
| SF parameters | A 2000 N, B 0.08 m, k 1.2e5, kappa 2.4e5, tau 0.5 s | config/models.yaml | Helbing et al. (2000) |
| CA parameters | k_s 8, k_d 2, friction 0.1 (+0.4 x herding) | config/models.yaml | Burstedde (2001), Kirchner (2002) |
| pre-movement times | per scenario | config/scenarios/*.yaml | |
| stadium dimensions | per preset | config/stadiums/*.yaml | design assumption |

## F. External verification with JuPedSim (results/jupedsim/)

| Test | Record | What it tells you |
|---|---|---|
| J1 room bottleneck | specific flow and T90 vs door width for our CA, SF, SFPE and JuPedSim CFSM / SF (`j1_room_bottleneck.png`, `jupedsim_report.md`) | whether our micro models agree with an independent simulator on the same crowd; how far apart the models are |
| J1 aborted runs | which JuPedSim social force runs stopped and when | the plain social force model is unstable at high density - why our SF uses implicit friction and a speed cap |
| J2 concourse | T50/T90/T100, time on concourse, spawn delay per simulator (`j2_<stadium>_summary.csv`, `j2_<stadium>_concourse.png`) | whether our concourse dynamics agree with JuPedSim for the same arrivals |
| J2 limits | one-way coupling (arrivals fixed from our run), gate choice fixed to nearest gate | what the cross-check can and cannot show |

Live runs (`python run.py live`): take screenshots / the recorded MP4 at key moments (first queue at
the vomitories, peak at the gates, last people out) and note the time stamps shown in the window.
