# JuPedSim cross-check (JuPedSim 1.4.2)

## J1 room bottleneck (180 people, 12 x 12 m room)

| door [m] | model | specific flow [p/(m s)] | T90 [s] | evacuated (min) |
|---|---|---|---|---|
| 1.2 | ca | 2.14 | 63.5 | 1.00 |
| 1.2 | jps-cfsm | 1.88 | 72.5 | 1.00 |
| 1.2 | jps-sf | nan | inf | 0.03 |
| 1.2 | sf | 1.99 | 67.8 | 1.00 |
| 1.2 | sfpe | 0.99 | nan | 1.00 |
| 1.6 | ca | 1.95 | 52.1 | 1.00 |
| 1.6 | jps-cfsm | 1.84 | 55.1 | 1.00 |
| 1.6 | jps-sf | 2.48 | inf | 0.05 |
| 1.6 | sf | 2.21 | 45.9 | 1.00 |
| 1.6 | sfpe | 1.07 | nan | 1.00 |
| 2.4 | ca | 1.66 | 40.3 | 1.00 |
| 2.4 | jps-cfsm | 1.66 | 41.2 | 1.00 |
| 2.4 | jps-sf | 2.76 | inf | 0.08 |
| 2.4 | sf | 2.19 | 30.9 | 1.00 |
| 2.4 | sfpe | 1.15 | nan | 1.00 |
| 3.6 | ca | 1.64 | 27.6 | 1.00 |
| 3.6 | jps-cfsm | 1.47 | 31.0 | 1.00 |
| 3.6 | jps-sf | 2.52 | 18.3 | 1.00 |
| 3.6 | sf | 2.16 | 21.0 | 1.00 |
| 3.6 | sfpe | 1.21 | nan | 1.00 |

JuPedSim runs that aborted: 8 (jps-sf door 1.2 m; jps-sf door 1.6 m; jps-sf door 2.4 m)
JuPedSim stops a run when a person is pushed outside the walkable area (crowd pressure).
Their flow is left out of the plot.

## J2 cricket_oval exit concourse

| model | status | evacuated | T50 [s] | T90 [s] | T100 [s] | mean time on concourse [s] | mean spawn delay [s] |
|---|---|---|---|---|---|---|---|
| sf | completed | 1.000 | 248 | 477 | 831 | 9.0 | 0.0 |
| jps-cfsm | completed | 1.000 | 291 | 637 | 876 | 13.8 | 54.7 |
| jps-sf | completed | 1.000 | 252 | 493 | 831 | 7.4 | 7.0 |

## J2 football_bowl exit concourse

| model | status | evacuated | T50 [s] | T90 [s] | T100 [s] | mean time on concourse [s] | mean spawn delay [s] |
|---|---|---|---|---|---|---|---|
| sf | completed | 1.000 | 253 | 497 | 847 | 6.5 | 0.0 |
| jps-cfsm | completed | 1.000 | 281 | 557 | 846 | 10.3 | 27.0 |
| jps-sf | completed | 1.000 | 253 | 497 | 847 | 5.4 | 1.3 |

## J2 indoor_arena exit concourse

| model | status | evacuated | T50 [s] | T90 [s] | T100 [s] | mean time on concourse [s] | mean spawn delay [s] |
|---|---|---|---|---|---|---|---|
| sf | completed | 1.000 | 225 | 522 | 800 | 8.8 | 0.0 |
| jps-cfsm | completed | 1.000 | 231 | 523 | 798 | 10.1 | 2.9 |
| jps-sf | completed | 1.000 | 230 | 521 | 800 | 8.2 | 3.2 |

Figures: j1_room_bottleneck.png, j2_<stadium>_concourse.png, *_pedpy_trajectories.png (PedPy)
JuPedSim trajectories: *.sqlite (open with PedPy or the JuPedSim visualizer).
