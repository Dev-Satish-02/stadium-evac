#!/usr/bin/env bash
# Part 2 (preview of Phase 4) - the same scenarios on all three stadium designs.
# Uses the two models that are cheapest and bracket the others (network, social force);
# change MODELS to "all" if you have the time.
#     bash scripts/part2_stadium_comparison.sh
set -euo pipefail
MODELS=network,sf
for STADIUM in football_bowl cricket_oval indoor_arena; do
  for SCEN in emergency blocked_exit mixed_crowd; do
    python run.py compare --stadium "$STADIUM" --scenario "$SCEN" --models "$MODELS" --seeds 3
  done
done
python run.py compare --stadium cricket_oval --scenario pitch_refuge --models "$MODELS" --seeds 3
echo "Part 2 finished. Results: results/compare/<stadium>/<scenario>/"
