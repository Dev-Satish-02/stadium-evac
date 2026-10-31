#!/usr/bin/env bash
# Part 1 - compare the four models on the reference stadium (football bowl).
# Every scenario: 3 random crowds, all four models on identical crowds.
# Run from the stadium-evac folder with the virtual environment active:
#     bash scripts/part1_model_comparison.sh
set -euo pipefail
STADIUM=football_bowl
for SCEN in emergency normal_egress blocked_exit mixed_crowd panic; do
  python run.py compare --stadium "$STADIUM" --scenario "$SCEN" --seeds 3
done
# design sensitivity: 5 gate widths x 3 crowds x 4 models
python run.py compare --stadium "$STADIUM" --scenario exit_width_sweep --seeds 3
echo "Part 1 finished. Results: results/compare/$STADIUM/"
