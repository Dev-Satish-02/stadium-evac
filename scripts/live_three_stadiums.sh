#!/usr/bin/env bash
# Watch the emergency evacuation of each of the three stadiums live, with the three spatial
# models (network, CA, social force) run one after another on the SAME crowd; SFPE adds its curve.
# Each run also records its window to an MP4 and saves all CSV results under results/live/.
# When a run finishes, CLOSE ITS WINDOW to start the next stadium.
# Usage (venv active, from the stadium-evac folder):  bash scripts/live_three_stadiums.sh [scenario]
set -euo pipefail
cd "$(dirname "$0")/.."
SCEN="${1:-emergency}"
for ST in football_bowl cricket_oval indoor_arena; do
  python run.py live --stadium "$ST" --scenario "$SCEN" --models all --seed 1 --record
done
echo "Done. Videos and results: results/live/<stadium>/$SCEN/"
