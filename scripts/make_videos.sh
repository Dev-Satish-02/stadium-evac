#!/usr/bin/env bash
# Replay videos for the report / presentation (run from the stadium-evac folder with the venv active).
# Each video shows network, CA and social force on the SAME crowd. Total ~1 h; videos in results/animations/.
set -euo pipefail
cd "$(dirname "$0")/.."
python run.py animate --stadium football_bowl --scenario emergency   --models network,ca,sf --seed 1
python run.py animate --stadium football_bowl --scenario blocked_exit --models network,ca,sf --seed 1
python run.py animate --stadium indoor_arena  --scenario emergency   --models network,ca,sf --seed 1
python run.py animate --stadium cricket_oval  --scenario emergency   --models network,ca,sf --seed 1
