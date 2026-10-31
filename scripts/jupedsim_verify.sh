#!/usr/bin/env bash
# Full JuPedSim cross-check: room bottleneck (J1) + exit concourse of all three stadiums (J2).
# Needs:  pip install -r requirements-jupedsim.txt   Results: results/jupedsim/
set -euo pipefail
cd "$(dirname "$0")/.."
python run.py jupedsim-verify --stadium all --record
