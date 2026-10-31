#!/usr/bin/env bash
# Model-parameter sensitivity examples (each writes to its own results folder).
#     bash scripts/sensitivity_examples.sh
set -euo pipefail
S=football_bowl; X=emergency
python run.py compare --stadium $S --scenario $X --models network --seeds 3 --set network.routing=static --out results/sensitivity/routing_static
python run.py compare --stadium $S --scenario $X --models sf --seeds 3 --set sf.tau=0.8 --out results/sensitivity/sf_tau_0.8
python run.py compare --stadium $S --scenario $X --models ca --seeds 3 --set ca.k_s=4 --out results/sensitivity/ca_ks_4
python run.py compare --stadium $S --scenario $X --models sfpe,network --seeds 3 --set movement.k.aisle=1.08 --out results/sensitivity/aisle_k_1.08
echo "Sensitivity runs finished: results/sensitivity/"
