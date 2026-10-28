#!/usr/bin/env bash
# Render every example scene.  Usage:
#   bash examples/run_examples.sh                       # GPU, 1920x1080, 256 spp
#   ARCH=cpu W=800 H=450 SPP=64 bash examples/run_examples.sh
#   bash examples/run_examples.sh corona dusk_ground    # only some scenes
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ARCH="${ARCH:-gpu}"; W="${W:-1920}"; H="${H:-1080}"; SPP="${SPP:-256}"; OUT="${OUT:-renders}"
SCENES=("$@")
[ ${#SCENES[@]} -eq 0 ] && SCENES=(day_side day_ground dusk_ground corona along_contrail)
for s in "${SCENES[@]}"; do
    contrail-render --scene "examples/$s.json" --arch "$ARCH" --width "$W" --height "$H" \
                    --spp "$SPP" --out "$OUT" --name "$s"
done
