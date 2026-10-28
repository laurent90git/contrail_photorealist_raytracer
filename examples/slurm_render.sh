#!/usr/bin/env bash
# SLURM template for a high-resolution render on a GPU node.  Adapt the
# partition / account / gres lines to your cluster, then:
#   sbatch examples/slurm_render.sh examples/day_ground.json
#SBATCH --job-name=contrail
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=contrail_%j.log
set -euo pipefail
SCENE="${1:-examples/day_ground.json}"
cd "${SLURM_SUBMIT_DIR:-.}"
source activate.sh                      # created by setup.sh
nvidia-smi || true
contrail-render --scene "$SCENE" --arch cuda \
                --width "${W:-7680}" --height "${H:-4320}" --spp "${SPP:-1024}" \
                --tile 2048 --batch 8 --out renders \
                --name "$(basename "${SCENE%.json}")_${SLURM_JOB_ID:-local}"
