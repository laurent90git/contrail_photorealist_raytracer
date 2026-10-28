#!/usr/bin/env bash
# Install the conda environment for the contrail renderer.
#
#   bash setup.sh                      # env "contrail" in the default conda envs dir
#   ENV_NAME=myenv bash setup.sh
#   ENV_PREFIX=/scratch/$USER/envs/contrail bash setup.sh   # explicit location (home quota)
#   PRECOMPUTE=1 bash setup.sh         # also build the Mie tables and run the tests
#   CONDA_MODULE=miniconda/24.1.2 bash setup.sh   # other module name
#
# Then:  source activate.sh   (or: module load miniconda/23.1.0 && conda activate <env>)
#
# Taichi's CUDA back-end only needs the NVIDIA driver (no CUDA toolkit / module).
set -euo pipefail

CONDA_MODULE="${CONDA_MODULE:-miniconda/23.1.0}"
ENV_NAME="${ENV_NAME:-contrail}"
ENV_PREFIX="${ENV_PREFIX:-}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"   # Taichi 1.7 supports Python 3.8-3.12
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- conda --------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    if command -v module >/dev/null 2>&1 || [ -n "${MODULESHOME:-}" ]; then
        [ -n "${MODULESHOME:-}" ] && [ -f "$MODULESHOME/init/bash" ] && source "$MODULESHOME/init/bash"
        echo "[setup] module load $CONDA_MODULE"
        module load "$CONDA_MODULE"
    fi
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "[setup] ERROR: conda not found (tried 'module load $CONDA_MODULE')." >&2
    exit 1
fi
eval "$(conda shell.bash hook)"

if [ -n "$ENV_PREFIX" ]; then
    TARGET=(--prefix "$ENV_PREFIX"); ACT="$ENV_PREFIX"
else
    TARGET=(--name "$ENV_NAME"); ACT="$ENV_NAME"
fi

# --- environment --------------------------------------------------------
if { [ -n "$ENV_PREFIX" ] && [ -d "$ENV_PREFIX/conda-meta" ]; } || \
   { [ -z "$ENV_PREFIX" ] && conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; }; then
    echo "[setup] environment '$ACT' already exists, updating packages"
else
    echo "[setup] creating environment '$ACT' (python $PYTHON_VERSION)"
    conda create -y "${TARGET[@]}" "python=$PYTHON_VERSION" pip
fi
conda activate "$ACT"
python -m pip install --upgrade pip
python -m pip install -e "$HERE[test]"

# --- check --------------------------------------------------------------
python - <<'EOF'
import taichi as ti, numpy, h5py, scipy, tifffile, PIL
print("[setup] python packages OK, taichi", ti.__version__)
for arch in ("cuda", "vulkan"):
    try:
        ti.init(arch=getattr(ti, arch), log_level=ti.ERROR)
        ok = ti.lang.impl.current_cfg().arch.name == arch
        ti.reset()
    except Exception:
        ok = False
    print(f"[setup] {arch:6s} back-end: {'available' if ok else 'not available'}")
EOF

cat > "$HERE/activate.sh" <<EOF
# source this file to activate the renderer environment
command -v conda >/dev/null 2>&1 || module load $CONDA_MODULE
eval "\$(conda shell.bash hook)"
conda activate $ACT
EOF

if [ "${PRECOMPUTE:-0}" = "1" ]; then
    cd "$HERE"
    echo "[setup] building Mie tables (cached in \${CONTRAIL_CACHE:-~/.cache/contrail_render}) and running the tests"
    contrail-mie --sigma-g 1.3
    python -m pytest -q tests
fi

echo "[setup] done.  Next:  source activate.sh && contrail-render --help"
