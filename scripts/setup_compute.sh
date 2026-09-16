#!/bin/bash
# Load GPU-node modules for the project env. Source it so modules stay in your shell.
# Usage (after salloc): source scripts/setup_compute.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

module purge
module load StdEnv/2023 gcc/12.3 python/3.10 scipy-stack/2024a mujoco/3.1.6 cuda/12.2

export UV_CACHE_DIR="${SCRATCH:-$HOME}/uv-cache"
export UV_PYTHON_PREFERENCE=only-system
export UV_LINK_MODE=copy

uv run python -c "import jax; print(jax.__version__, jax.devices())"
