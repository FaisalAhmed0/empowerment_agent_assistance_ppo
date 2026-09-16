#!/bin/bash
# Install the project env on a Compute Canada login node (has internet).
# Usage: bash scripts/setup_login.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

module purge
module load StdEnv/2023 gcc/12.3 python/3.10 scipy-stack/2024a mujoco/3.1.6
module load httpproxy

export UV_CACHE_DIR="${SCRATCH:-$HOME}/uv-cache"
export UV_PYTHON_PREFERENCE=only-system
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
mkdir -p "$UV_CACHE_DIR"

uv venv --python "$(which python3)"
uv sync

uv run python -c "import numpy, jax, mujoco; print(numpy.__version__, jax.__version__, mujoco.__version__)"
