#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BASELINES=(
  scripts/no_teacher.sh
  scripts/oracle_teacher.sh
  scripts/uniform_teacher.sh
  scripts/icm.sh
  scripts/rnd.sh
)

for baseline in "${BASELINES[@]}"; do
  echo "=== Running ${baseline} ==="
  bash "${baseline}"
done

echo "All baselines submitted."
