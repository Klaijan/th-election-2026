#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   source scripts/activate_venv.sh            # activates .venv
#   source scripts/activate_venv.sh .venv311   # activates a named venv
#
# This uses whatever `python` is on PATH (recommended: controlled by pyenv).

venv_dir="${1:-.venv}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ ! -d "$venv_dir" ]]; then
  python -m venv "$venv_dir"
fi

# shellcheck disable=SC1090
source "$venv_dir/bin/activate"

python -m pip install -U pip >/dev/null
echo "Activated: $venv_dir ($(python --version))"


