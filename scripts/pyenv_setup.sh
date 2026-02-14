#!/usr/bin/env bash
set -euo pipefail

# One-time-ish project setup when using pyenv (+ optional pyenv-virtualenv).
#
# What it does:
# - Ensures the pyenv "local" selected by `.python-version` exists
#   - If `.python-version` names a pyenv-virtualenv environment (e.g. `vote69-311`) and it
#     doesn't exist yet, the script can create it if you provide `PYTHON_VERSION`.
# - Creates a project venv (default: .venv311) using the pyenv-selected python
# - Installs requirements.txt
#
# Usage:
#   bash scripts/pyenv_setup.sh
#   VENV_DIR=.venv311 bash scripts/pyenv_setup.sh
#   PYTHON_VERSION=3.11.8 bash scripts/pyenv_setup.sh   # create missing pyenv env named in .python-version

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v pyenv >/dev/null 2>&1; then
  echo "pyenv not found. Install it first (e.g. brew install pyenv) and set up your shell." >&2
  exit 2
fi

py_local="$(cat .python-version | tr -d '[:space:]')"
if [[ -z "$py_local" ]]; then
  echo "Missing .python-version" >&2
  exit 3
fi

if pyenv versions --bare | tr -d '[:space:]' | grep -Fxq "$py_local"; then
  : "pyenv local '$py_local' already exists"
else
  base_py="${PYTHON_VERSION:-}"
  if [[ -z "$base_py" ]]; then
    echo "pyenv local '$py_local' (from .python-version) does not exist." >&2
    echo "Either create it manually, e.g.:" >&2
    echo "  pyenv install 3.11.8" >&2
    echo "  pyenv virtualenv 3.11.8 $py_local" >&2
    echo "" >&2
    echo "Or re-run with PYTHON_VERSION set, e.g.:" >&2
    echo "  PYTHON_VERSION=3.11.8 bash scripts/pyenv_setup.sh" >&2
    exit 4
  fi

  if ! pyenv commands | grep -Fxq "virtualenv"; then
    echo "pyenv-virtualenv not installed (missing 'pyenv virtualenv' command)." >&2
    echo "Install it (e.g. brew install pyenv-virtualenv) and ensure your shell init loads it." >&2
    exit 5
  fi

  pyenv install -s "$base_py"
  pyenv virtualenv "$base_py" "$py_local"
fi

pyenv local "$py_local"

venv_dir="${VENV_DIR:-.venv311}"
pyenv exec python -m venv "$venv_dir"

# shellcheck disable=SC1090
source "$venv_dir/bin/activate"
python -m pip install -U pip
python -m pip install -r requirements.txt

echo "Ready: $venv_dir ($(python --version))"


