#!/usr/bin/env bash
#
# One-shot setup for the DIT Reinforcement Learning course.
#
# Usage:
#   git clone <repo-url> RL_exercices
#   cd RL_exercices
#   ./install.sh
#
# It will:
#   1. install uv (the Python package manager) if it isn't already present
#   2. create the virtual environment and install every dependency (uv sync)
#   3. register the "Python (rl-course)" Jupyter kernel
#
set -euo pipefail

# Always run from the directory this script lives in (the repo root).
cd "$(dirname "$0")"

echo "=================================================================="
echo " Reinforcement Learning course - environment setup"
echo " Dakar Institute of Technology (DIT)"
echo "=================================================================="

# --- 1. Ensure uv is installed --------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "[1/3] uv not found -> installing it..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Make uv available in THIS shell session (installer adds it for new shells).
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
else
    echo "[1/3] uv is already installed ($(uv --version))."
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv was installed but is not on your PATH."
    echo "       Open a new terminal (or run: export PATH=\"\$HOME/.local/bin:\$PATH\")"
    echo "       and re-run ./install.sh"
    exit 1
fi

# --- 2. Create the venv and install all dependencies ----------------------
echo "[2/3] Installing dependencies (this downloads PyTorch, so give it a minute)..."
uv sync

# --- 3. Register the Jupyter kernel ---------------------------------------
echo "[3/3] Registering the 'Python (rl-course)' Jupyter kernel..."
uv run python -m ipykernel install --user \
    --name rl-course --display-name "Python (rl-course)"

echo
echo "=================================================================="
echo " All set!"
echo
echo " Launch the notebooks with:"
echo "     uv run jupyter lab"
echo
echo " Then open any assignment notebook and select the"
echo " 'Python (rl-course)' kernel (top-right corner)."
echo "=================================================================="
