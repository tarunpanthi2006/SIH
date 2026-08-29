#!/usr/bin/env bash
# ==============================================================
# SatQuery — Environment Setup
# ==============================================================
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== SatQuery Setup ==="

# 1. Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi

# 2. Activate
source .venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# 5. Create directories
mkdir -p uploads artifacts

# 6. Copy .env if not present
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example"
fi

echo ""
echo "=== Setup complete ==="
echo "Activate with:  source .venv/bin/activate"
echo "Run with:       bash scripts/run_backend.sh"
