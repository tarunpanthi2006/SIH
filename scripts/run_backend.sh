#!/usr/bin/env bash
# ==============================================================
# SatQuery — Run Backend (development)
# ==============================================================
set -euo pipefail

cd "$(dirname "$0")/.."

# Load .env if present
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Default to mock mode
export SATQUERY_MOCK_MODE="${SATQUERY_MOCK_MODE:-true}"

echo "Starting SatQuery backend (MOCK_MODE=${SATQUERY_MOCK_MODE})..."

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

uvicorn backend.main:app \
    --reload \
    --host 0.0.0.0 \
    --port 8000
