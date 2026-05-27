#!/usr/bin/env bash
set -e

echo "=== Vehicle Detection API container startup ==="

echo "Checking model artifacts..."
uv run python -m src.cli download-weights
uv run python -m src.cli check-artifacts

echo "Starting API on 0.0.0.0:${PORT:-8000}"
exec uv run uvicorn src.api:app --host 0.0.0.0 --port "${PORT:-8000}"