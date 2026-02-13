#!/usr/bin/env bash
set -euo pipefail

PORT_VALUE="${PORT:-10000}"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "No Python runtime found in PATH" >&2
  exit 1
fi

echo "Using ${PYTHON_BIN}"
echo "Starting API on port ${PORT_VALUE}"
exec "${PYTHON_BIN}" -m uvicorn src.api.main:app --host 0.0.0.0 --port "${PORT_VALUE}" --log-level info
