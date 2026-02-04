#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_FILE="$ROOT_DIR/patches/mlflow-compare-parents.patch"
VENV_MLFLOW="$ROOT_DIR/.venv/lib/python3.12/site-packages/mlflow/server/__init__.py"
BACKUP_DIR="$ROOT_DIR/patches/backup"

if [[ ! -f "$PATCH_FILE" ]]; then
  echo "Patch file not found: $PATCH_FILE" >&2
  exit 1
fi

if [[ ! -f "$VENV_MLFLOW" ]]; then
  echo "Target file not found: $VENV_MLFLOW" >&2
  exit 1
fi

mkdir -p "$BACKUP_DIR"
BACKUP_FILE="$BACKUP_DIR/__init__.py.$(date +%Y%m%d_%H%M%S)"
cp "$VENV_MLFLOW" "$BACKUP_FILE"

rm -f "$VENV_MLFLOW"
patch -p0 < "$PATCH_FILE"

echo "Patched MLflow server __init__.py"
