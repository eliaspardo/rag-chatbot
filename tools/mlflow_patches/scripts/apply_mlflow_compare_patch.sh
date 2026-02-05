#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PATCH_FILE="$ROOT_DIR/tools/mlflow_patches/patches/mlflow-compare-parents.patch"
TEMPLATE_FILE="$ROOT_DIR/tools/mlflow_patches/patches/mlflow_server_init.py"

VENV_PATH="${VIRTUAL_ENV:-}"
if [[ -z "$VENV_PATH" ]]; then
  VENV_PATH="$ROOT_DIR/.venv"
fi

MLFLOW_SERVER_INIT="${MLFLOW_SERVER_INIT:-}"

PYTHON_BIN=""
if [[ -x "$VENV_PATH/bin/python" ]]; then
  PYTHON_BIN="$VENV_PATH/bin/python"
elif [[ -x "$VENV_PATH/Scripts/python.exe" ]]; then
  PYTHON_BIN="$VENV_PATH/Scripts/python.exe"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
fi

if [[ -n "$MLFLOW_SERVER_INIT" ]]; then
  VENV_MLFLOW="$MLFLOW_SERVER_INIT"
elif [[ -n "$PYTHON_BIN" ]]; then
  SITE_PACKAGES="$("$PYTHON_BIN" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  VENV_MLFLOW="$SITE_PACKAGES/mlflow/server/__init__.py"
fi

if [[ -z "${VENV_MLFLOW:-}" ]]; then
  echo "Unable to determine site-packages path from venv: $VENV_PATH" >&2
  exit 1
fi

if [[ -z "${VENV_MLFLOW:-}" ]]; then
  echo "Unable to determine site-packages path from venv: $VENV_PATH" >&2
  exit 1
fi
BACKUP_DIR="$ROOT_DIR/tools/mlflow_patches/backup"

if [[ ! -f "$PATCH_FILE" && ! -f "$TEMPLATE_FILE" ]]; then
  echo "Patch/template file not found: $PATCH_FILE or $TEMPLATE_FILE" >&2
  exit 1
fi

mkdir -p "$BACKUP_DIR"
BACKUP_FILE="$BACKUP_DIR/__init__.py.$(date +%Y%m%d_%H%M%S)"
if [[ -f "$VENV_MLFLOW" ]]; then
  cp "$VENV_MLFLOW" "$BACKUP_FILE"
fi

rm -f "$VENV_MLFLOW"
mkdir -p "$(dirname "$VENV_MLFLOW")"

if [[ -f "$TEMPLATE_FILE" ]]; then
  cp "$TEMPLATE_FILE" "$VENV_MLFLOW"
else
  patch -p0 < "$PATCH_FILE"
fi

echo "Patched MLflow server __init__.py"
