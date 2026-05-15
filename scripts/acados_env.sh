#!/usr/bin/env bash
# Source this script before running dq-codegen or dq-run manually.
# Not needed when using `uv run` (CLI entry points set env vars automatically).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(realpath "$SCRIPT_DIR/..")"

export ACADOS_SOURCE_DIR="$PROJECT/deps/acados"
export LD_LIBRARY_PATH="$PROJECT/_acados_build/install/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$ACADOS_SOURCE_DIR/interfaces/acados_template:$PYTHONPATH"
