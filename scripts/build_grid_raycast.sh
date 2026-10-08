#!/bin/bash
# Build villa's vc.grid_raycast extension (unmodified source) and place it in vcshim/vc/.
# Env: PYTHON (default python), VILLA_DIR (default $WINDEVAL_DATA/villa), WINDEVAL_DATA (default ~/kaggle-data/vesuvius).
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python}"
DATA="${WINDEVAL_DATA:-$HOME/kaggle-data/vesuvius}"
export VILLA_DIR="${VILLA_DIR:-$DATA/villa}"
BUILD="$DATA/build/grid_raycast"
PATH="$(dirname "$("$PY" -c 'import sys; print(sys.executable)')"):$PATH" cmake -S "$HERE/native" -B "$BUILD" -G Ninja -DPython_EXECUTABLE="$("$PY" -c 'import sys; print(sys.executable)')"
cmake --build "$BUILD"
cp "$BUILD"/grid_raycast*.so "$HERE/vcshim/vc/"
echo "installed $(ls "$HERE"/vcshim/vc/grid_raycast*.so)"
