#!/bin/bash
set -euo pipefail

echo ""
echo "Building NMPC with acados..."
echo ""

# --------------------------------------------------
# 1. Build acados (if not already built)
# --------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACADOS_DIR="$SCRIPT_DIR/deps/acados"

if [ ! -d "$ACADOS_DIR/build" ]; then
    echo "[INFO] Building acados from $ACADOS_DIR ..."
    mkdir -p "$ACADOS_DIR/build"
    (
        cd "$ACADOS_DIR"
        cmake -B build -DACADOS_WITH_OSQP=ON -DACADOS_PYTHON=ON
        cmake --build build
    )
    echo "[INFO] acados build complete."
else
    echo "[INFO] acados already built, skipping."
fi

# Set acados env vars
export ACADOS_SOURCE_DIR="$ACADOS_DIR"
export LD_LIBRARY_PATH="$ACADOS_DIR/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$ACADOS_DIR/interfaces/acados_template:$PYTHONPATH"

# --------------------------------------------------
# 2. Select platform type
# --------------------------------------------------
PLATFORM_TYPE="${1:-mujoco}"
echo "[INFO] Platform type: $PLATFORM_TYPE"

# --------------------------------------------------
# 3. Run NMPC code generation (generates C code)
# --------------------------------------------------
CONTROL_YAML="$SCRIPT_DIR/config/$PLATFORM_TYPE/default/dq_control.yaml"
GENERATED_DIR="$SCRIPT_DIR/c_generated_code"

echo "[INFO] Running NMPC code generation with config: $CONTROL_YAML"
python3 "$SCRIPT_DIR/src/dq_nmpc/nmpc/controller.py" "$CONTROL_YAML"

echo "[INFO] Code generation complete. Output in: $GENERATED_DIR"

# --------------------------------------------------
# 4. (Optional) Build ROS2 dq_cpp if $WS is set
# --------------------------------------------------
if [ -n "${WS:-}" ]; then
    echo "[INFO] WS=$WS detected, building dq_cpp with colcon..."

    mkdir -p "$WS/install/dq_cpp/lib"
    cp "$GENERATED_DIR"/libacados_ocp_solver_quadrotor.so "$WS/install/dq_cpp/lib/" 2>/dev/null || true

    rm -rf "$WS/src/dq_cpp/c_generated_code" 2>/dev/null || true
    cp -r "$GENERATED_DIR" "$WS/src/dq_cpp/"

    (
        cd "$WS"
        colcon build --symlink-install --packages-select dq_cpp
    )
    echo "[INFO] dq_cpp build complete."
else
    echo "[INFO] WS not set, skipping colcon dq_cpp build."
fi

echo ""
echo "NMPC build complete."
echo ""
