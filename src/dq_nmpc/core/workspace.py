"""Pure path resolution — single source of truth for all project paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MUJOCO_ROOT = PROJECT_ROOT / "deps" / "mujoco_quadrotor"
BUILD_DIR = MUJOCO_ROOT / "build_standalone"
CORE_BIN = BUILD_DIR / "quadrotor_sim_core"
MODEL_PATH = MUJOCO_ROOT / "deps" / "model" / "mujoco" / "drone.xml"


def project_root() -> Path:
    """Return the absolute path of the repository root.

    This is the single authoritative function for project root resolution.
    """
    return PROJECT_ROOT


def mujoco_root() -> Path:
    """Return the path to the MuJoCo quadrotor simulator submodule."""
    return MUJOCO_ROOT


def build_dir() -> Path:
    """Return the xmake build directory path."""
    return BUILD_DIR


def core_bin() -> Path:
    """Return the path to the quadrotor_sim_core binary."""
    return CORE_BIN


def model_path() -> Path:
    """Return the path to drone.xml MuJoCo model."""
    return MODEL_PATH
