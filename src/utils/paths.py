"""Canonical project paths.

Every path is derived from the repository root so that the project works
regardless of the current working directory or machine.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
ASSETS_DIR = PROJECT_ROOT / "assets"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Per-source dataset configs (there is no single root data.yaml anymore).
KAGGLE_DATA_YAML = DATA_DIR / "kaggle" / "data.yaml"
CUSTOM_DATA_YAML = DATA_DIR / "custom" / "data.yaml"

# Default dataset config (used by training / diagnostics).
DATA_YAML = KAGGLE_DATA_YAML
DEFAULT_MODEL_PATH = WEIGHTS_DIR / "best.pt"

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "WEIGHTS_DIR",
    "ASSETS_DIR",
    "EXPERIMENTS_DIR",
    "KAGGLE_DATA_YAML",
    "CUSTOM_DATA_YAML",
    "DATA_YAML",
    "DEFAULT_MODEL_PATH",
]
