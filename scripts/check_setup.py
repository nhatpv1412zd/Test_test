"""Diagnostic check: dependencies, dataset and weights.

Usage (from the repository root):

    python scripts/check_setup.py
"""

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.paths import CUSTOM_DATA_YAML, DEFAULT_MODEL_PATH, KAGGLE_DATA_YAML, PROJECT_ROOT

REQUIRED_MODULES = ["fastapi", "uvicorn", "multipart", "ultralytics", "easyocr", "cv2", "numpy"]

REQUIRED_PATHS = {
    "kaggle dataset config": KAGGLE_DATA_YAML,
    "custom dataset config": CUSTOM_DATA_YAML,
    "inference weights": DEFAULT_MODEL_PATH,
    "backend entry point": PROJECT_ROOT / "be" / "main.py",
    "frontend": PROJECT_ROOT / "fe" / "index.html",
}

OPTIONAL_PATHS = {
    "kaggle train images": PROJECT_ROOT / "data" / "kaggle" / "images" / "train",
    "kaggle val images": PROJECT_ROOT / "data" / "kaggle" / "images" / "val",
    "custom train images": PROJECT_ROOT / "data" / "custom" / "images" / "train",
}


def main() -> int:
    ok = True

    print(f"Project root: {PROJECT_ROOT}\n")

    print("Dependencies:")
    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
            print(f"  [OK]   {name}")
        except Exception as exc:  # pragma: no cover - diagnostic path
            ok = False
            print(f"  [FAIL] {name}: {exc}")

    print("\nRequired paths:")
    for label, path in REQUIRED_PATHS.items():
        exists = path.exists()
        ok = ok and exists
        print(f"  [{'OK' if exists else 'FAIL'}] {label}: {path}")

    print("\nOptional paths:")
    for label, path in OPTIONAL_PATHS.items():
        exists = path.exists()
        print(f"  [{'OK' if exists else 'MISSING'}] {label}: {path}")

    print("\nResult:", "OK" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
