"""Small stdlib-only helpers for lightweight tests, never production startup."""

from contextlib import contextmanager
import importlib.abc
import os
import sys


# Explicit test configuration; no developer-specific values or private .env reads.
# Production defaults and production config.py remain unchanged.
TEST_ENVIRONMENT = {
    "MODEL_PATH": "weights/best.pt",
    "OCR_LANGUAGES": "en",
    "OCR_USE_GPU": "false",
    "OCR_ALLOWLIST": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.",
    "OCR_UPSCALE": "2",
    "OCR_CRAFT_MAX_PIXELS": "2097152",
    "OCR_CRAFT_WORKSPACE_FACTOR": "4",
    "OCR_MEMORY_RESERVE_BYTES": "134217728",
    "OCR_MIN_CONFIDENCE": "0.25",
    "PLATE_CLASS_IDS": "0",
    "CROP_PADDING": "0.1",
    "OCR_USE_CLAHE": "true",
    "OCR_USE_BILATERAL": "false",
    "OCR_USE_DESKEW": "true",
    "OCR_TWO_LINE_ASPECT": "1.9",
    "OCR_SPLIT_TWO_LINE": "true",
    "API_HOST": "127.0.0.1",
    "API_PORT": "8000",
    "CORS_ORIGINS": "*",
    "LOG_LEVEL": "INFO",
    "TRAIN_MODEL": "yolov8n.pt",
    "TRAIN_EPOCHS": "50",
    "TRAIN_IMGSZ": "640",
    "TRAIN_BATCH": "8",
    "TRAIN_DEVICE": "cpu",
    "PYTHON_DOTENV_DISABLED": "1",
}


@contextmanager
def isolated_test_environment():
    saved = {name: os.environ.get(name) for name in TEST_ENVIRONMENT}
    os.environ.update(TEST_ENVIRONMENT)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class ModelImportGuard(importlib.abc.MetaPathFinder):
    forbidden = frozenset({"ultralytics", "easyocr", "torch", "torchvision"})

    def __init__(self):
        self.attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.forbidden:
            self.attempts.append(fullname)
            raise ImportError("Real model imports forbidden in lightweight tests: " + fullname)

    def loaded_modules(self):
        return sorted(name for name in sys.modules if name.split(".")[0] in self.forbidden)
