"""Runtime configuration.

Values come from environment variables (optionally loaded from a local
``.env`` file) with safe defaults that mirror the original hard-coded
behaviour of the application.
"""

import os

from .paths import DEFAULT_MODEL_PATH, PROJECT_ROOT


def _load_dotenv() -> None:
    """Load a local ``.env`` file when python-dotenv is available."""
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except Exception:
        # python-dotenv is optional; environment variables still work.
        pass


_load_dotenv()


def _as_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_list(value: str, default):
    if not value:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


MODEL_PATH = os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))

# EasyOCR (language ['en'] mirrors the original setup; CPU by default).
OCR_LANGUAGES = _as_list(os.getenv("OCR_LANGUAGES", "en"), ["en"])
OCR_USE_GPU = _as_bool(os.getenv("OCR_USE_GPU", "false"), False)
OCR_ALLOWLIST = os.getenv("OCR_ALLOWLIST", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.")
OCR_UPSCALE = float(os.getenv("OCR_UPSCALE", "2"))
# Budget AFTER EasyOCR's aspect resize and 32-pixel padding. A 64-channel
# float32 VGG activation then uses at most 512 MiB (not a total-RAM guarantee).
# Smaller inputs retain EasyOCR's default canvas and original pixels.
OCR_CRAFT_MAX_PIXELS = int(os.getenv("OCR_CRAFT_MAX_PIXELS", str(2 * 1024 * 1024)))
# CPU admission budget above already-loaded models, calibrated from A/B peak
# private memory. Windows commit headroom matters even with free physical RAM.
OCR_CRAFT_WORKSPACE_FACTOR = int(os.getenv("OCR_CRAFT_WORKSPACE_FACTOR", "4"))
OCR_MEMORY_RESERVE_BYTES = int(os.getenv("OCR_MEMORY_RESERVE_BYTES", str(128 * 1024 * 1024)))

# Only detections above this confidence are sent to OCR (drops noise boxes).
# Kept low (0.25) so real plates are not dropped; raise if too many false boxes.
OCR_MIN_CONFIDENCE = float(os.getenv("OCR_MIN_CONFIDENCE", "0.25"))

# Preprocessing toggles applied before EasyOCR (Otsu was removed on purpose).
OCR_USE_CLAHE = _as_bool(os.getenv("OCR_USE_CLAHE", "true"), True)
OCR_USE_BILATERAL = _as_bool(os.getenv("OCR_USE_BILATERAL", "false"), False)
OCR_USE_DESKEW = _as_bool(os.getenv("OCR_USE_DESKEW", "true"), True)

# A plate is treated as two-line (motorcycle) when W/H is below this ratio.
OCR_TWO_LINE_ASPECT = float(os.getenv("OCR_TWO_LINE_ASPECT", "1.9"))
# Split a two-line plate into two halves and OCR each half separately.
OCR_SPLIT_TWO_LINE = _as_bool(os.getenv("OCR_SPLIT_TWO_LINE", "true"), True)

# YOLO class ids treated as licence plates.
# The dataset was merged to a single class, so the retrained model emits class 0.
PLATE_CLASS_IDS = {int(x) for x in _as_list(os.getenv("PLATE_CLASS_IDS", "0"), ["0"])}

# Extra margin around the detected box before cropping (fraction of box size).
# 0.1 adds a ~10% margin so characters close to the box edge are not cut off.
# A negative value crops inward, 0.0 uses the raw box.
CROP_PADDING = float(os.getenv("CROP_PADDING", "0.1"))

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
CORS_ORIGINS = _as_list(os.getenv("CORS_ORIGINS", "*"), ["*"])

# Default training settings extracted from the original notebook.
TRAIN_MODEL = os.getenv("TRAIN_MODEL", "yolov8n.pt")
TRAIN_EPOCHS = int(os.getenv("TRAIN_EPOCHS", "50"))
TRAIN_IMGSZ = int(os.getenv("TRAIN_IMGSZ", "640"))
TRAIN_BATCH = int(os.getenv("TRAIN_BATCH", "8"))
TRAIN_DEVICE = os.getenv("TRAIN_DEVICE", "cpu")

__all__ = [
    "MODEL_PATH",
    "OCR_LANGUAGES",
    "OCR_USE_GPU",
    "OCR_ALLOWLIST",
    "OCR_UPSCALE",
    "OCR_CRAFT_MAX_PIXELS",
    "OCR_CRAFT_WORKSPACE_FACTOR",
    "OCR_MEMORY_RESERVE_BYTES",
    "OCR_MIN_CONFIDENCE",
    "OCR_USE_CLAHE",
    "OCR_USE_BILATERAL",
    "OCR_USE_DESKEW",
    "OCR_TWO_LINE_ASPECT",
    "OCR_SPLIT_TWO_LINE",
    "PLATE_CLASS_IDS",
    "CROP_PADDING",
    "API_HOST",
    "API_PORT",
    "CORS_ORIGINS",
    "TRAIN_MODEL",
    "TRAIN_EPOCHS",
    "TRAIN_IMGSZ",
    "TRAIN_BATCH",
    "TRAIN_DEVICE",
]
