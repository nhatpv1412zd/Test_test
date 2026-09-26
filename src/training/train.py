"""YOLOv8 licence-plate training entry point.

Extracted from the original notebook (``notebooks/*.ipynb``) so that training
is reproducible from the command line instead of relying on notebook state.
The training logic and hyper-parameters are unchanged.

Usage (from the repository root):

    python -m src.training.train --data data/kaggle/data.yaml --epochs 50 --batch 8 --device cpu
    python -m src.training.train --data data/custom/data.yaml --epochs 50 --batch 8 --device 0

``--data`` defaults to ``data/kaggle/data.yaml``.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    TRAIN_BATCH,
    TRAIN_DEVICE,
    TRAIN_EPOCHS,
    TRAIN_IMGSZ,
    TRAIN_MODEL,
)
from src.utils.paths import DATA_YAML, EXPERIMENTS_DIR


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 licence-plate detector.")
    parser.add_argument("--model", default=TRAIN_MODEL, help="Base weights (default: yolov8n.pt).")
    parser.add_argument("--data", default=str(DATA_YAML), help="Dataset YAML path.")
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=TRAIN_IMGSZ)
    parser.add_argument("--batch", type=int, default=TRAIN_BATCH)
    parser.add_argument("--device", default=TRAIN_DEVICE, help="'cpu', '0', '0,1', ...")
    parser.add_argument("--project", default=str(EXPERIMENTS_DIR), help="Output project directory.")
    parser.add_argument("--name", default="exp001-baseline", help="Run name inside --project.")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    from ultralytics import YOLO

    print("Loading YOLOv8 framework...")
    model = YOLO(args.model)

    print("Training licence-plate detector...")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )

    print(f"Training finished. Check {args.project}/{args.name}/weights/best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
