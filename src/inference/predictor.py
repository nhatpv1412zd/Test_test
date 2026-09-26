"""YOLO licence-plate detector.

The model is loaded lazily on first use so that importing this module (or the
FastAPI app) never triggers a model download. Each detection now carries its
``class_id`` so the caller can decide which classes are licence plates.
"""

from typing import List, NamedTuple, Optional

import numpy as np

from ..utils.config import MODEL_PATH


class Detection(NamedTuple):
    """A single YOLO detection."""

    xyxy: List[float]
    confidence: float
    class_id: int


class PlateDetector:
    """Thin wrapper around an Ultralytics YOLO model."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = str(model_path) if model_path else MODEL_PATH
        self._model = None

    @property
    def model(self):
        """Return the (lazily created) YOLO model instance."""
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
        return self._model

    def detect(self, image: np.ndarray) -> List[Detection]:
        """Run detection and return ``Detection`` objects.

        The ``results -> boxes`` iteration order matches the original backend,
        so the API response order is preserved.
        """
        results = self.model(image)
        detections: List[Detection] = []
        for result in results:
            for box in result.boxes:
                detections.append(
                    Detection(
                        xyxy=box.xyxy[0].tolist(),
                        confidence=float(box.conf[0]),
                        class_id=int(box.cls[0]),
                    )
                )
        return detections


__all__ = ["Detection", "PlateDetector"]
