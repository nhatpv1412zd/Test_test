"""Detection service.

Orchestrates the full flow that used to live inside ``be/main.py``:

    bytes -> decode -> YOLO detect -> filter (class + confidence)
          -> crop (with padding) -> preprocess -> OCR -> JSON

No AI/CV logic lives here; it only coordinates ``src.inference`` components.
Every filtering step is logged so the pipeline can be traced from the terminal.
"""

import logging
from typing import Dict, List, Optional

from src.inference.postprocessing import PlateOCR, build_detection
from src.inference.predictor import PlateDetector
from src.inference.preprocessing import crop_plate, decode_image, preprocess_plate
from src.utils.config import (
    CROP_PADDING,
    OCR_MIN_CONFIDENCE,
    OCR_UPSCALE,
    PLATE_CLASS_IDS,
)

logger = logging.getLogger(__name__)


class DetectionService:
    """Run plate detection + OCR on an uploaded image."""

    def __init__(
        self,
        detector: Optional[PlateDetector] = None,
        ocr: Optional[PlateOCR] = None,
        plate_class_ids=None,
        min_confidence: Optional[float] = None,
        crop_padding: Optional[float] = None,
    ):
        self._detector = detector or PlateDetector()
        self._ocr = ocr or PlateOCR()
        self._plate_class_ids = set(plate_class_ids) if plate_class_ids is not None else set(PLATE_CLASS_IDS)
        self._min_confidence = OCR_MIN_CONFIDENCE if min_confidence is None else min_confidence
        self._crop_padding = CROP_PADDING if crop_padding is None else crop_padding

        logger.info(
            "DetectionService ready | model=%s | plate_class_ids=%s | min_confidence=%.3f | "
            "crop_padding=%s | upscale=%s",
            getattr(self._detector, "model_path", "?"),
            sorted(self._plate_class_ids) if self._plate_class_ids else "ALL",
            self._min_confidence,
            self._crop_padding,
            OCR_UPSCALE,
        )

    def detect_bytes(self, contents: bytes) -> Dict[str, List[Dict[str, object]]]:
        """Detect plates in raw image bytes and read their text."""
        try:
            image = decode_image(contents)
        except Exception as exc:
            logger.exception("decode_image failed: %s", exc)
            return {"results": []}
        if image is None:
            logger.warning("decode_image returned None - invalid/unsupported image bytes")
            return {"results": []}

        logger.info("decoded image: shape=%s", image.shape)

        try:
            raw_detections = list(self._detector.detect(image))
        except Exception as exc:
            logger.exception("YOLO detect failed: %s", exc)
            return {"results": []}
        logger.info("YOLO detected %d box(es)", len(raw_detections))

        results: List[Dict[str, object]] = []
        for index, det in enumerate(raw_detections):
            logger.info(
                "det[%d] class=%s conf=%.4f xyxy=%s",
                index,
                det.class_id,
                det.confidence,
                [round(float(v), 2) for v in det.xyxy],
            )

            # 1) class filter (empty set => accept every class)
            if self._plate_class_ids and det.class_id not in self._plate_class_ids:
                logger.info("  -> skipped: class %s not in %s", det.class_id, sorted(self._plate_class_ids))
                continue

            # 2) confidence filter
            if det.confidence < self._min_confidence:
                logger.info("  -> skipped: conf %.4f < min_confidence %.4f", det.confidence, self._min_confidence)
                continue

            # 3) crop -> deskew -> CLAHE -> (two-line split) -> OCR
            try:
                plate_image = crop_plate(image, det.xyxy, padding=self._crop_padding)
                if plate_image.size == 0:
                    logger.info("  -> skipped: empty crop from xyxy")
                    continue
                logger.info("  -> crop shape=%s (padding=%s)", plate_image.shape, self._crop_padding)

                processed = preprocess_plate(plate_image, upscale=OCR_UPSCALE)
                logger.info("  -> preprocessed shape=%s", processed.shape)

                plate_text = self._ocr.read_text(processed)
            except Exception as exc:  # one bad box must not fail the whole request
                logger.exception("  -> processing failed for det[%d]: %s", index, exc)
                continue

            if plate_text:
                logger.info("  -> OCR text=%r", plate_text)
            else:
                # An empty public string also covers rejected format/runtime
                # failure. PlateOCR logs the distinct ocr_status + traceback.
                logger.warning("  -> no plate_number emitted for det[%d]; see OCR status/exception logs", index)

            # 4) keep the detection even if the OCR text is empty (valid JSON result)
            results.append(build_detection(det.confidence, plate_text))

        if results:
            logger.info("returning %d result(s)", len(results))
        elif raw_detections:
            logger.warning(
                "returning 0 result(s) even though YOLO detected %d box(es) - every box was "
                "filtered. Check OCR_MIN_CONFIDENCE=%.3f and PLATE_CLASS_IDS=%s",
                len(raw_detections),
                self._min_confidence,
                sorted(self._plate_class_ids) if self._plate_class_ids else "ALL",
            )
        else:
            logger.warning("returning 0 result(s) - YOLO detected no boxes (model/weights or image issue)")

        return {"results": results}


__all__ = ["DetectionService"]
