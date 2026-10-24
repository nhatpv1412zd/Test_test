"""End-to-end smoke test.

Skipped unless ``ultralytics``/``easyocr`` are installed and the weights/sample
image are present, so the default test run never downloads models.
"""

import pytest

pytest.importorskip("ultralytics")
pytest.importorskip("easyocr")

import cv2

from src.utils.paths import ASSETS_DIR, DEFAULT_MODEL_PATH

SAMPLE = ASSETS_DIR / "samples" / "z7659940505816_1c2add17a54ecb849b85fbb059965480.jpg"


@pytest.mark.skipif(not DEFAULT_MODEL_PATH.is_file(), reason="weights/best.pt missing")
@pytest.mark.skipif(not SAMPLE.is_file(), reason="sample image missing")
def test_end_to_end_detection_and_ocr():
    from src.inference.postprocessing import PlateOCR
    from src.inference.predictor import PlateDetector
    from src.inference.preprocessing import crop_plate, preprocess_plate

    image = cv2.imread(str(SAMPLE))
    assert image is not None

    detections = PlateDetector().detect(image)
    assert len(detections) >= 1

    ocr = PlateOCR()
    for det in detections:
        assert 0.0 <= det.confidence <= 1.0
        crop = crop_plate(image, det.xyxy, padding=0.06)
        text = ocr.read_text(preprocess_plate(crop))
        assert isinstance(text, str)
