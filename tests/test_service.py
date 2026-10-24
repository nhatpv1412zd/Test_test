"""Service orchestration tests (no YOLO/EasyOCR weights required)."""

import cv2
import numpy as np

from be.services.detection_service import DetectionService
from src.inference.predictor import Detection


class _StubDetector:
    def __init__(self, detections):
        self._detections = detections

    def detect(self, image):
        return self._detections


class _StubOCR:
    def read_text(self, image):
        return "51A-123.45"


def _png_bytes() -> bytes:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return buf.tobytes()


def test_service_returns_expected_schema():
    service = DetectionService(
        detector=_StubDetector([Detection([10, 10, 110, 60], 0.87, 0)]),
        ocr=_StubOCR(),
    )
    result = service.detect_bytes(_png_bytes())
    assert result == {"results": [{"confidence": 0.87, "plate_number": "51A-123.45"}]}


def test_service_handles_zero_detections():
    service = DetectionService(detector=_StubDetector([]), ocr=_StubOCR())
    assert service.detect_bytes(_png_bytes()) == {"results": []}


def test_service_preserves_confidence_as_float():
    service = DetectionService(
        detector=_StubDetector([Detection([0, 0, 10, 10], 1, 0)]),
        ocr=_StubOCR(),
    )
    confidence = service.detect_bytes(_png_bytes())["results"][0]["confidence"]
    assert isinstance(confidence, float)


def test_service_drops_low_confidence_detections():
    service = DetectionService(
        detector=_StubDetector([Detection([0, 0, 10, 10], 0.2, 0)]),
        ocr=_StubOCR(),
        min_confidence=0.5,
    )
    assert service.detect_bytes(_png_bytes()) == {"results": []}


def test_service_drops_non_plate_classes():
    service = DetectionService(
        detector=_StubDetector([Detection([0, 0, 10, 10], 0.9, 7)]),
        ocr=_StubOCR(),
        plate_class_ids={0, 1},
    )
    assert service.detect_bytes(_png_bytes()) == {"results": []}


def test_ocr_runtime_error_does_not_drop_detection_or_block_next_box(caplog):
    from src.inference.postprocessing import PlateOCR
    class Reader:
        def __init__(self):self.calls=0
        def readtext(self,image,**kwargs):
            self.calls+=1
            if self.calls==1:raise RuntimeError('injected CPU allocation error')
            return [([[0,0],[90,0],[90,20],[0,20]],'17AK04470',.9)]
    ocr=PlateOCR();ocr._reader=Reader()
    service=DetectionService(detector=_StubDetector([
        Detection([10,10,90,50],.9,0),Detection([110,10,190,50],.8,0)]),ocr=ocr)
    assert service.detect_bytes(_png_bytes())=={'results':[
        {'confidence':.9,'plate_number':''},{'confidence':.8,'plate_number':'17AK-044.70'}]}
    assert 'ocr_runtime_error' in caplog.text
    assert 'OCR returned EMPTY' not in caplog.text


def test_low_commit_call_does_not_kill_request_or_block_next_detection(monkeypatch,caplog):
    from src.inference import runtime_memory
    from src.inference.postprocessing import PlateOCR
    available=iter([0,16*1024**3])
    monkeypatch.setattr(runtime_memory,'_available_commit_bytes',lambda:next(available))
    class Reader:
        calls=0
        def readtext(self,*args,**kwargs):
            self.calls+=1
            return [([[0,0],[90,0],[90,20],[0,20]],'17AK04470',.9)]
    ocr=PlateOCR();reader=Reader();ocr._reader=reader
    service=DetectionService(detector=_StubDetector([
        Detection([10,10,90,50],.9,0),Detection([110,10,190,50],.8,0)]),ocr=ocr)
    assert service.detect_bytes(_png_bytes())=={'results':[
        {'confidence':.9,'plate_number':''},{'confidence':.8,'plate_number':'17AK-044.70'}]}
    assert reader.calls==1
    assert 'ocr_runtime_error' in caplog.text
    assert 'memory_admission' in caplog.text
