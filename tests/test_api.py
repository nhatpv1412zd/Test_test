import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient


def test_detect_endpoint_contract(monkeypatch):
    from be.routes import detection

    class _StubService:
        def detect_bytes(self, contents):
            assert contents == b"fake-image"
            return {"results": [{"confidence": 0.9, "plate_number": "51A-12345"}]}

    monkeypatch.setattr(detection, "_service", _StubService())

    from be.main import app

    with TestClient(app) as client:
        response = client.post(
            "/detect",
            files={"file": ("plate.jpg", b"fake-image", "image/jpeg")},
        )

    assert response.status_code == 200
    assert response.json() == {"results": [{"confidence": 0.9, "plate_number": "51A-12345"}]}


@pytest.mark.parametrize('outcome,status',[
    ('runtime','ocr_runtime_error'),('empty','empty_ocr'),('partial','unrecognized_format')])
def test_public_api_shape_unchanged_but_failure_status_is_in_logs(monkeypatch,caplog,outcome,status):
    import logging
    import cv2
    import numpy as np
    from src.inference.postprocessing import PlateOCR
    from src.inference.predictor import Detection
    from be.services.detection_service import DetectionService
    from be.routes import detection
    from be.main import app
    class Detector:
        def detect(self,image):return [Detection([10,10,110,60],.9,0)]
    class Reader:
        def readtext(self,image,**kwargs):
            if outcome=='runtime':raise RuntimeError('injected allocator failure')
            if outcome=='empty':return []
            return [([[0,0],[90,0],[90,20],[0,20]],'L8953',.4)]
    ocr=PlateOCR();ocr._reader=Reader()
    monkeypatch.setattr(detection,'_service',DetectionService(detector=Detector(),ocr=ocr))
    ok,buf=cv2.imencode('.png',np.zeros((100,200,3),np.uint8));assert ok
    with caplog.at_level(logging.INFO),TestClient(app) as client:
        response=client.post('/detect',files={'file':('test.png',buf.tobytes(),'image/png')})
    assert response.status_code==200
    assert response.json()=={'results':[{'confidence':.9,'plate_number':''}]}
    assert 'status='+status in caplog.text
    if outcome=='runtime':
        assert any(record.exc_info for record in caplog.records)


def test_api_survives_memory_refusal_then_processes_next_detection(monkeypatch,caplog):
    import cv2
    import numpy as np
    from src.inference import runtime_memory
    from src.inference.postprocessing import PlateOCR
    from src.inference.predictor import Detection
    from be.services.detection_service import DetectionService
    from be.routes import detection
    from be.main import app
    headroom=iter([0,16*1024**3])
    monkeypatch.setattr(runtime_memory,'_available_commit_bytes',lambda:next(headroom))
    class Detector:
        def detect(self,image):return [Detection([10,10,90,50],.9,0),Detection([110,10,190,50],.8,0)]
    class Reader:
        calls=0
        def readtext(self,image,**kwargs):
            self.calls+=1
            return [([[0,0],[90,0],[90,20],[0,20]],'17AK04470',.9)]
    reader=Reader();ocr=PlateOCR();ocr._reader=reader
    monkeypatch.setattr(detection,'_service',DetectionService(detector=Detector(),ocr=ocr))
    ok,buf=cv2.imencode('.png',np.zeros((100,200,3),np.uint8));assert ok
    with TestClient(app) as client:
        response=client.post('/detect',files={'file':('test.png',buf.tobytes(),'image/png')})
    assert response.status_code==200
    assert response.json()=={'results':[{'confidence':.9,'plate_number':''},
                                     {'confidence':.8,'plate_number':'17AK-044.70'}]}
    assert reader.calls==1
    assert 'status=ocr_runtime_error stage=memory_admission' in caplog.text
    assert any(r.exc_info for r in caplog.records)
