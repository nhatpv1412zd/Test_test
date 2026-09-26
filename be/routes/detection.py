"""``/detect`` endpoint."""

from fastapi import APIRouter, File, UploadFile

from ..services.detection_service import DetectionService

router = APIRouter()

# Single long-lived service; model weights are loaded lazily on first request.
_service = DetectionService()


@router.post("/detect")
async def detect_and_ocr(file: UploadFile = File(...)):
    """Receive an image and return detected licence plates with OCR text."""
    contents = await file.read()
    return _service.detect_bytes(contents)


__all__ = ["router"]
