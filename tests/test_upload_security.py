"""Small upload-boundary tests: no YOLO/EasyOCR or resource-exhaustion inputs."""

import asyncio
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from be import upload_security as security
from be.main import app
from be.routes import detection
from be.inference_admission import admission_for


def encoded(fmt="PNG"):
    buffer = BytesIO()
    Image.new("RGB", (12, 8), (231, 37, 12)).save(buffer, fmt)
    return buffer.getvalue()


@pytest.fixture
def spies(monkeypatch):
    service = Mock()
    service.detect_image.return_value = {"results": []}
    decoder = Mock(wraps=detection.decode_image)
    monkeypatch.setattr(detection, "_service", service)
    monkeypatch.setattr(detection, "decode_image", decoder)
    return service, decoder


def post(contents, mime="image/png", filename="test.png", target=app):
    with TestClient(target) as client:
        return client.post("/detect", files={"file": (filename, contents, mime)})


def small_app(body_limit):
    target = FastAPI()
    target.add_middleware(security.DetectBodyLimitMiddleware, max_bytes=body_limit)
    target.include_router(detection.router)
    return target


def multipart(contents):
    head = (b"--r01\r\nContent-Disposition: form-data; name=\"file\"; "
            b"filename=\"image.png\"\r\nContent-Type: image/png\r\n\r\n")
    return head, contents, b"\r\n--r01--\r\n"


def asgi_request(target, chunks, headers=(), path="/detect"):
    """Send only tiny, explicitly supplied ASGI events; never use a live server."""
    consumed = []
    sent = []
    events = [dict(chunk) if isinstance(chunk, dict) else {
        "type": "http.request", "body": chunk, "more_body": i < len(chunks)-1
    } for i, chunk in enumerate(chunks)]

    async def run():
        async def receive():
            if not events:
                return {"type": "http.disconnect"}
            event = events.pop(0)
            consumed.append(event)
            return event

        async def send(message):
            sent.append(message.copy())

        await target({"type": "http", "asgi": {"version": "3.0"},
                      "http_version": "1.1", "scheme": "http", "method": "POST",
                      "path": path, "raw_path": path.encode(), "root_path": "",
                      "query_string": b"", "headers": list(headers),
                      "server": ("test", 80), "client": ("test", 1234)}, receive, send)

    try:
        asyncio.run(run())
    finally:
        # Bare ASGI calls do not invoke FastAPI lifespan, including on failure.
        if isinstance(target, FastAPI):
            admission = admission_for(target)
            pool = admission._executor
            admission.shutdown()
            assert admission._executor is None
            if pool is not None:
                assert all(not thread.is_alive() for thread in pool._threads)
    assert sum(m["type"] == "http.response.start" for m in sent) == 1
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    return status, sent, consumed


@pytest.mark.parametrize("fmt,mime", [("JPEG", "image/jpeg"), ("PNG", "image/png"), ("WEBP", "image/webp")])
def test_supported_images_decode_once_and_keep_success_schema(fmt, mime, spies):
    service, decoder = spies
    response = post(encoded(fmt), mime)
    assert response.status_code == 200
    assert response.json() == {"results": []}
    decoder.assert_called_once()
    service.detect_image.assert_called_once()
    image = service.detect_image.call_args.args[0]
    assert image.shape == (8, 12, 3)
    assert image.dtype == np.uint8
    if fmt == "PNG":
        assert image[0, 0].tolist() == [12, 37, 231]  # Original OpenCV BGR.


@pytest.mark.parametrize("mime", ["image/jpeg", "text/plain", "application/octet-stream", None])
def test_content_not_filename_or_client_mime_is_authoritative(mime, spies):
    response = post(encoded(), mime, "../../not-a-picture.exe")
    assert response.status_code == 200
    spies[0].detect_image.assert_called_once()


def test_missing_file_remains_422(spies):
    with TestClient(app) as client:
        response = client.post("/detect")
    assert response.status_code == 422
    spies[0].detect_image.assert_not_called()
    spies[1].assert_not_called()


def test_text_field_instead_of_upload_remains_422(spies):
    with TestClient(app) as client:
        response = client.post("/detect", data={"file": "not-an-upload"})
    assert response.status_code == 422
    spies[0].detect_image.assert_not_called()
    spies[1].assert_not_called()


def test_approved_policy_constants():
    assert security.MAX_FILE_BYTES == 20_971_520
    assert security.MAX_BODY_BYTES == 22_020_096
    assert security.MAX_IMAGE_EDGE == 8192
    assert security.MAX_IMAGE_PIXELS == 16_000_000
    assert security.SUPPORTED_FORMATS == ("JPEG", "PNG", "WEBP")


@pytest.mark.parametrize("contents,status", [
    (b"", 400), (b"fake-image", 415), (b"<script>example</script>", 415),
    (encoded("GIF"), 415), (b"\xff\xd8broken", 400),
    (encoded("JPEG")[:-2], 400),
    (b"\x89PNG\r\n\x1a\nbroken", 400), (b"RIFF\x04\x00\x00\x00WEBPbroken", 400),
])
def test_rejected_content_never_reaches_decoder_or_inference(contents, status, spies):
    response = post(contents, "image/jpeg", "fake.jpg")
    assert response.status_code == status
    assert set(response.json()) == {"detail"}
    spies[0].detect_image.assert_not_called()
    spies[1].assert_not_called()


def test_corrupt_png_crc_is_rejected_before_decoder(spies):
    contents = bytearray(encoded())
    # Damage an IDAT CRC without allocating additional image data.
    position = contents.index(b"IDAT")
    length = int.from_bytes(contents[position-4:position], "big")
    contents[position+4+length] ^= 1
    assert post(bytes(contents)).status_code == 400
    spies[1].assert_not_called()
    spies[0].detect_image.assert_not_called()


@pytest.mark.parametrize("kind,payload", [
    (b"VP8X", b"\x00"*4 + (8192).to_bytes(3, "little") + b"\x00"*3),
    (b"VP8L", b"\x2f" + (8192).to_bytes(4, "little")),
    (b"VP8 ", b"\x00"*3 + b"\x9d\x01\x2a" + (8193).to_bytes(2, "little") + (1).to_bytes(2, "little")),
])
def test_webp_canvas_is_bounded_before_native_header_plugin(monkeypatch, spies, kind, payload):
    chunk = kind + len(payload).to_bytes(4, "little") + payload
    if len(payload) % 2:
        chunk += b"\x00"
    contents = b"RIFF" + (len(chunk)+4).to_bytes(4, "little") + b"WEBP" + chunk
    opener = Mock(side_effect=AssertionError("must not initialize native WebP decoder"))
    monkeypatch.setattr(security.Image, "open", opener)
    assert post(contents, "image/webp").status_code == 413
    opener.assert_not_called()
    spies[1].assert_not_called()
    spies[0].detect_image.assert_not_called()


@pytest.mark.parametrize("variant", ["lossless", "alpha"])
def test_webp_lossless_and_extended_formats_remain_supported(spies, variant):
    buffer = BytesIO()
    image = Image.new("RGBA" if variant == "alpha" else "RGB", (12, 8),
                      (231, 37, 12, 128) if variant == "alpha" else (231, 37, 12))
    image.save(buffer, "WEBP", lossless=variant == "lossless")
    assert buffer.getvalue()[12:16] == (b"VP8L" if variant == "lossless" else b"VP8X")
    assert post(buffer.getvalue(), "image/webp").status_code == 200
    spies[0].detect_image.assert_called_once()


@pytest.mark.parametrize("difference,status", [(0, 200), (-1, 413)])
def test_file_budget_exact_boundary_with_real_small_multipart(monkeypatch, spies, difference, status):
    contents = encoded()
    monkeypatch.setattr(security, "MAX_FILE_BYTES", len(contents)+difference)
    assert post(contents).status_code == status
    assert spies[0].detect_image.call_count == (1 if status == 200 else 0)
    assert spies[1].call_count == (1 if status == 200 else 0)


def test_chunk_reader_stops_at_budget_plus_one():
    class Stream:
        position = 0
        sizes = []
        async def read(self, size):
            self.sizes.append(size)
            count = min(size, 3)
            self.position += count
            return b"x" * count
    stream = Stream()
    with pytest.raises(security.UploadValidationError) as error:
        asyncio.run(security.read_upload(stream, max_bytes=8))
    assert error.value.status_code == 413
    assert stream.position == 9
    assert all(0 < size <= security.READ_CHUNK_BYTES for size in stream.sizes)


def test_declared_oversized_body_rejected_before_receive_or_spool(monkeypatch, spies):
    import starlette.formparsers as parsers
    factory = Mock(side_effect=AssertionError("must not parse multipart"))
    monkeypatch.setattr(parsers, "SpooledTemporaryFile", factory)
    status, _, consumed = asgi_request(small_app(32), [b"unused"],
        [(b"content-length", b"33"), (b"content-type", b"multipart/form-data; boundary=r01")])
    assert status == 413
    assert consumed == []
    factory.assert_not_called()
    spies[0].detect_image.assert_not_called()


@pytest.mark.parametrize("declared", [None, b"1", b"not-an-integer"])
def test_stream_body_limit_before_parser_with_cleanup(monkeypatch, spies, declared):
    import starlette.formparsers as parsers
    real_factory = parsers.SpooledTemporaryFile
    files = []
    def factory(*args, **kwargs):
        result = real_factory(*args, **kwargs)
        files.append(result)
        return result
    monkeypatch.setattr(parsers, "SpooledTemporaryFile", factory)
    head, _, tail = multipart(b"")
    headers = [(b"content-type", b"multipart/form-data; boundary=r01")]
    if declared is not None:
        headers.append((b"content-length", declared))
    status, _, consumed = asgi_request(small_app(len(head)+4),
        [head+b"abc", b"de", tail+b"not-read"], headers)
    assert status == 413
    assert len(consumed) == 2
    assert files and all(file.closed for file in files)
    spies[0].detect_image.assert_not_called()
    spies[1].assert_not_called()


@pytest.mark.parametrize("difference,status", [(0, 200), (-1, 413)])
def test_total_multipart_boundary_without_content_length(spies, difference, status):
    body = b"".join(multipart(encoded()))
    actual, _, _ = asgi_request(small_app(len(body)+difference), [body[:-1], body[-1:]],
        [(b"content-type", b"multipart/form-data; boundary=r01")])
    assert actual == status
    assert spies[0].detect_image.call_count == (1 if status == 200 else 0)


def test_unused_parts_still_count_toward_body_limit(spies):
    body = (b"--r01\r\nContent-Disposition: form-data; name=\"unused\"\r\n\r\n"
            b"xxxxxxxxxxxxxxxx\r\n--r01--\r\n")
    status, _, _ = asgi_request(small_app(len(body)-1), [body],
        [(b"content-type", b"multipart/form-data; boundary=r01")])
    assert status == 413
    spies[0].detect_image.assert_not_called()


@pytest.mark.parametrize("headers,body", [
    ([(b"content-type", b"multipart/form-data")], b"missing boundary"),
    ([(b"content-type", b"multipart/form-data; boundary=r01")], b"bad multipart"),
    ([(b"content-type", b"multipart/form-data; boundary=r01")], b"--r01\r\ninvalid\r\n"),
])
def test_malformed_multipart_is_4xx_not_double_response(headers, body, spies):
    status, _, _ = asgi_request(app, [body], headers)
    assert status in {400, 422}
    spies[0].detect_image.assert_not_called()
    spies[1].assert_not_called()


def test_disconnect_closes_spool_without_inference(monkeypatch, spies):
    import starlette.formparsers as parsers
    real = parsers.SpooledTemporaryFile
    files = []
    def factory(*args, **kwargs):
        result = real(*args, **kwargs)
        files.append(result)
        return result
    monkeypatch.setattr(parsers, "SpooledTemporaryFile", factory)
    head, _, _ = multipart(b"")
    status, _, _ = asgi_request(app, [head+b"abc", {"type": "http.disconnect"}],
        [(b"content-type", b"multipart/form-data; boundary=r01")])
    assert status == 400
    assert files and all(file.closed for file in files)
    spies[0].detect_image.assert_not_called()


@pytest.mark.parametrize("width,height,status", [
    (8193, 1, 413), (1, 8193, 413), (4001, 4000, 413), (0, 8, 400),
    (8192, 1, 200), (1, 8192, 200), (4000, 4000, 200),
])
def test_metadata_budget_checked_before_full_decode(monkeypatch, spies, width, height, status):
    class Metadata:
        format = "PNG"
        size = (width, height)
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def verify(self):
            assert status == 200  # Oversized input must not reach this step.
    monkeypatch.setattr(security.Image, "open", lambda *args, **kwargs: Metadata())
    # Only allocate a tiny decoded stand-in for accepted metadata boundaries.
    spies[1].side_effect = lambda contents: np.zeros((2, 3, 3), np.uint8)
    response = post(encoded())
    assert response.status_code == status
    assert spies[1].call_count == (1 if status == 200 else 0)
    assert spies[0].detect_image.call_count == (1 if status == 200 else 0)


@pytest.mark.parametrize("kind,status", [("none", 400), ("empty", 400), ("channels", 400),
                                         ("error", 400), ("edge", 413), ("pixels", 413)])
def test_defensive_decode_check_never_passes_bad_image(monkeypatch, spies, caplog, kind, status):
    values = {"none": None, "empty": np.zeros((0, 0, 3), np.uint8),
              "channels": np.zeros((2, 3), np.uint8),
              "edge": SimpleNamespace(size=1, ndim=3, shape=(8193, 1, 3)),
              "pixels": SimpleNamespace(size=1, ndim=3, shape=(4001, 4000, 3))}
    marker = "INTERNAL_TEST_PATH_NEVER_PUBLIC"
    if kind == "error":
        spies[1].side_effect = RuntimeError(marker)
    else:
        spies[1].side_effect = lambda contents: values[kind]
    response = post(encoded())
    assert response.status_code == status
    assert marker not in response.text
    assert marker not in caplog.text
    spies[0].detect_image.assert_not_called()


def test_unexpected_inference_error_remains_generic_500(spies):
    marker = "INTERNAL_TEST_ONLY_NOT_A_SECRET"
    spies[0].detect_image.side_effect = RuntimeError(marker)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/detect", files={"file": ("test.png", encoded(), "image/png")})
    assert response.status_code == 500
    assert marker not in response.text
    assert "Traceback" not in response.text


def test_yolo_runtime_failure_remains_empty_success(monkeypatch):
    from be.services.detection_service import DetectionService
    detector = Mock()
    detector.detect.side_effect = RuntimeError("injected detector failure")
    ocr = Mock()
    monkeypatch.setattr(detection, "_service", DetectionService(detector=detector, ocr=ocr))
    response = post(encoded())
    assert response.status_code == 200
    assert response.json() == {"results": []}
    ocr.read_text.assert_not_called()


def test_bare_asgi_limit_fallback_and_other_paths():
    async def consumer(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})
    target = security.DetectBodyLimitMiddleware(consumer, max_bytes=2)
    assert asgi_request(target, [b"abc"])[0] == 413
    assert asgi_request(target, [b"abc"], path="/other")[0] == 204
