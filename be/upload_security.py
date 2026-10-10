"""Upload boundary controls; do not alter accepted images or inference settings."""

from io import BytesIO
import struct
import warnings

from PIL import Image, UnidentifiedImageError
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse


MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_BODY_BYTES = 21 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
MAX_IMAGE_EDGE = 8192
READ_CHUNK_BYTES = 64 * 1024
SUPPORTED_FORMATS = ("JPEG", "PNG", "WEBP")


class UploadValidationError(ValueError):
    """Only fixed, public messages cross the API boundary."""

    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class _BodyLimitExceeded(HTTPException):
    def __init__(self):
        super().__init__(413, "Upload body too large")


class DetectBodyLimitMiddleware:
    """Count ingress bytes before multipart parsing, without buffering the body."""

    def __init__(self, app, max_bytes=MAX_BODY_BYTES):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if (scope["type"] != "http" or scope.get("method") != "POST"
                or scope.get("path") not in {"/detect", "/detect/"}):
            await self.app(scope, receive, send)
            return

        for name, value in scope.get("headers", []):
            if name.lower() == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    continue  # Actual receive counting remains authoritative.
                if declared > self.max_bytes:
                    await JSONResponse({"detail": "Upload body too large"}, 413)(scope, receive, send)
                    return

        received = 0
        response_started = False

        async def bounded_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    # HTTPException survives FastAPI's multipart error handling.
                    # The parser unwinds and closes its temporary files.
                    raise _BodyLimitExceeded()
            return message

        async def track_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, bounded_receive, track_send)
        except _BodyLimitExceeded:
            # FastAPI normally handles this itself. Also support a bare ASGI app,
            # but never send a second response after downstream has started one.
            if response_started:
                raise
            await JSONResponse({"detail": "Upload body too large"}, 413)(scope, receive, send)


async def read_upload(file, max_bytes=None):
    """Read at most the file budget plus one sentinel byte, not read-all."""
    limit = MAX_FILE_BYTES if max_bytes is None else max_bytes
    chunks = []
    total = 0
    while True:
        chunk = await file.read(min(READ_CHUNK_BYTES, limit - total + 1))
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise UploadValidationError(413, "Image file too large")
        chunks.append(chunk)
    if total == 0:
        raise UploadValidationError(400, "Empty image file")
    return b"".join(chunks)


def _check_dimensions(width, height):
    if width <= 0 or height <= 0:
        raise UploadValidationError(400, "Invalid image")
    if (width > MAX_IMAGE_EDGE or height > MAX_IMAGE_EDGE
            or width * height > MAX_IMAGE_PIXELS):
        raise UploadValidationError(413, "Image dimensions too large")


def _check_webp_canvas(contents):
    """Read only fixed WebP canvas fields before Pillow's native AnimDecoder.

    Pillow opens WebP through an animation decoder, so cap the RIFF/VP8 canvas
    before constructing it. This is a metadata fence, not a pixel decoder.
    """
    kind = contents[12:16]
    size = int.from_bytes(contents[16:20], "little")
    if len(contents) < 20 or size > len(contents) - 20:
        raise UploadValidationError(400, "Invalid image")
    if kind == b"VP8X" and size >= 10:
        width = int.from_bytes(contents[24:27], "little") + 1
        height = int.from_bytes(contents[27:30], "little") + 1
    elif kind == b"VP8L" and size >= 5 and contents[20] == 0x2f:
        bits = int.from_bytes(contents[21:25], "little")
        width, height = (bits & 0x3fff) + 1, ((bits >> 14) & 0x3fff) + 1
    elif kind == b"VP8 " and size >= 10 and contents[23:26] == b"\x9d\x01\x2a":
        width = int.from_bytes(contents[26:28], "little") & 0x3fff
        height = int.from_bytes(contents[28:30], "little") & 0x3fff
    else:
        raise UploadValidationError(400, "Invalid image")
    _check_dimensions(width, height)


def validate_image_header(contents):
    """Probe/verify supported content before OpenCV allocates the BGR image."""
    if not contents:
        raise UploadValidationError(400, "Empty image file")
    recognized = (contents.startswith(b"\xff\xd8")
                  or contents.startswith(b"\x89PNG\r\n\x1a\n")
                  or (contents.startswith(b"RIFF") and contents[8:12] == b"WEBP"))
    if not recognized:
        raise UploadValidationError(415, "Unsupported image format")
    # Pillow's JPEG verify is metadata-only, while OpenCV may recover a
    # truncated JPEG. A complete JPEG must contain an end-of-image marker.
    if contents.startswith(b"\xff\xd8") and contents.find(b"\xff\xd9", 2) == -1:
        raise UploadValidationError(400, "Invalid image")
    if contents.startswith(b"RIFF"):
        _check_webp_canvas(contents)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(contents), formats=SUPPORTED_FORMATS) as image:
                if image.format not in SUPPORTED_FORMATS:
                    raise UploadValidationError(415, "Unsupported image format")
                _check_dimensions(*image.size)
                # Verify the encoded stream where supported, without load/convert
                # or re-encoding. The existing OpenCV decoder remains authoritative
                # for pixels and must still succeed after this metadata gate.
                image.verify()
    except (Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise UploadValidationError(413, "Image dimensions too large") from None
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, EOFError, struct.error) as exc:
        if isinstance(exc, UploadValidationError):
            raise
        raise UploadValidationError(400, "Invalid image") from None


def validate_decoded_image(image):
    """Defense after the existing OpenCV IMREAD_COLOR decode."""
    if image is None or not image.size or image.ndim != 3 or image.shape[2] != 3:
        raise UploadValidationError(400, "Invalid image")
    _check_dimensions(image.shape[1], image.shape[0])
