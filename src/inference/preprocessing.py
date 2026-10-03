"""Image preprocessing for licence-plate OCR.

Pipeline (revised):

    crop -> deskew/perspective correction -> grayscale
         -> optional bilateral denoise -> CLAHE -> cubic upscale

Helpers are also provided to detect a two-line (motorcycle) plate by aspect
ratio and to split it into its two line images, so the OCR step can read each
line separately.
"""

from typing import Optional, Sequence, Tuple

import cv2
import logging
import numpy as np

from ..utils.config import (
    OCR_TWO_LINE_ASPECT,
    OCR_USE_BILATERAL,
    OCR_USE_CLAHE,
    OCR_USE_DESKEW,
)

logger = logging.getLogger(__name__)

# A single CLAHE instance is reused (creating one per call is wasteful).
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE to sharpen characters / remove soft shadows."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    return _CLAHE.apply(gray)


def decode_image(contents: bytes) -> np.ndarray:
    """Decode raw upload bytes into a BGR image (as ``cv2`` expects)."""
    nparr = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def crop_plate(image: np.ndarray, xyxy: Sequence[float], padding: float = 0.0) -> np.ndarray:
    """Crop the plate region from ``image``.

    ``padding`` is a fraction of the box size added on every side (positive adds
    a margin, negative crops inward). Coordinates are clamped to the image.
    """
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    if padding:
        pad_x = (x2 - x1) * padding
        pad_y = (y2 - y1) * padding
    else:
        pad_x = pad_y = 0.0

    h, w = image.shape[:2]
    xi1 = max(0, int(round(x1 - pad_x)))
    yi1 = max(0, int(round(y1 - pad_y)))
    xi2 = min(w, int(round(x2 + pad_x)))
    yi2 = min(h, int(round(y2 + pad_y)))
    return image[yi1:yi2, xi1:xi2]


def _order_quad(points: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def _find_plate_quad(gray: np.ndarray):
    """Best-effort detection of the plate quadrilateral inside the crop."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if float(thr.mean()) > 127:  # make the plate white on black
        thr = cv2.bitwise_not(thr)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    # The plate must dominate the crop (the crop is normally the YOLO box).
    # A small contour is background noise and would produce a bad warp.
    if area < 0.5 * gray.size:
        return None
    rect = cv2.minAreaRect(contour)
    return _order_quad(cv2.boxPoints(rect))


def deskew_plate(image: np.ndarray) -> np.ndarray:
    """Deskew / perspective-correct the plate crop.

    Finds the plate quadrilateral and warps it to a straight rectangle. If no
    reliable quadrilateral is found the image is returned unchanged, so this is
    safe to always call.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    quad = _find_plate_quad(gray)
    if quad is None:
        return image

    (tl, tr, br, bl) = quad
    width = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    height = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
    max_w, max_h = int(round(width)), int(round(height))

    # Safety net: never let the warp collapse the crop. If the target size is
    # much smaller than the input, the quadrilateral was wrong -> keep original.
    orig_h, orig_w = gray.shape[:2]
    if max_w < 20 or max_h < 10:
        return image
    if max_w < 0.5 * orig_w or max_h < 0.5 * orig_h:
        logger.debug("deskew skipped: warp target %sx%s too small for %sx%s", max_w, max_h, orig_w, orig_h)
        return image

    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(
        image, matrix, (max_w, max_h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )
    logger.info("deskew applied: %sx%s -> %sx%s", orig_w, orig_h, max_w, max_h)
    return warped


def preprocess_plate(
    plate_image: np.ndarray,
    upscale: float = 2,
    use_clahe: bool = None,
    use_bilateral: bool = None,
    use_deskew: bool = None,
) -> np.ndarray:
    """Prepare a cropped plate image for OCR.

    Steps: deskew -> grayscale -> optional bilateral -> CLAHE -> cubic upscale.
    """
    if use_clahe is None:
        use_clahe = OCR_USE_CLAHE
    if use_bilateral is None:
        use_bilateral = OCR_USE_BILATERAL
    if use_deskew is None:
        use_deskew = OCR_USE_DESKEW

    image = plate_image
    if use_deskew:
        image = deskew_plate(image)

    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if use_bilateral:
        gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

    if use_clahe:
        gray = apply_clahe(gray)

    if upscale and upscale != 1:
        gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)

    return gray


def is_two_line(plate_image: np.ndarray, aspect_threshold: float = None) -> bool:
    """Return whether the aspect ratio is a candidate for two-line analysis.

    This is only a cheap candidate gate, never evidence that a safe split exists.
    """
    if aspect_threshold is None:
        aspect_threshold = OCR_TWO_LINE_ASPECT
    height, width = plate_image.shape[:2]
    if height == 0:
        return False
    return (width / height) < aspect_threshold


def find_split_row(plate_image: np.ndarray) -> Optional[int]:
    """Find a sustained light gap separating two ink bands, or abstain.

    A minimum always exists, even inside a glyph. Require a gap of at least
    2% of image height with <=10% dark pixels across the inner 80% of width,
    plus substantial text above AND below. Borders are excluded horizontally;
    uncertain, blurred or tilted plates fall back to whole-crop OCR.
    """
    if plate_image.size == 0:
        return None
    gray = plate_image if plate_image.ndim == 2 else cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    if height < 20 or width < 20 or int(gray.max())-int(gray.min()) < 20:
        return None
    margin = max(1, int(width * 0.1))
    inner = gray[:, margin:width-margin]
    ink = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    profile = (ink > 0).mean(axis=1)
    lo, hi = int(height * 0.3), int(height * 0.7)
    minimum_gap = max(3, int(np.ceil(height * 0.02)))
    minimum_band = max(3, int(np.ceil(height * 0.08)))
    _, _, components, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    candidates = []
    start = lo
    while start < hi:
        if profile[start] > 0.1:
            start += 1
            continue
        end = start + 1
        while end < hi and profile[end] <= 0.1:
            end += 1
        if end-start >= minimum_gap:
            # Require real ink for a sustained vertical span on both sides;
            # this rejects blank crops, lone rows and margins around one row.
            above = profile[max(1, int(height*0.1)):start]
            below = profile[end:min(height-1, int(height*0.9))]
            if (np.count_nonzero(above >= 0.2) >= minimum_band
                    and np.count_nonzero(below >= 0.2) >= minimum_band):
                mid = (start+end)//2
                crosses_glyph = any(
                    ch >= minimum_band and area >= 4 and y < mid-1 and y+ch > mid+1
                    for x,y,cw,ch,area in components[1:]
                )
                if not crosses_glyph:
                    candidates.append((end-start, -abs(mid-height/2), mid))
        start = end
    return max(candidates)[2] if candidates else None


def split_two_lines(plate_image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return two grayscale slices only for a reliable gap; otherwise None."""
    row = find_split_row(plate_image)
    if row is None:
        return None
    gray = plate_image if plate_image.ndim == 2 else cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    return gray[:row], gray[row:]


__all__ = [
    "decode_image",
    "crop_plate",
    "deskew_plate",
    "preprocess_plate",
    "apply_clahe",
    "is_two_line",
    "split_two_lines",
    "find_split_row",
]
