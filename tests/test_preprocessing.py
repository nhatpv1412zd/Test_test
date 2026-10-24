import cv2
import numpy as np

from src.inference.preprocessing import (
    crop_plate,
    decode_image,
    deskew_plate,
    is_two_line,
    preprocess_plate,
    split_two_lines,
    find_split_row,
)


def _sample_bgr() -> np.ndarray:
    img = np.zeros((120, 240, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 40), (220, 90), (255, 255, 255), -1)
    gradient = np.linspace(0, 60, 240, dtype=np.float32)
    img = np.clip(img.astype(np.float32) * 0.6 + gradient[None, :, None], 0, 255).astype(np.uint8)
    return img


def test_decode_image_roundtrip():
    ok, buf = cv2.imencode(".jpg", _sample_bgr())
    assert ok
    decoded = decode_image(buf.tobytes())
    assert decoded is not None
    assert decoded.ndim == 3
    assert decoded.shape[2] == 3


def test_crop_plate_without_padding():
    crop = crop_plate(_sample_bgr(), [20, 40, 220, 90])
    assert crop.shape == (50, 200, 3)


def test_crop_plate_with_padding_clamps_to_bounds():
    img = _sample_bgr()  # 120 (h) x 240 (w)
    crop = crop_plate(img, [20, 40, 220, 90], padding=0.1)
    assert crop.shape == (60, 240, 3)


def test_preprocess_plate_is_gray_upscaled_and_not_binary():
    crop = crop_plate(_sample_bgr(), [20, 40, 220, 90])
    out = preprocess_plate(crop, upscale=2, use_deskew=False)
    assert out.shape == (crop.shape[0] * 2, crop.shape[1] * 2)
    assert out.ndim == 2  # grayscale
    assert out.dtype == np.uint8
    assert len(np.unique(out)) > 2  # not a 0/255 binary image


def test_deskew_plate_returns_image():
    out = deskew_plate(_sample_bgr())
    assert isinstance(out, np.ndarray)
    assert out.ndim == 3
    assert out.shape[0] > 0 and out.shape[1] > 0


def test_is_two_line_by_aspect_ratio():
    assert is_two_line(np.zeros((200, 150, 3), dtype=np.uint8)) is True   # 0.75
    assert is_two_line(np.zeros((60, 300, 3), dtype=np.uint8)) is False   # 5.0


def test_is_two_line_boundary_1_9():
    assert is_two_line(np.zeros((100, 185, 3), dtype=np.uint8)) is True    # 1.85
    assert is_two_line(np.zeros((100, 195, 3), dtype=np.uint8)) is False   # 1.95


def test_split_two_lines_splits_at_the_gap():
    img = np.full((100, 200), 255, dtype=np.uint8)
    img[20:35, 20:180] = 0   # top line
    img[65:80, 20:180] = 0   # bottom line
    top, bottom = split_two_lines(img)
    assert top.shape[0] + bottom.shape[0] == 100
    assert 25 <= top.shape[0] <= 55  # split lands in the blank gap


def test_split_abstains_on_blank_tiny_empty_or_one_band():
    images = [np.zeros((200,150),np.uint8),np.full((100,100),255,np.uint8),
              np.zeros((3,10),np.uint8),np.empty((0,0),np.uint8)]
    single = np.full((100,150),255,np.uint8); single[35:65,20:130] = 0
    images.append(single)
    for image in images:
        assert find_split_row(image) is None
        assert split_two_lines(image) is None


def test_local_minimum_with_ink_is_not_a_gap():
    image = np.full((100,150),255,np.uint8)
    image[10:90,20:130] = 0
    image[60:63,20:80] = 255  # Lowest projection still crosses half a glyph band.
    assert find_split_row(image) is None


def test_one_pixel_valley_is_not_enough_to_split():
    image = np.full((100,150),255,np.uint8)
    image[15:85,20:130] = 0; image[50,20:130] = 255
    assert split_two_lines(image) is None


def test_split_never_crosses_a_thin_connected_glyph():
    image = np.full((100,200),255,np.uint8)
    image[15:35,20:180] = 0; image[65:85,20:180] = 0
    image[30:70,100:104] = 0  # Low ink fraction but a connected glyph bridges gap.
    assert find_split_row(image) is None


def test_reported_0026_split_is_rejected_when_saved_crop_available():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1]/'experiments/diagnostic-test/visualizations/preprocessed/002-0.png'
    if not path.exists():
        import pytest
        pytest.skip('Optional diagnostic image not present')
    image = cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    assert find_split_row(image) is None
