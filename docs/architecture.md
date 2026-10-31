# Architecture

## Runtime flow (inference)

```
fe/index.html
   │  multipart POST /detect
   ▼
be/routes/detection.py            APIRouter, reads UploadFile
   ▼
be/services/detection_service.py  orchestration only
   ├─ src/inference/preprocessing.decode_image        bytes -> BGR ndarray
   ├─ src/inference/predictor.PlateDetector.detect     YOLO -> [Detection(xyxy, conf, class_id)]
   │     filter: class_id in PLATE_CLASS_IDS and confidence >= OCR_MIN_CONFIDENCE
   │     for each kept detection:
   ├─ src/inference/preprocessing.crop_plate           xyxy crop (+ CROP_PADDING)
   ├─ src/inference/preprocessing.preprocess_plate     deskew -> gray -> (bilateral) -> CLAHE -> upscale
   └─ src/inference/postprocessing.PlateOCR.read_text  two-line split? -> EasyOCR -> VN format
   ▼
{"results": [{"confidence": float, "plate_number": str}]}
```

Notes on the revised inference decision logic:

- **Class filtering** — only detections whose class is in `PLATE_CLASS_IDS`
  (default `{0}`) are sent to OCR. The kaggle dataset was merged to a single
  `license_plate` class (id 0), so the retrained model emits class 0 only. An
  empty set means "accept every class".
- **Confidence threshold** — detections below `OCR_MIN_CONFIDENCE` are dropped
  (removes noise boxes such as text/logos on the plate).
- **No Otsu binarisation** — the old pipeline thresholded the plate to black &
  white, which destroyed thin/curved strokes (e.g. `3`→`1`, `6`→`8`). The
  revised preprocessing keeps grey levels (`gray -> optional bilateral -> CLAHE
  -> cubic upscale`).
- **Flexible plate formatting** — OCR fragments are grouped into lines by an
  adaptive geometric threshold, then formatted for both 1-line car plates
  (`30A-531.03`) and 2-line motorcycle plates (`68-G1 668.86`), without a single
  hard-coded template.

## Configuration

`src/utils/config.py` reads environment variables (optionally from `.env`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_PATH` | `weights/best.pt` | YOLO weights (relative to repo root) |
| `OCR_LANGUAGES` | `en` | EasyOCR languages |
| `OCR_USE_GPU` | `false` | EasyOCR GPU flag |
| `OCR_ALLOWLIST` | `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.` | OCR character whitelist (incl. dot) |
| `OCR_UPSCALE` | `2` | Plate upscale factor |
| `OCR_MIN_CONFIDENCE` | `0.25` | Drop detections below this confidence |
| `PLATE_CLASS_IDS` | `0` | YOLO classes treated as plates (1-class model) |
| `CROP_PADDING` | `0.1` | Margin around the box (fraction; negative crops inward) |
| `OCR_USE_CLAHE` | `true` | Local contrast enhancement |
| `OCR_USE_BILATERAL` | `false` | Bilateral denoise before CLAHE |
| `OCR_USE_DESKEW` | `true` | Deskew / perspective-correct the plate crop |
| `OCR_TWO_LINE_ASPECT` | `1.9` | W/H below this => two-line plate |
| `OCR_SPLIT_TWO_LINE` | `true` | OCR each line of a two-line plate separately |
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8000` | Uvicorn bind address |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |
| `TRAIN_*` | notebook values | Training defaults |

## Lazy loading

`PlateDetector` and `PlateOCR` create their heavy objects (`YOLO`, `easyocr.Reader`)
on first use. Importing `be.main` therefore does not download models, which
keeps tests and tooling fast. This does not change inference results.
