# License Plate Recognition (LPR)

Automatic Vietnamese licence-plate detection and recognition.

- **Detection:** YOLOv8n (`weights/best.pt`), single class `license_plate` (retrain pending — current checkpoint is 2-class).
- **OCR:** EasyOCR (English charset, allowlist restricted to `0-9 A-Z -`).
- **Backend:** FastAPI (`be/`).
- **Frontend:** single-page HTML + Tailwind (`fe/`).

The AI/CV logic was extracted from the old monolithic `be/main.py` into
`src/inference/`; the API layer only orchestrates.

---

## Project structure

```
.
├── .github/workflows/        CI (compile + tests)
├── assets/
│   ├── samples/              demo images used for manual checks
│   ├── screenshots/
│   └── diagrams/
├── data/
│   ├── custom/                self-collected dataset (fine-tuning source)
│   │   ├── images/{train,val,test}/   518 / 148 / 74 (.png)
│   │   ├── labels/{train,val,test}/   present but empty (unlabelled)
│   │   └── data.yaml          nc=1  names=[license_plate]
│   ├── kaggle/                dataset downloaded from Kaggle
│   │   ├── images/{train,val}/        3433 / 1145 (.png)
│   │   ├── labels/                    bbox labels, 1 class (license_plate)
│   │   ├── labels_polygon/            original polygon labels (backup)
│   │   ├── labels_2class_backup/      pre-merge bbox labels (backup)
│   │   └── data.yaml          nc=1  names=[license_plate]
│   └── README.md
├── docs/                     audit report, how-to-run, architecture
├── experiments/              training outputs (empty; recreated when training)
├── fe/                       frontend (index.html)
├── notebooks/                exploratory notebook (NOT production code)
├── scripts/                  helper scripts
├── src/
│   ├── inference/            preprocessing.py, predictor.py, postprocessing.py
│   ├── training/             train.py
│   └── utils/                paths.py, config.py
├── be/
│   ├── main.py               FastAPI app
│   ├── routes/               /detect endpoint
│   └── services/             detection orchestration
├── tests/
├── weights/
│   ├── best.pt               model used for inference
│   └── last.pt               final-epoch checkpoint (kept for reference)
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` to override defaults (all settings are optional):

```bash
cp .env.example .env
```

---

## Run

### Backend

From the repository root (recommended):

```bash
uvicorn be.main:app --reload
```

or from inside `be/`:

```bash
cd be
uvicorn main:app --reload
```

The API listens on `http://127.0.0.1:8000`.

### Frontend

Open `fe/index.html` with Live Server (or any static server). The page posts
images to `POST http://127.0.0.1:8000/detect`.

### API

```
POST /detect
  form-data: file=<image>
  200 -> {"results": [{"confidence": 0.99, "plate_number": "51A-12345"}]}
```

---

## Training

```bash
# Kaggle dataset
python -m src.training.train --data data/kaggle/data.yaml --epochs 50 --batch 8 --device cpu

# Custom (fine-tuning) dataset
python -m src.training.train --data data/custom/data.yaml --epochs 50 --batch 8 --device cpu
```

Each source has its own config: `data/kaggle/data.yaml` (single class
`license_plate`; bbox labels converted from polygons, then merged to one class)
and `data/custom/data.yaml` (1 class, labels not added yet). Training output
goes to `experiments/<name>/` (default `exp001-baseline`).

---

## Tests

```bash
pytest -q
```

Tests that need `ultralytics`/`easyocr` (model downloads) are skipped
automatically when those packages are not installed.

---

## Docker

```bash
docker compose up --build
```

---

## Notes

- `weights/best.pt` and `weights/last.pt` are **not identical** (different
  SHA-256). Inference uses `best.pt` (best validation checkpoint).
- The previous backend hard-coded an absolute Windows path that no longer
  existed; the model path is now resolved relative to the repo root
  (`MODEL_PATH`, default `weights/best.pt`).
- See `docs/audit-report.md` for the audit and `docs/architecture.md` for the
  data flow.
