"""FastAPI application entry point.

Run from the repository root:

    uvicorn be.main:app --reload

or from inside ``be/``:

    uvicorn main:app --reload
"""

import logging
import os
import sys
from pathlib import Path

# Allow ``import src.*`` / ``import be.*`` no matter which directory uvicorn
# is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from be.routes.detection import router as detection_router
from src.utils.config import API_HOST, API_PORT, CORS_ORIGINS

# Show the inference pipeline logs (class/confidence filtering, OCR, ...).
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(title="License Plate Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detection_router)

if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)
