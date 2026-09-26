"""Start the FastAPI backend with uvicorn.

Usage (from the repository root):

    python scripts/run_backend.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    import uvicorn

    from src.utils.config import API_HOST, API_PORT

    uvicorn.run("be.main:app", host=API_HOST, port=API_PORT, reload=False)


if __name__ == "__main__":
    main()
