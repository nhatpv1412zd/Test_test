"""Wrapper script for benchmark and validation.

Referenced in Chapter 4.6 of Yolo.docx as ``scripts/run_val_and_benchmark.py``.
Directs execution to the full ALPR diagnostic benchmark suite in
``scripts/diagnostic_benchmark.py``.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnostic_benchmark import main

if __name__ == "__main__":
    main()
