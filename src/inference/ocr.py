"""Compatibility alias for PlateOCR.

Referenced in Chapter 7.6.3 of Yolo.docx as ``src/inference/ocr.py``.
Provides the ``PlateOCR`` class re-exported from ``src.inference.postprocessing``.
"""

from .postprocessing import PlateOCR, build_detection, format_vn_plate_details

__all__ = ["PlateOCR", "build_detection", "format_vn_plate_details"]
