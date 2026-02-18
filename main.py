#!/usr/bin/env python3
"""
Convenience entrypoint for the multi-page Thai form OCR pipeline.

This file delegates to `extract_handwritten_numbers.main` so you can run:

  python main.py --input path/to/form.pdf --out output --debug
  python main.py --input path/to/folder --out output --debug
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path even if this file is executed from another cwd.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    # Prefer the repo-local pipeline implementation.
    from extract_handwritten_numbers.main import _cli
except ModuleNotFoundError as e:  # pragma: no cover
    # Common causes:
    # - Dependencies like opencv-python (cv2) are not installed
    if getattr(e, "name", "") == "cv2":
        raise SystemExit(
            "Missing dependency: OpenCV (cv2).\n"
            "Install project deps (in your venv):\n"
            "  python -m pip install -r requirements.txt\n"
        ) from e
    raise

if __name__ == "__main__":
    raise SystemExit(_cli())
