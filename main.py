#!/usr/bin/env python3
"""
Convenience entrypoint for the multi-page Thai form OCR pipeline.

This file delegates to `extract_handwritten_numbers.main` so you can run:

  python main.py --input path/to/form.pdf --out output --debug
  python main.py --input path/to/folder --out output --debug
"""

from extract_handwritten_numbers.main import _cli


if __name__ == "__main__":
    raise SystemExit(_cli())


