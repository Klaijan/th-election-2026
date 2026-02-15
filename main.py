#!/usr/bin/env python3
"""
Convenience entrypoint for the multi-page Thai form OCR pipeline.

This file delegates to `vote69_form_ocr.main` so you can run:

  python main.py --input path/to/form.pdf --out output --debug
  python main.py --input path/to/folder --out output --debug
"""

from vote69_form_ocr.main import _cli


if __name__ == "__main__":
    raise SystemExit(_cli())


