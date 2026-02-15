"""
Vote69 multi-page Thai form OCR pipeline.

This package provides a production-oriented pipeline focused on extracting handwritten
numbers written on/above dotted lines, and **only** extracting table column 3 across
multi-page continuations.
"""

from .main import process_form, process_inputs

__all__ = ["process_form", "process_inputs"]


