from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np

from . import config

log = logging.getLogger("extract_handwritten_numbers")


class PDFPasswordError(RuntimeError):
    pass


class PDFCorruptedError(RuntimeError):
    pass


@dataclass(frozen=True)
class PDFLoadInfo:
    pages: int
    dpi: int
    width: int
    height: int


class PDFLoader:
    """
    PDF → list of OpenCV BGR images.

    Primary backend: pdf2image (Poppler).
    Fallback backend (best effort): PyMuPDF (already used in this repo) when pdf2image isn't available.
    """

    def __init__(self, *, dpi: int = config.PDF_DPI):
        self.dpi = int(dpi)

    def load_pdf(self, pdf_path: str) -> List[np.ndarray]:
        """
        Load all pages from a PDF as BGR numpy arrays (OpenCV format).
        """
        try:
            pages = self._load_with_pdf2image(pdf_path)
        except Exception as e_pdf2image:
            log.warning("pdf2image load failed (%s). Trying PyMuPDF fallback.", str(e_pdf2image))
            pages = self._load_with_pymupdf(pdf_path)

        if not pages:
            raise PDFCorruptedError(f"No pages loaded from PDF: {pdf_path}")

        widths = {int(p.shape[1]) for p in pages}
        if len(widths) != 1:
            if bool(getattr(config, "PDF_NORMALIZE_PAGE_WIDTHS", True)):
                try:
                    import cv2
                except Exception as e:  # pragma: no cover
                    raise PDFCorruptedError(
                        f"PDF pages have inconsistent widths: {sorted(widths)} (and OpenCV unavailable to normalize)"
                    ) from e

                target_w = int(max(widths))
                norm: List[np.ndarray] = []
                for img in pages:
                    h, w = img.shape[:2]
                    if int(w) == int(target_w):
                        norm.append(img)
                        continue
                    pad_right = int(target_w - int(w))
                    # pad with white background (BGR 255)
                    padded = cv2.copyMakeBorder(
                        img,
                        0,
                        0,
                        0,
                        int(pad_right),
                        borderType=cv2.BORDER_CONSTANT,
                        value=(255, 255, 255),
                    )
                    norm.append(padded)
                pages = norm
                log.warning("Normalized PDF page widths by padding: %s -> %dpx", str(sorted(widths)), int(target_w))
            else:
                raise PDFCorruptedError(f"PDF pages have inconsistent widths: {sorted(widths)}")

        h0, w0 = pages[0].shape[:2]
        log.info("Loaded %d pages at %d DPI, dimensions: %d×%dpx", len(pages), self.dpi, w0, h0)
        return pages

    def _load_with_pdf2image(self, pdf_path: str) -> List[np.ndarray]:
        try:
            from pdf2image import convert_from_path
        except ModuleNotFoundError as e:  # pragma: no cover
            raise RuntimeError("pdf2image not installed") from e

        try:
            pil_pages = convert_from_path(pdf_path, dpi=int(self.dpi))
        except Exception as e:
            msg = str(e).lower()
            if "password" in msg or "incorrect password" in msg:
                raise PDFPasswordError(f"PDF is password protected: {pdf_path}") from e
            raise PDFCorruptedError(f"Failed to load PDF via pdf2image: {pdf_path} ({e})") from e

        import cv2

        out: List[np.ndarray] = []
        for pil in pil_pages:
            rgb = np.array(pil.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            out.append(bgr)
        return out

    def _load_with_pymupdf(self, pdf_path: str) -> List[np.ndarray]:
        try:
            import fitz  # PyMuPDF
        except ModuleNotFoundError as e:  # pragma: no cover
            raise PDFCorruptedError(
                "Neither pdf2image nor PyMuPDF is available. Install pdf2image+poppler (recommended)."
            ) from e

        import cv2

        doc = None
        try:
            doc = fitz.open(str(pdf_path))
            if doc.needs_pass:
                raise PDFPasswordError(f"PDF is password protected: {pdf_path}")
            pages: List[np.ndarray] = []
            mat = fitz.Matrix(float(self.dpi) / 72.0, float(self.dpi) / 72.0)
            for i in range(int(doc.page_count)):
                pg = doc.load_page(i)
                pix = pg.get_pixmap(matrix=mat, alpha=False)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                if pix.n == 3:
                    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                pages.append(bgr)
            return pages
        except PDFPasswordError:
            raise
        except Exception as e:
            raise PDFCorruptedError(f"Failed to load PDF via PyMuPDF: {pdf_path} ({e})") from e
        finally:
            if doc is not None:
                try:
                    doc.close()
                except Exception:
                    pass


