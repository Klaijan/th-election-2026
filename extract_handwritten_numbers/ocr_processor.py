from __future__ import annotations

import logging
import random
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

from . import config
from .types import OCRItem

log = logging.getLogger("extract_handwritten_numbers")

_THAI_DIGITS = str.maketrans(
    {
        "๐": "0",
        "๑": "1",
        "๒": "2",
        "๓": "3",
        "๔": "4",
        "๕": "5",
        "๖": "6",
        "๗": "7",
        "๘": "8",
        "๙": "9",
    }
)


def _digits_only(s: str) -> str:
    s = (s or "").translate(_THAI_DIGITS)
    return "".join(c for c in s if c.isdigit() or c == ".")


def _normalize_thai_digits(s: str) -> str:
    """
    Convert Thai numerals to ASCII numerals, preserving all other characters.
    """
    return (s or "").translate(_THAI_DIGITS)


class OCRProcessor:
    """
    Batch OCR handler.

    Primary: Google Cloud Vision (batch_annotate_images).
    Fallback: Tesseract via pytesseract (if installed); lower accuracy.
    """

    def __init__(
        self,
        *,
        provider: str = config.OCR_PROVIDER,
        languages: Optional[List[str]] = None,
        credentials_path: Optional[str] = None,
    ):
        self.provider = str(provider or "google").lower().strip()
        self.languages = list(languages or config.OCR_LANGUAGES)
        self._credentials_path = credentials_path

    def batch_ocr(self, images: List[np.ndarray], image_ids: List[str], *, retries: int = 3) -> Dict[str, OCRItem]:
        if len(images) != len(image_ids):
            raise ValueError("images and image_ids length mismatch")
        if not images:
            return {}

        if self.provider == "google":
            try:
                return self._batch_google(images, image_ids, retries=int(retries))
            except Exception as e:
                log.warning("Google OCR failed (%s). Trying Tesseract fallback.", str(e))
                return self._batch_tesseract(images, image_ids)

        if self.provider == "tesseract":
            return self._batch_tesseract(images, image_ids)

        raise ValueError(f"Unknown OCR provider: {self.provider}")

    def _batch_google(self, images: List[np.ndarray], image_ids: List[str], *, retries: int) -> Dict[str, OCRItem]:
        import os

        try:
            from google.cloud import vision  # type: ignore
            from google.api_core import exceptions as gexc  # type: ignore
        except ModuleNotFoundError as e:  # pragma: no cover
            raise RuntimeError("google-cloud-vision not installed") from e

        # Support explicit service account key file
        creds_path = self._credentials_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        if creds_path:
            from google.oauth2 import service_account  # type: ignore

            credentials = service_account.Credentials.from_service_account_file(creds_path)
            client = vision.ImageAnnotatorClient(credentials=credentials)
        else:
            client = vision.ImageAnnotatorClient()

        requests = []
        for img in images:
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if not ok:
                raise RuntimeError("cv2.imencode(.jpg) failed")
            req = vision.AnnotateImageRequest(
                image=vision.Image(content=buf.tobytes()),
                features=[vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)],
                image_context=vision.ImageContext(language_hints=self.languages),
            )
            requests.append(req)

        last_err: Optional[Exception] = None
        for attempt in range(int(retries) + 1):
            try:
                resp = client.batch_annotate_images(
                    requests=requests,
                    timeout=30.0,
                )
                return self._parse_google_response(resp, image_ids)
            except Exception as e:
                last_err = e
                retriable = False
                err_name = type(e).__name__
                if "ResourceExhausted" in err_name or "DeadlineExceeded" in err_name:
                    retriable = True
                if isinstance(e, getattr(gexc, "ResourceExhausted", ())):
                    retriable = True
                if isinstance(e, getattr(gexc, "DeadlineExceeded", ())):
                    retriable = True
                # Expired/revoked credentials are not retriable
                if "RefreshError" in err_name or "invalid_grant" in str(e):
                    log.error("Google credentials expired or revoked — not retriable")
                    break
                if not retriable or attempt >= int(retries):
                    break
                sleep_s = (2**attempt) + random.random()
                time.sleep(sleep_s)

        raise RuntimeError(f"Google OCR failed after retries: {last_err}")

    def _parse_google_response(self, resp, image_ids: List[str]) -> Dict[str, OCRItem]:
        out: Dict[str, OCRItem] = {}
        responses = list(getattr(resp, "responses", []) or [])
        for i, r in enumerate(responses):
            img_id = image_ids[i] if i < len(image_ids) else f"img_{i}"
            if getattr(r, "error", None) and getattr(r.error, "message", ""):
                out[img_id] = OCRItem(image_id=img_id, text="", raw_text="", confidence=0.0, provider="google")
                continue

            raw = ""
            if getattr(r, "full_text_annotation", None) and getattr(r.full_text_annotation, "text", None) is not None:
                raw = str(r.full_text_annotation.text or "")
            elif getattr(r, "text_annotations", None):
                # fallback: first text annotation is usually the full text
                raw = str(r.text_annotations[0].description or "")

            conf = self._estimate_google_confidence(r)
            norm = _normalize_thai_digits(raw)
            out[img_id] = OCRItem(
                image_id=img_id,
                text=norm,
                raw_text=raw,
                confidence=float(conf),
                provider="google",
            )
        return out

    @staticmethod
    def _estimate_google_confidence(r) -> float:
        """
        DOCUMENT_TEXT_DETECTION does not provide a single top-level confidence everywhere.
        We compute a best-effort average over word confidences.
        """
        fta = getattr(r, "full_text_annotation", None)
        if not fta:
            return 0.0
        pages = getattr(fta, "pages", None) or []
        vals: List[float] = []
        for pg in pages:
            for blk in getattr(pg, "blocks", None) or []:
                for par in getattr(blk, "paragraphs", None) or []:
                    for word in getattr(par, "words", None) or []:
                        c = getattr(word, "confidence", None)
                        if c is not None:
                            vals.append(float(c))
        if not vals:
            return 0.0
        return float(sum(vals) / float(len(vals)))

    def _batch_tesseract(self, images: List[np.ndarray], image_ids: List[str]) -> Dict[str, OCRItem]:
        try:
            import pytesseract  # type: ignore
        except ModuleNotFoundError as e:  # pragma: no cover
            raise RuntimeError("Tesseract fallback requested but pytesseract is not installed") from e

        def _tess_langs(langs: List[str]) -> str:
            # Map BCP47-ish hints to Tesseract language packs.
            # Thai is `tha` in Tesseract traineddata.
            mapped: List[str] = []
            for l in (langs or []):
                x = str(l or "").strip().lower()
                if not x:
                    continue
                if x in {"th", "tha"}:
                    mapped.append("tha")
                else:
                    mapped.append(x)
            # Deduplicate while preserving order.
            out: List[str] = []
            for x in mapped:
                if x not in out:
                    out.append(x)
            return "+".join(out) if out else "tha"

        tess_lang = _tess_langs(self.languages)

        out: Dict[str, OCRItem] = {}
        for img, img_id in zip(images, image_ids):
            # Make sure it's single-channel uint8
            if img.ndim == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            # Note: we intentionally do NOT restrict to digits-only; downstream parsing can extract digits as needed.
            txt = pytesseract.image_to_string(gray, lang=str(tess_lang), config="--oem 3 --psm 6")
            out[img_id] = OCRItem(
                image_id=img_id,
                text=_normalize_thai_digits(txt),
                raw_text=txt,
                confidence=0.0,
                provider="tesseract",
            )
        return out


