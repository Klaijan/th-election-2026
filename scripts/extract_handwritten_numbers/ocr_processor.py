from __future__ import annotations

import logging
import random
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

from . import config
from .gemini_client import ocr_image_bytes_with_gemini, resolve_gemini_api_key, validate_gemini_model
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


class OCRProcessor:
    """
    Batch OCR handler.

    Primary: Google Cloud Vision (batch_annotate_images) or Gemini API.
    Fallback: Tesseract via pytesseract (if installed); lower accuracy.
    """

    def __init__(
        self,
        *,
        provider: str = config.OCR_PROVIDER,
        languages: Optional[List[str]] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.provider = str(provider or "google").lower().strip()
        self.languages = list(languages or config.OCR_LANGUAGES)
        self.model = str(model or config.OCR_MODEL).strip()
        self.api_key = str(api_key or "").strip()

    def batch_ocr(self, images: List[np.ndarray], image_ids: List[str], *, retries: int = config.OCR_RETRIES) -> Dict[str, OCRItem]:
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

        if self.provider == "gemini":
            try:
                return self._batch_gemini(images, image_ids, retries=int(retries))
            except Exception as e:
                log.warning("Gemini OCR failed (%s). Trying Tesseract fallback.", str(e))
                return self._batch_tesseract(images, image_ids)

        if self.provider == "tesseract":
            return self._batch_tesseract(images, image_ids)

        raise ValueError(f"Unknown OCR provider: {self.provider}")

    def _batch_gemini(self, images: List[np.ndarray], image_ids: List[str], *, retries: int) -> Dict[str, OCRItem]:
        model = validate_gemini_model(self.model)
        api_key = resolve_gemini_api_key(self.api_key or None)
        lang_hint = ", ".join([x for x in self.languages if str(x).strip()])
        prompt = "Extract all visible text from this image exactly. Return only text."
        if lang_hint:
            prompt += f" Preferred languages: {lang_hint}."

        out: Dict[str, OCRItem] = {}
        for img, img_id in zip(images, image_ids):
            ok, buf = cv2.imencode(".png", img)
            if not ok:
                raise RuntimeError("cv2.imencode(.png) failed")

            last_err: Optional[Exception] = None
            raw = ""
            for attempt in range(int(retries) + 1):
                try:
                    raw = ocr_image_bytes_with_gemini(
                        image_bytes=buf.tobytes(),
                        mime_type="image/png",
                        model=model,
                        api_key=api_key,
                        prompt=prompt,
                        timeout_s=float(config.OCR_TIMEOUT_S),
                    )
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    if attempt >= int(retries):
                        break
                    sleep_s = (2**attempt) + random.random()
                    time.sleep(sleep_s)

            if last_err is not None:
                raise RuntimeError(f"Gemini OCR failed for image {img_id}: {last_err}")

            conf = 0.5 if str(raw).strip() else 0.0
            out[img_id] = OCRItem(
                image_id=img_id,
                text=_digits_only(raw),
                raw_text=str(raw),
                confidence=float(conf),
                provider="gemini",
            )
        return out

    def _batch_google(self, images: List[np.ndarray], image_ids: List[str], *, retries: int) -> Dict[str, OCRItem]:
        try:
            from google.cloud import vision  # type: ignore
            from google.api_core import exceptions as gexc  # type: ignore
        except ModuleNotFoundError as e:  # pragma: no cover
            raise RuntimeError("google-cloud-vision not installed") from e

        client = vision.ImageAnnotatorClient()

        requests = []
        for img in images:
            # Encode to JPEG (small + good enough for handwriting)
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
                resp = client.batch_annotate_images(requests=requests)
                return self._parse_google_response(resp, image_ids)
            except Exception as e:
                last_err = e
                retriable = False
                if "ResourceExhausted" in type(e).__name__:
                    retriable = True
                if "DeadlineExceeded" in type(e).__name__:
                    retriable = True
                if isinstance(e, getattr(gexc, "ResourceExhausted", ())):
                    retriable = True
                if isinstance(e, getattr(gexc, "DeadlineExceeded", ())):
                    retriable = True
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
            out[img_id] = OCRItem(
                image_id=img_id,
                text=_digits_only(raw),
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

        out: Dict[str, OCRItem] = {}
        for img, img_id in zip(images, image_ids):
            # Make sure it's single-channel uint8
            if img.ndim == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            txt = pytesseract.image_to_string(
                gray,
                lang="eng",
                config="--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.",
            )
            out[img_id] = OCRItem(image_id=img_id, text=_digits_only(txt), raw_text=txt, confidence=0.0, provider="tesseract")
        return out


