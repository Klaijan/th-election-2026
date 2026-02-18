from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Optional
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest


GEMINI_SUPPORTED_MODELS: tuple[str, str] = ("gemini-3-pro", "gemini-3-flash")
_GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GeminiAPIError(RuntimeError):
    pass


def validate_gemini_model(model: str) -> str:
    normalized = str(model or "").strip().lower()
    if normalized not in GEMINI_SUPPORTED_MODELS:
        supported = ", ".join(GEMINI_SUPPORTED_MODELS)
        raise ValueError(f"Unsupported Gemini model: {model!r}. Supported models: {supported}")
    return normalized


def resolve_gemini_api_key(explicit_api_key: Optional[str] = None) -> str:
    explicit = str(explicit_api_key or "").strip()
    if explicit:
        return explicit

    v1 = str(os.environ.get("GEMINI_API_KEY") or "").strip()
    v2 = str(os.environ.get("GOOGLE_API_KEY") or "").strip()

    if v1 and v2 and v1 != v2:
        raise RuntimeError(
            "Both GEMINI_API_KEY and GOOGLE_API_KEY are set with different values. "
            "Set only one or pass --gemini-api-key explicitly."
        )
    if v1:
        return v1
    if v2:
        return v2
    raise RuntimeError("Missing Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY.")


def _mime_type_for_path(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if ext == ".gif":
        return "image/gif"
    return "application/octet-stream"


def _extract_text(payload: dict[str, Any]) -> str:
    out: list[str] = []
    for cand in list(payload.get("candidates") or []):
        content = cand.get("content") if isinstance(cand, dict) else None
        parts = (content or {}).get("parts") if isinstance(content, dict) else None
        for part in list(parts or []):
            if not isinstance(part, dict):
                continue
            txt = part.get("text")
            if isinstance(txt, str) and txt.strip():
                out.append(txt.strip())
    return "\n".join(out).strip()


def _request_json(url: str, body: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    req = urlrequest.Request(
        url=url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlrequest.urlopen(req, timeout=float(timeout_s)) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urlerror.HTTPError as e:
        detail = ""
        try:
            raw = e.read().decode("utf-8", errors="replace")
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                err = parsed.get("error")
                if isinstance(err, dict):
                    detail = str(err.get("message") or "").strip()
        except Exception:
            detail = ""
        msg = f"Gemini API HTTP {e.code}"
        if detail:
            msg += f": {detail}"
        raise GeminiAPIError(msg) from e
    except urlerror.URLError as e:
        raise GeminiAPIError(f"Gemini API request failed: {e}") from e

    try:
        payload = json.loads(raw)
    except Exception as e:
        raise GeminiAPIError("Gemini API returned invalid JSON") from e
    if not isinstance(payload, dict):
        raise GeminiAPIError("Gemini API returned an invalid response payload")
    if isinstance(payload.get("error"), dict):
        err = payload["error"]
        detail = str(err.get("message") or "").strip()
        raise GeminiAPIError(f"Gemini API error: {detail or 'unknown error'}")
    return payload


def ocr_image_bytes_with_gemini(
    *,
    image_bytes: bytes,
    mime_type: str,
    model: str,
    api_key: str,
    prompt: str,
    timeout_s: float = 90.0,
) -> str:
    model_name = validate_gemini_model(model)
    key = str(api_key or "").strip()
    if not key:
        raise RuntimeError("Gemini API key is required")

    query = urlparse.urlencode({"key": key})
    url = _GEMINI_API_URL.format(model=model_name) + f"?{query}"
    body = {
        "contents": [
            {
                "parts": [
                    {"text": str(prompt or "Extract all text from this image.")},
                    {
                        "inlineData": {
                            "mimeType": str(mime_type or "application/octet-stream"),
                            "data": base64.b64encode(image_bytes).decode("ascii"),
                        }
                    },
                ]
            }
        ],
        "generationConfig": {"temperature": 0},
    }
    payload = _request_json(url, body, timeout_s=float(timeout_s))
    text = _extract_text(payload)
    if text:
        return text

    reason = ""
    cands = list(payload.get("candidates") or [])
    if cands and isinstance(cands[0], dict):
        reason = str(cands[0].get("finishReason") or "").strip()
    if reason:
        raise GeminiAPIError(f"Gemini API returned no text (finishReason={reason})")
    raise GeminiAPIError("Gemini API returned no OCR text")


def ocr_image_path_with_gemini(
    path: str | Path,
    *,
    model: str,
    api_key: str,
    prompt: str,
    timeout_s: float = 90.0,
) -> str:
    p = Path(path)
    data = p.read_bytes()
    return ocr_image_bytes_with_gemini(
        image_bytes=data,
        mime_type=_mime_type_for_path(p),
        model=model,
        api_key=api_key,
        prompt=prompt,
        timeout_s=float(timeout_s),
    )
