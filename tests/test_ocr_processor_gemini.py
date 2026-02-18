import numpy as np

from extract_handwritten_numbers.ocr_processor import OCRProcessor


def test_batch_ocr_gemini_returns_digits(monkeypatch):
    monkeypatch.setattr(
        "extract_handwritten_numbers.ocr_processor.validate_gemini_model",
        lambda m: str(m).lower(),
    )
    monkeypatch.setattr(
        "extract_handwritten_numbers.ocr_processor.resolve_gemini_api_key",
        lambda explicit_api_key=None: "test-key",
    )
    monkeypatch.setattr(
        "extract_handwritten_numbers.ocr_processor.ocr_image_bytes_with_gemini",
        lambda **kwargs: "รวม ๑๒.๓",
    )

    img = np.full((32, 32), 255, dtype=np.uint8)
    proc = OCRProcessor(provider="gemini", model="gemini-3-flash")
    out = proc.batch_ocr([img], ["img1"], retries=0)

    assert out["img1"].provider == "gemini"
    assert out["img1"].text == "12.3"
