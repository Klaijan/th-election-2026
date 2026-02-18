import pytest

import run_typhoon_ocr


def test_resolve_provider_auto_selects_gemini_for_supported_model():
    assert run_typhoon_ocr._resolve_provider("auto", "gemini-3-pro") == "gemini"


def test_resolve_provider_auto_selects_typhoon_for_non_gemini_model():
    assert run_typhoon_ocr._resolve_provider("auto", "typhoon-ocr") == "typhoon"


def test_resolve_provider_rejects_invalid_gemini_model():
    with pytest.raises(SystemExit):
        run_typhoon_ocr._resolve_provider("gemini", "gemini-2.0-flash")
