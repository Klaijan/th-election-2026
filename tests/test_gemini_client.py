import pytest

from extract_handwritten_numbers.gemini_client import (
    resolve_gemini_api_key,
    validate_gemini_model,
)


def test_validate_gemini_model_accepts_supported_values():
    assert validate_gemini_model("gemini-3-pro") == "gemini-3-pro"
    assert validate_gemini_model("GEMINI-3-FLASH") == "gemini-3-flash"


def test_validate_gemini_model_rejects_unsupported_value():
    with pytest.raises(ValueError):
        validate_gemini_model("gemini-2.0-flash")


def test_resolve_gemini_api_key_uses_explicit_value(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    assert resolve_gemini_api_key("explicit-key") == "explicit-key"


def test_resolve_gemini_api_key_accepts_either_env_name(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    assert resolve_gemini_api_key() == "google-key"


def test_resolve_gemini_api_key_rejects_conflicting_env_values(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "key-a")
    monkeypatch.setenv("GOOGLE_API_KEY", "key-b")
    with pytest.raises(RuntimeError):
        resolve_gemini_api_key()
