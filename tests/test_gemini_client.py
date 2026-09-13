from unittest.mock import Mock, patch

import httpx
import pytest
from src.narrative.gemini_client import generate_narrative, GeminiQuotaExceededError


def _mock_response(status_code: int, text: str = "quota error"):
    request = httpx.Request("POST", "https://example.com")
    response = httpx.Response(status_code, request=request, text=text)
    return response


def test_generate_narrative_raises_quota_error_on_429():
    with patch("src.config.settings.gemini_api_key", "fake-key"), \
         patch("httpx.post", return_value=_mock_response(429)):
        with pytest.raises(GeminiQuotaExceededError):
            generate_narrative("grid_000_000", 2, 0.5, 30.0)


def test_generate_narrative_raises_plain_runtime_error_on_other_status():
    with patch("src.config.settings.gemini_api_key", "fake-key"), \
         patch("httpx.post", return_value=_mock_response(500, "server error")):
        with pytest.raises(RuntimeError) as exc_info:
            generate_narrative("grid_000_000", 2, 0.5, 30.0)
        assert not isinstance(exc_info.value, GeminiQuotaExceededError)


def test_quota_error_is_a_runtime_error_subclass():
    # A bare `except RuntimeError` elsewhere must still catch it -- only
    # run_pipeline.py's more specific except needs to distinguish it.
    assert issubclass(GeminiQuotaExceededError, RuntimeError)
