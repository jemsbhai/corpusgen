"""Tests for LLMBackend — multi-provider LLM generation via litellm."""

from unittest.mock import patch, MagicMock

import pytest

from corpusgen.generate.phon_ctg.loop import GenerationBackend
from corpusgen.generate.backends.llm_api import (
    LLMBackend,
    DEFAULT_PROMPT_TEMPLATE,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_mock_response(text: str) -> MagicMock:
    """Create a mock litellm completion response."""
    choice = MagicMock()
    choice.message.content = text
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_multi_response(texts: list[str]) -> MagicMock:
    """Create a mock response with multiple choices."""
    choices = []
    for text in texts:
        choice = MagicMock()
        choice.message.content = text
        choices.append(choice)
    response = MagicMock()
    response.choices = choices
    return response


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """LLMBackend creation and configuration."""

    def test_basic_creation(self):
        backend = LLMBackend(model="openai/gpt-4o-mini")
        assert isinstance(backend, GenerationBackend)
        assert backend.name == "llm_api"

    def test_model_property(self):
        backend = LLMBackend(model="anthropic/claude-3-5-sonnet-latest")
        assert backend.model == "anthropic/claude-3-5-sonnet-latest"

    def test_default_language(self):
        backend = LLMBackend(model="openai/gpt-4o-mini")
        assert backend.language == "en-us"

    def test_custom_language(self):
        backend = LLMBackend(model="openai/gpt-4o-mini", language="fr-fr")
        assert backend.language == "fr-fr"

    def test_custom_api_key(self):
        backend = LLMBackend(
            model="openai/gpt-4o-mini",
            api_key="sk-test-123",
        )
        assert backend.api_key == "sk-test-123"

    def test_default_prompt_template(self):
        backend = LLMBackend(model="openai/gpt-4o-mini")
        assert backend.prompt_template == DEFAULT_PROMPT_TEMPLATE

    def test_custom_prompt_template(self):
        template = "Generate a sentence with: {target_units}"
        backend = LLMBackend(
            model="openai/gpt-4o-mini",
            prompt_template=template,
        )
        assert backend.prompt_template == template

    def test_default_max_retries(self):
        backend = LLMBackend(model="openai/gpt-4o-mini")
        assert backend.max_retries == 3

    def test_custom_max_retries(self):
        backend = LLMBackend(model="openai/gpt-4o-mini", max_retries=5)
        assert backend.max_retries == 5

    def test_default_retry_delay(self):
        backend = LLMBackend(model="openai/gpt-4o-mini")
        assert backend.retry_delay == 1.0

    def test_custom_retry_delay(self):
        backend = LLMBackend(model="openai/gpt-4o-mini", retry_delay=0.5)
        assert backend.retry_delay == 0.5

    def test_default_request_delay(self):
        backend = LLMBackend(model="openai/gpt-4o-mini")
        assert backend.request_delay == 0.0

    def test_custom_request_delay(self):
        backend = LLMBackend(model="openai/gpt-4o-mini", request_delay=0.5)
        assert backend.request_delay == 0.5


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------


class TestPromptTemplate:
    """Prompt template formatting."""

    def test_default_template_has_placeholders(self):
        assert "{target_units}" in DEFAULT_PROMPT_TEMPLATE
        assert "{language}" in DEFAULT_PROMPT_TEMPLATE

    def test_format_prompt(self):
        backend = LLMBackend(model="openai/gpt-4o-mini")
        prompt = backend.format_prompt(
            target_units=["ʃ", "θ", "ŋ"],
        )
        assert "ʃ" in prompt
        assert "θ" in prompt
        assert isinstance(prompt, str)

    def test_custom_template_format(self):
        template = "Make a sentence with these sounds: {target_units}"
        backend = LLMBackend(
            model="openai/gpt-4o-mini",
            prompt_template=template,
        )
        prompt = backend.format_prompt(target_units=["p", "b"])
        assert "p" in prompt
        assert "b" in prompt


# ---------------------------------------------------------------------------
# Generate: basic flow (mocked)
# ---------------------------------------------------------------------------


class TestGenerate:
    """generate() calls LLM and returns phonemized candidates."""

    @patch("corpusgen.generate.backends.llm_api._call_llm")
    def test_returns_candidates(self, mock_call):
        mock_call.return_value = _make_mock_response(
            "The ship sank in the thick fog.\n"
            "She thought about nothing."
        )
        backend = LLMBackend(model="openai/gpt-4o-mini")
        candidates = backend.generate(target_units=["ʃ", "θ"], k=5)
        assert isinstance(candidates, list)
        assert len(candidates) >= 1

    @patch("corpusgen.generate.backends.llm_api._call_llm")
    def test_candidates_have_text_and_phonemes(self, mock_call):
        mock_call.return_value = _make_mock_response("The cat sat on the mat.")
        backend = LLMBackend(model="openai/gpt-4o-mini")
        candidates = backend.generate(target_units=["k", "æ"], k=5)
        for c in candidates:
            assert "text" in c
            assert "phonemes" in c
            assert isinstance(c["phonemes"], list)

    @patch("corpusgen.generate.backends.llm_api._call_llm")
    def test_phonemes_are_populated(self, mock_call):
        mock_call.return_value = _make_mock_response("The cat sat.")
        backend = LLMBackend(model="openai/gpt-4o-mini")
        candidates = backend.generate(target_units=["k"], k=5)
        # At least one candidate should have non-empty phonemes
        assert any(len(c["phonemes"]) > 0 for c in candidates)

    @patch("corpusgen.generate.backends.llm_api._call_llm")
    def test_passes_model_to_litellm(self, mock_call):
        mock_call.return_value = _make_mock_response("Hello world.")
        backend = LLMBackend(model="anthropic/claude-3-5-sonnet-latest")
        backend.generate(target_units=["h"], k=3)
        call_kwargs = mock_call.call_args
        assert call_kwargs is not None
        assert call_kwargs[1]["model"] == "anthropic/claude-3-5-sonnet-latest"

    @patch("corpusgen.generate.backends.llm_api._call_llm")
    def test_passes_api_key_to_litellm(self, mock_call):
        mock_call.return_value = _make_mock_response("Hello.")
        backend = LLMBackend(
            model="openai/gpt-4o-mini",
            api_key="sk-test-key",
        )
        backend.generate(target_units=["h"], k=3)
        call_kwargs = mock_call.call_args
        assert call_kwargs[1]["api_key"] == "sk-test-key"

    @patch("corpusgen.generate.backends.llm_api._call_llm")
    def test_splits_multiline_response(self, mock_call):
        mock_call.return_value = _make_mock_response(
            "First sentence here.\nSecond sentence here.\nThird one."
        )
        backend = LLMBackend(model="openai/gpt-4o-mini")
        candidates = backend.generate(target_units=["s"], k=5)
        assert len(candidates) >= 2

    @patch("corpusgen.generate.backends.llm_api._call_llm")
    def test_filters_empty_lines(self, mock_call):
        mock_call.return_value = _make_mock_response(
            "Good sentence.\n\n\nAnother good one.\n"
        )
        backend = LLMBackend(model="openai/gpt-4o-mini")
        candidates = backend.generate(target_units=["s"], k=5)
        for c in candidates:
            assert c["text"].strip() != ""

    @patch("corpusgen.generate.backends.llm_api._call_llm")
    def test_respects_k_limit(self, mock_call):
        mock_call.return_value = _make_mock_response(
            "One.\nTwo.\nThree.\nFour.\nFive.\nSix.\nSeven."
        )
        backend = LLMBackend(model="openai/gpt-4o-mini")
        candidates = backend.generate(target_units=["s"], k=3)
        assert len(candidates) <= 3


# ---------------------------------------------------------------------------
# Generate: extra LLM parameters
# ---------------------------------------------------------------------------


class TestGenerateParams:
    """Extra parameters forwarded to litellm."""

    @patch("corpusgen.generate.backends.llm_api._call_llm")
    def test_temperature_forwarded(self, mock_call):
        mock_call.return_value = _make_mock_response("Hello.")
        backend = LLMBackend(
            model="openai/gpt-4o-mini",
            temperature=0.9,
        )
        backend.generate(target_units=["h"], k=3)
        call_kwargs = mock_call.call_args[1]
        assert call_kwargs["temperature"] == 0.9

    @patch("corpusgen.generate.backends.llm_api._call_llm")
    def test_max_tokens_forwarded(self, mock_call):
        mock_call.return_value = _make_mock_response("Hello.")
        backend = LLMBackend(
            model="openai/gpt-4o-mini",
            max_tokens=500,
        )
        backend.generate(target_units=["h"], k=3)
        call_kwargs = mock_call.call_args[1]
        assert call_kwargs["max_tokens"] == 500

    @patch("corpusgen.generate.backends.llm_api._call_llm")
    def test_default_temperature(self, mock_call):
        mock_call.return_value = _make_mock_response("Hello.")
        backend = LLMBackend(model="openai/gpt-4o-mini")
        backend.generate(target_units=["h"], k=3)
        call_kwargs = mock_call.call_args[1]
        assert "temperature" in call_kwargs


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


class TestRetryLogic:
    """Retry on transient failures."""

    @patch("corpusgen.generate.backends.llm_api.time.sleep")
    @patch("corpusgen.generate.backends.llm_api._call_llm")
    def test_retries_on_failure(self, mock_call, mock_sleep):
        mock_call.side_effect = [
            Exception("rate limited"),
            _make_mock_response("Success after retry."),
        ]
        backend = LLMBackend(model="openai/gpt-4o-mini", max_retries=3, retry_delay=0.0)
        candidates = backend.generate(target_units=["s"], k=5)
        assert len(candidates) >= 1
        assert mock_call.call_count == 2

    @patch("corpusgen.generate.backends.llm_api.time.sleep")
    @patch("corpusgen.generate.backends.llm_api._call_llm")
    def test_returns_empty_after_max_retries(self, mock_call, mock_sleep):
        mock_call.side_effect = Exception("always fails")
        backend = LLMBackend(model="openai/gpt-4o-mini", max_retries=2, retry_delay=0.0)
        candidates = backend.generate(target_units=["s"], k=5)
        assert candidates == []
        assert mock_call.call_count == 3  # initial + 2 retries

    @patch("corpusgen.generate.backends.llm_api.time.sleep")
    @patch("corpusgen.generate.backends.llm_api._call_llm")
    def test_retry_delay_called(self, mock_call, mock_sleep):
        mock_call.side_effect = [
            Exception("fail"),
            _make_mock_response("OK."),
        ]
        backend = LLMBackend(model="openai/gpt-4o-mini", max_retries=3, retry_delay=2.0)
        backend.generate(target_units=["s"], k=5)
        mock_sleep.assert_called_with(2.0)


# ---------------------------------------------------------------------------
# Rate limiting (request delay)
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Basic rate limiting via request_delay."""

    @patch("corpusgen.generate.backends.llm_api.time.sleep")
    @patch("corpusgen.generate.backends.llm_api._call_llm")
    def test_request_delay_applied(self, mock_call, mock_sleep):
        mock_call.return_value = _make_mock_response("Hello.")
        backend = LLMBackend(
            model="openai/gpt-4o-mini",
            request_delay=0.5,
        )
        backend.generate(target_units=["h"], k=3)
        # Should sleep for request_delay before the call
        mock_sleep.assert_any_call(0.5)

    @patch("corpusgen.generate.backends.llm_api.time.sleep")
    @patch("corpusgen.generate.backends.llm_api._call_llm")
    def test_no_delay_when_zero(self, mock_call, mock_sleep):
        mock_call.return_value = _make_mock_response("Hello.")
        backend = LLMBackend(
            model="openai/gpt-4o-mini",
            request_delay=0.0,
        )
        backend.generate(target_units=["h"], k=3)
        # sleep should not be called with 0.0 for rate limiting
        # (may still be called for retry, but not for request delay)
        for call in mock_sleep.call_args_list:
            assert call[0][0] != 0.0 or len(mock_sleep.call_args_list) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary conditions."""

    @patch("corpusgen.generate.backends.llm_api._call_llm")
    def test_empty_target_units(self, mock_call):
        mock_call.return_value = _make_mock_response("Hello.")
        backend = LLMBackend(model="openai/gpt-4o-mini")
        candidates = backend.generate(target_units=[], k=5)
        assert isinstance(candidates, list)

    @patch("corpusgen.generate.backends.llm_api._call_llm")
    def test_k_zero(self, mock_call):
        backend = LLMBackend(model="openai/gpt-4o-mini")
        candidates = backend.generate(target_units=["p"], k=0)
        assert candidates == []

    @patch("corpusgen.generate.backends.llm_api._call_llm")
    def test_llm_returns_empty_string(self, mock_call):
        mock_call.return_value = _make_mock_response("")
        backend = LLMBackend(model="openai/gpt-4o-mini")
        candidates = backend.generate(target_units=["p"], k=5)
        assert candidates == []

    @patch("corpusgen.generate.backends.llm_api._call_llm")
    def test_llm_returns_numbered_list(self, mock_call):
        """LLMs often return numbered lists — strip numbering."""
        mock_call.return_value = _make_mock_response(
            "1. The ship sailed smoothly.\n"
            "2. She thought about things.\n"
            "3. A thick fog covered the path."
        )
        backend = LLMBackend(model="openai/gpt-4o-mini")
        candidates = backend.generate(target_units=["ʃ"], k=5)
        # Sentences should have numbering stripped
        for c in candidates:
            assert not c["text"].startswith("1.")
            assert not c["text"].startswith("2.")


# ---------------------------------------------------------------------------
# Missing dependency
# ---------------------------------------------------------------------------


class TestMissingDependency:
    """Graceful handling when litellm is not installed."""

    def test_import_error_message(self):
        """LLMBackend should give a clear error if litellm is missing."""
        with patch(
            "corpusgen.generate.backends.llm_api._call_llm",
            side_effect=ImportError("No module named 'litellm'"),
        ):
            backend = LLMBackend(model="openai/gpt-4o-mini")
            with pytest.raises(ImportError):
                backend.generate(target_units=["p"], k=5)
