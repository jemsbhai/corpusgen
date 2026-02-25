"""LLMBackend: multi-provider LLM generation via litellm.

Prompts LLMs to generate sentences targeting specific phonetic units.
Uses litellm for unified access to 100+ providers (OpenAI, Anthropic,
Mistral, Google, HuggingFace, etc.). Users bring their own API keys.

Includes basic retry logic and rate limiting. Generated text is
automatically phonemized via G2P before being returned as candidates.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from corpusgen.generate.phon_ctg.loop import GenerationBackend


logger = logging.getLogger(__name__)


DEFAULT_PROMPT_TEMPLATE = """Generate {k} short, natural sentences in {language} that contain the following sounds (IPA phonemes):

Target sounds: {target_units}

Requirements:
- Each sentence should be a complete, grammatically correct sentence.
- Sentences should sound natural, not contrived.
- Try to include as many of the target sounds as possible in each sentence.
- One sentence per line, no numbering or bullet points.
"""


def _call_llm(**kwargs: Any) -> Any:
    """Call litellm.completion. Isolated for mockability.

    Raises:
        ImportError: If litellm is not installed.
    """
    from litellm import completion  # type: ignore[import-untyped]

    return completion(**kwargs)


def _clean_line(line: str) -> str:
    """Strip numbering, bullets, and whitespace from a response line."""
    stripped = line.strip()
    # Remove common numbering patterns: "1.", "1)", "1:", "- ", "* "
    stripped = re.sub(r"^\d+[\.\)\:]\s*", "", stripped)
    stripped = re.sub(r"^[-\*]\s+", "", stripped)
    return stripped.strip()


class LLMBackend(GenerationBackend):
    """Generation backend using LLM APIs via litellm.

    Prompts an LLM to generate sentences targeting specific phonetic
    units, then phonemizes the output via G2P.

    Args:
        model: litellm model string (e.g., "openai/gpt-4o-mini",
            "anthropic/claude-3-5-sonnet-latest").
        language: Language code for G2P and prompt context.
        api_key: API key for the provider. If None, litellm falls
            back to environment variables.
        prompt_template: Custom prompt template. Must contain
            ``{target_units}`` placeholder. May also contain
            ``{language}`` and ``{k}``.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in LLM response.
        max_retries: Maximum retry attempts on transient failures.
        retry_delay: Seconds to wait between retries.
        request_delay: Seconds to wait before each API call
            (basic rate limiting).
    """

    def __init__(
        self,
        model: str,
        language: str = "en-us",
        api_key: str | None = None,
        prompt_template: str | None = None,
        temperature: float = 0.8,
        max_tokens: int = 1024,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        request_delay: float = 0.0,
    ) -> None:
        self._model = model
        self._language = language
        self._api_key = api_key
        self._prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._request_delay = request_delay

    # -------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "llm_api"

    @property
    def model(self) -> str:
        """litellm model string."""
        return self._model

    @property
    def language(self) -> str:
        """Language code."""
        return self._language

    @property
    def api_key(self) -> str | None:
        """API key (if provided)."""
        return self._api_key

    @property
    def prompt_template(self) -> str:
        """Prompt template string."""
        return self._prompt_template

    @property
    def max_retries(self) -> int:
        """Maximum retry attempts."""
        return self._max_retries

    @property
    def retry_delay(self) -> float:
        """Seconds between retries."""
        return self._retry_delay

    @property
    def request_delay(self) -> float:
        """Seconds before each API call."""
        return self._request_delay

    # -------------------------------------------------------------------
    # Prompt formatting
    # -------------------------------------------------------------------

    def format_prompt(
        self,
        target_units: list[str],
        k: int = 5,
    ) -> str:
        """Format the prompt template with target units.

        Args:
            target_units: Phonetic units to target.
            k: Number of sentences to request.

        Returns:
            Formatted prompt string.
        """
        units_str = ", ".join(target_units) if target_units else "(any)"
        return self._prompt_template.format(
            target_units=units_str,
            language=self._language,
            k=k,
        )

    # -------------------------------------------------------------------
    # LLM call with retry
    # -------------------------------------------------------------------

    def _call_with_retry(self, **kwargs: Any) -> Any | None:
        """Call the LLM with retry logic.

        Returns the response, or None if all attempts fail.
        """
        last_error: Exception | None = None

        for attempt in range(1 + self._max_retries):
            try:
                if self._request_delay > 0:
                    time.sleep(self._request_delay)
                return _call_llm(**kwargs)
            except ImportError:
                raise  # missing package — retrying won't help
            except Exception as e:
                last_error = e
                if attempt < self._max_retries:
                    logger.warning(
                        "LLM call attempt %d/%d failed: %s. Retrying...",
                        attempt + 1,
                        1 + self._max_retries,
                        str(e),
                    )
                    time.sleep(self._retry_delay)
                else:
                    logger.error(
                        "LLM call failed after %d attempts: %s",
                        1 + self._max_retries,
                        str(e),
                    )

        return None

    # -------------------------------------------------------------------
    # Generate
    # -------------------------------------------------------------------

    def generate(
        self,
        target_units: list[str],
        k: int = 5,
        **kwargs: Any,
    ) -> list[dict]:
        """Generate candidate sentences via LLM.

        Args:
            target_units: Phonetic units to target.
            k: Number of candidates to generate.

        Returns:
            List of candidate dicts with "text" and "phonemes".
        """
        if k <= 0:
            return []

        prompt = self.format_prompt(target_units=target_units, k=k)

        # Build litellm call kwargs
        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if self._api_key is not None:
            call_kwargs["api_key"] = self._api_key

        # Call LLM
        response = self._call_with_retry(**call_kwargs)
        if response is None:
            return []

        # Extract text from response
        raw_text = response.choices[0].message.content or ""

        # Split into individual sentences (one per line)
        lines = raw_text.strip().split("\n")
        sentences = [_clean_line(line) for line in lines]
        sentences = [s for s in sentences if s]

        # Limit to k
        sentences = sentences[:k]

        if not sentences:
            return []

        # Phonemize via G2P
        from corpusgen.g2p.manager import G2PManager

        g2p = G2PManager()
        g2p_results = g2p.phonemize_batch(sentences, language=self._language)

        candidates: list[dict] = []
        for text, g2p_result in zip(sentences, g2p_results):
            if g2p_result.phonemes:
                candidates.append({
                    "text": text,
                    "phonemes": g2p_result.phonemes,
                })

        return candidates
