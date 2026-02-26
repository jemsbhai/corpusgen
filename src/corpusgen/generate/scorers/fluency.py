"""Perplexity-based fluency scorer for Phon-CTG candidate evaluation.

Scores text by its perplexity under a pretrained causal language model.
Lower perplexity indicates more fluent, natural text. The raw perplexity
is normalized to a [0, 1] score where higher = more fluent.

Requires torch and transformers (optional dependency group ``local``).

Model loading is lazy — the model is not loaded until the first scoring
call. Alternatively, use ``from_model()`` to inject an already-loaded
model and tokenizer (e.g., to share with the LocalBackend).

Normalization:
    score = 1 / (1 + log(perplexity))
    where perplexity = exp(loss). This maps:
        - perplexity 1.0 (perfect) → score 1.0
        - perplexity e (~2.7)      → score 0.5
        - perplexity → ∞           → score → 0.0
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Isolated helpers (mockable in tests)
# ---------------------------------------------------------------------------


def _load_tokenizer(model_name: str) -> Any:
    """Load a HuggingFace tokenizer. Isolated for mockability.

    Raises:
        ImportError: If transformers is not installed.
    """
    from transformers import AutoTokenizer  # type: ignore[import-untyped]

    return AutoTokenizer.from_pretrained(model_name)


def _load_model(model_name: str, device: str) -> Any:
    """Load a HuggingFace causal LM. Isolated for mockability.

    Raises:
        ImportError: If transformers or torch is not installed.
    """
    from transformers import AutoModelForCausalLM  # type: ignore[import-untyped]

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return model


def _detect_device() -> str:
    """Auto-detect the best available device."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ---------------------------------------------------------------------------
# PerplexityFluencyScorer
# ---------------------------------------------------------------------------


class PerplexityFluencyScorer:
    """Fluency scorer based on causal LM perplexity.

    Callable interface: ``scorer(text) -> float`` in [0, 1].

    Higher scores indicate more fluent text (lower perplexity).

    Args:
        model_name: HuggingFace model ID (e.g., "gpt2").
        device: Device string ("cuda", "cpu", "auto"). If None,
            auto-detects.

    Raises:
        ImportError: On first call if torch/transformers not installed.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None
        self._tokenizer: Any = None

    # -------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """HuggingFace model ID."""
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        """Whether the model and tokenizer have been loaded."""
        return self._model is not None and self._tokenizer is not None

    # -------------------------------------------------------------------
    # Alternative constructor
    # -------------------------------------------------------------------

    @classmethod
    def from_model(cls, model: Any, tokenizer: Any) -> PerplexityFluencyScorer:
        """Create a scorer from an already-loaded model and tokenizer.

        Use this to share a model with the LocalBackend, avoiding
        loading the same model twice.

        Args:
            model: A HuggingFace causal LM instance.
            tokenizer: The corresponding tokenizer.

        Returns:
            A PerplexityFluencyScorer with the model pre-loaded.
        """
        scorer = cls.__new__(cls)
        scorer._model_name = getattr(model, "name_or_path", "unknown")
        scorer._device = None
        scorer._model = model
        scorer._tokenizer = tokenizer
        return scorer

    # -------------------------------------------------------------------
    # Lazy loading
    # -------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load model and tokenizer if not already loaded."""
        if self.is_loaded:
            return

        if self._device is None:
            self._device = _detect_device()

        logger.info(
            "Loading fluency model %s on %s",
            self._model_name,
            self._device,
        )

        self._tokenizer = _load_tokenizer(self._model_name)
        self._model = _load_model(self._model_name, self._device)

        # Ensure pad token
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        logger.info("Fluency model loaded successfully.")

    # -------------------------------------------------------------------
    # Scoring
    # -------------------------------------------------------------------

    def __call__(self, text: str | None) -> float:
        """Score text for fluency via perplexity.

        Args:
            text: Text to score. Returns 0.0 for None or empty/whitespace.

        Returns:
            Float in [0, 1]. Higher = more fluent (lower perplexity).
        """
        if text is None or not text.strip():
            return 0.0

        self._ensure_loaded()

        import torch

        # Tokenize
        inputs = self._tokenizer(
            text.strip(),
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        input_ids = inputs["input_ids"].to(self._model.device)

        # Compute loss (negative log-likelihood per token)
        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss.item()

        # loss = log(perplexity), so perplexity = exp(loss)
        # Normalize: score = 1 / (1 + loss)
        # This maps loss=0 → 1.0, loss=∞ → 0.0, monotonically decreasing
        score = 1.0 / (1.0 + loss)

        return score
