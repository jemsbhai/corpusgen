"""LocalBackend: HuggingFace transformers local generation.

Loads a causal language model locally and generates sentences targeting
specific phonetic units. Supports quantization via bitsandbytes (4-bit/8-bit)
for VRAM-constrained GPUs, and accepts a GuidanceStrategy for inference-time
logit steering (Phon-DATG) or RL-adjusted generation (Phon-RL).

Model and tokenizer are loaded lazily on first ``generate()`` call.
Generated text is automatically phonemized via G2P before being returned.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from corpusgen.generate.guidance import GuidanceStrategy
from corpusgen.generate.phon_ctg.loop import GenerationBackend


logger = logging.getLogger(__name__)


DEFAULT_PROMPT_TEMPLATE = (
    "Generate {k} short, natural sentences containing these sounds: {target_units}\n"
    "One sentence per line, no numbering."
)

_VALID_QUANTIZATIONS = {None, "4bit", "8bit"}


# ---------------------------------------------------------------------------
# Isolated helpers (mockable in tests)
# ---------------------------------------------------------------------------


def _cuda_available() -> bool:
    """Check CUDA availability. Isolated for mockability."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _detect_device() -> str:
    """Auto-detect the best available device."""
    return "cuda" if _cuda_available() else "cpu"


def _load_tokenizer(model_name: str, **kwargs: Any) -> Any:
    """Load a HuggingFace tokenizer. Isolated for mockability.

    Raises:
        ImportError: If transformers is not installed.
    """
    from transformers import AutoTokenizer  # type: ignore[import-untyped]

    return AutoTokenizer.from_pretrained(model_name, **kwargs)


def _load_model(
    model_name: str,
    device: str,
    quantization: str | None = None,
    **kwargs: Any,
) -> Any:
    """Load a HuggingFace causal LM. Isolated for mockability.

    Raises:
        ImportError: If transformers or torch is not installed.
    """
    from transformers import AutoModelForCausalLM  # type: ignore[import-untyped]

    load_kwargs: dict[str, Any] = dict(kwargs)

    if quantization is not None:
        from transformers import BitsAndBytesConfig  # type: ignore[import-untyped]

        if quantization == "4bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="float16",
            )
        elif quantization == "8bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        # Quantized models handle device placement internally
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    return model


def _clean_line(line: str) -> str:
    """Strip numbering, bullets, and whitespace from a generated line."""
    stripped = line.strip()
    stripped = re.sub(r"^\d+[\.\)\:]\s*", "", stripped)
    stripped = re.sub(r"^[-\*]\s+", "", stripped)
    return stripped.strip()


# ---------------------------------------------------------------------------
# LocalBackend
# ---------------------------------------------------------------------------


class LocalBackend(GenerationBackend):
    """Generation backend using a local HuggingFace transformers model.

    Loads a causal LM and generates text conditioned on phonetic targets
    via prompt formatting. Supports quantization for VRAM efficiency and
    an optional GuidanceStrategy for inference-time logit modification.

    Model and tokenizer are loaded lazily on the first ``generate()`` call.

    Args:
        model_name: HuggingFace model ID or local path.
        language: Language code for G2P and prompt context.
        prompt_template: Custom prompt template. Must contain
            ``{target_units}`` placeholder. May also contain
            ``{language}`` and ``{k}``. Defaults to a short template
            suited for smaller models.
        device: Device string (``"cuda"``, ``"cpu"``). If None,
            auto-detects CUDA with CPU fallback.
        quantization: Quantization mode: ``None``, ``"4bit"``, or
            ``"8bit"``. Requires bitsandbytes when non-None.
        guidance_strategy: Optional GuidanceStrategy instance for
            inference-time logit steering (Phon-DATG or Phon-RL).
        max_new_tokens: Maximum new tokens to generate per sequence.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        do_sample: Whether to sample (True) or use greedy decoding.
        model_kwargs: Extra keyword arguments forwarded to
            ``AutoModelForCausalLM.from_pretrained()``.
    """

    def __init__(
        self,
        model_name: str,
        language: str = "en-us",
        prompt_template: str | None = None,
        device: str | None = None,
        quantization: str | None = None,
        guidance_strategy: GuidanceStrategy | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        do_sample: bool = True,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if quantization not in _VALID_QUANTIZATIONS:
            raise ValueError(
                f"quantization must be one of {_VALID_QUANTIZATIONS}, "
                f"got {quantization!r}"
            )

        self._model_name = model_name
        self._language = language
        self._prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        self._device = device
        self._quantization = quantization
        self._guidance_strategy = guidance_strategy
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._do_sample = do_sample
        self._model_kwargs = model_kwargs or {}

        # Lazy-loaded
        self._model: Any = None
        self._tokenizer: Any = None

    # -------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "local"

    @property
    def model_name(self) -> str:
        """HuggingFace model ID or local path."""
        return self._model_name

    @property
    def language(self) -> str:
        """Language code."""
        return self._language

    @property
    def prompt_template(self) -> str:
        """Prompt template string."""
        return self._prompt_template

    @property
    def device(self) -> str | None:
        """Device string, or None if not yet resolved."""
        return self._device

    @property
    def quantization(self) -> str | None:
        """Quantization mode."""
        return self._quantization

    @property
    def guidance_strategy(self) -> GuidanceStrategy | None:
        """Optional guidance strategy."""
        return self._guidance_strategy

    @property
    def max_new_tokens(self) -> int:
        """Maximum new tokens per generated sequence."""
        return self._max_new_tokens

    @property
    def temperature(self) -> float:
        """Sampling temperature."""
        return self._temperature

    @property
    def top_p(self) -> float:
        """Nucleus sampling threshold."""
        return self._top_p

    @property
    def do_sample(self) -> bool:
        """Whether to use sampling."""
        return self._do_sample

    @property
    def model_kwargs(self) -> dict[str, Any]:
        """Extra kwargs for model loading."""
        return dict(self._model_kwargs)

    @property
    def is_loaded(self) -> bool:
        """Whether the model and tokenizer have been loaded."""
        return self._model is not None and self._tokenizer is not None

    # -------------------------------------------------------------------
    # Lazy loading
    # -------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load model and tokenizer if not already loaded."""
        if self.is_loaded:
            return

        # Resolve device
        if self._device is None:
            self._device = _detect_device()
            logger.info("Auto-detected device: %s", self._device)

        logger.info(
            "Loading model %s (device=%s, quantization=%s)",
            self._model_name,
            self._device,
            self._quantization,
        )

        self._tokenizer = _load_tokenizer(
            self._model_name,
            **{k: v for k, v in self._model_kwargs.items()
               if k == "trust_remote_code"},
        )

        self._model = _load_model(
            self._model_name,
            device=self._device,
            quantization=self._quantization,
            **self._model_kwargs,
        )

        # Ensure pad token is set
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        logger.info("Model loaded successfully.")

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
    # Generate
    # -------------------------------------------------------------------

    def generate(
        self,
        target_units: list[str],
        k: int = 5,
        **kwargs: Any,
    ) -> list[dict]:
        """Generate candidate sentences using the local model.

        Args:
            target_units: Phonetic units to target.
            k: Number of candidates to generate.

        Returns:
            List of candidate dicts with ``"text"`` and ``"phonemes"``.
        """
        if k <= 0:
            return []

        self._ensure_loaded()

        prompt = self.format_prompt(target_units=target_units, k=k)

        # Tokenize
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(self._model.device)

        # Build generate kwargs
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self._max_new_tokens,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "do_sample": self._do_sample,
            "num_return_sequences": k,
            "pad_token_id": self._tokenizer.eos_token_id,
        }

        # Guidance strategy: prepare + logits processor
        if self._guidance_strategy is not None:
            self._guidance_strategy.prepare(
                target_units=target_units,
                model=self._model,
                tokenizer=self._tokenizer,
            )
            gen_kwargs["logits_processor"] = [
                _GuidanceLogitsProcessor(self._guidance_strategy)
            ]

        # Generate
        output_ids = self._model.generate(
            **inputs,
            **gen_kwargs,
        )

        # Decode — strip the prompt tokens
        prompt_len = inputs.input_ids.shape[-1]
        new_ids = output_ids[:, prompt_len:]
        texts = self._tokenizer.batch_decode(
            new_ids,
            skip_special_tokens=True,
        )

        # Extract individual sentences from each generated sequence
        sentences: list[str] = []
        for text in texts:
            lines = text.strip().split("\n")
            for line in lines:
                cleaned = _clean_line(line)
                if cleaned:
                    sentences.append(cleaned)

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


# ---------------------------------------------------------------------------
# Logits processor wrapper
# ---------------------------------------------------------------------------


class _GuidanceLogitsProcessor:
    """Wraps a GuidanceStrategy into a HuggingFace LogitsProcessor.

    Compatible with the ``logits_processor`` argument of
    ``model.generate()``.
    """

    def __init__(self, strategy: GuidanceStrategy) -> None:
        self._strategy = strategy

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        return self._strategy.modify_logits(input_ids, scores)
