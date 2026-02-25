"""DATGStrategy: Phon-DATG GuidanceStrategy implementation.

Orchestrates AttributeWordIndex and LogitModulator to provide
inference-time logit steering for local text generation. Implements
the GuidanceStrategy ABC so it plugs directly into LocalBackend.

The strategy:
    1. Lazily builds an AttributeWordIndex from the tokenizer vocabulary
    2. At each ``prepare()`` call, computes attribute token IDs (words
       containing target phonetic units) and anti-attribute token IDs
       (words containing only already-covered units)
    3. At each ``modify_logits()`` call, delegates to LogitModulator
       to apply additive boosts and penalties
"""

from __future__ import annotations

import logging
from typing import Any

from corpusgen.generate.guidance import GuidanceStrategy
from corpusgen.generate.phon_ctg.targets import PhoneticTargetInventory
from corpusgen.generate.phon_datg.attribute_words import AttributeWordIndex
from corpusgen.generate.phon_datg.modulator import LogitModulator

logger = logging.getLogger(__name__)

_VALID_ANTI_ATTRIBUTE_MODES = {"covered", "frequency"}


class DATGStrategy(GuidanceStrategy):
    """Phon-DATG: inference-time logit steering via dynamic attribute graphs.

    Uses an AttributeWordIndex to identify vocabulary tokens that contain
    target phonetic units (attribute words) or only already-covered units
    (anti-attribute words), then applies additive logit adjustments via
    a LogitModulator during autoregressive generation.

    Args:
        targets: PhoneticTargetInventory for tracking covered/missing units.
        language: Language code for G2P conversion.
        boost_strength: Additive boost for attribute token logits.
        penalty_strength: Additive penalty for anti-attribute token logits.
        anti_attribute_mode: How to determine anti-attribute tokens:
            ``"covered"`` — tokens whose units are all already covered.
            ``"frequency"`` — tokens whose units all exceed a count threshold.
        frequency_threshold: Minimum count for frequency mode. Tokens are
            anti-attribute only if ALL their units have counts above this.
        attribute_word_index: Optional pre-built index. If None, one is
            created and built lazily on first ``prepare()`` call.
        batch_size: Batch size for vocabulary phonemization during index build.
    """

    def __init__(
        self,
        targets: PhoneticTargetInventory,
        language: str = "en-us",
        boost_strength: float = 5.0,
        penalty_strength: float = -5.0,
        anti_attribute_mode: str = "covered",
        frequency_threshold: int = 10,
        attribute_word_index: AttributeWordIndex | None = None,
        batch_size: int = 512,
    ) -> None:
        if anti_attribute_mode not in _VALID_ANTI_ATTRIBUTE_MODES:
            raise ValueError(
                f"anti_attribute_mode must be one of "
                f"{_VALID_ANTI_ATTRIBUTE_MODES}, got {anti_attribute_mode!r}"
            )

        self._targets = targets
        self._language = language
        self._boost_strength = boost_strength
        self._penalty_strength = penalty_strength
        self._anti_attribute_mode = anti_attribute_mode
        self._frequency_threshold = frequency_threshold
        self._batch_size = batch_size

        # Index: injected or created lazily
        self._index = attribute_word_index or AttributeWordIndex(
            language=language,
            batch_size=batch_size,
        )

        # Modulator (stateless, reusable)
        self._modulator = LogitModulator(
            boost_strength=boost_strength,
            penalty_strength=penalty_strength,
        )

        # Current attribute/anti-attribute sets (updated in prepare())
        self._current_attribute_tokens: set[int] | None = None
        self._current_anti_attribute_tokens: set[int] | None = None
        self._prepared = False

    # -------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Strategy identifier."""
        return "phon_datg"

    @property
    def language(self) -> str:
        """Language code."""
        return self._language

    @property
    def boost_strength(self) -> float:
        """Additive boost for attribute tokens."""
        return self._boost_strength

    @property
    def penalty_strength(self) -> float:
        """Additive penalty for anti-attribute tokens."""
        return self._penalty_strength

    @property
    def anti_attribute_mode(self) -> str:
        """Anti-attribute determination mode."""
        return self._anti_attribute_mode

    @property
    def frequency_threshold(self) -> int:
        """Frequency threshold for frequency mode."""
        return self._frequency_threshold

    @property
    def attribute_word_index(self) -> AttributeWordIndex:
        """The underlying AttributeWordIndex."""
        return self._index

    @property
    def current_attribute_tokens(self) -> set[int]:
        """Current attribute token IDs (after prepare()).

        Raises:
            RuntimeError: If prepare() has not been called.
        """
        if not self._prepared:
            raise RuntimeError(
                "DATGStrategy has not been prepared yet. "
                "Call prepare() first."
            )
        assert self._current_attribute_tokens is not None
        return set(self._current_attribute_tokens)

    @property
    def current_anti_attribute_tokens(self) -> set[int]:
        """Current anti-attribute token IDs (after prepare()).

        Raises:
            RuntimeError: If prepare() has not been called.
        """
        if not self._prepared:
            raise RuntimeError(
                "DATGStrategy has not been prepared yet. "
                "Call prepare() first."
            )
        assert self._current_anti_attribute_tokens is not None
        return set(self._current_anti_attribute_tokens)

    # -------------------------------------------------------------------
    # GuidanceStrategy interface
    # -------------------------------------------------------------------

    def prepare(
        self,
        target_units: list[str],
        model: Any,
        tokenizer: Any,
    ) -> None:
        """Build index lazily and compute attribute/anti-attribute sets.

        Args:
            target_units: Phonetic units to target in this generation.
            model: The HuggingFace model instance (unused here, part of ABC).
            tokenizer: The HuggingFace tokenizer instance.
        """
        # Lazy index build (idempotent)
        if not self._index.is_built:
            logger.info("Building AttributeWordIndex on first prepare()...")
            self._index.build(tokenizer)

        # Compute attribute tokens: words containing any target unit
        self._current_attribute_tokens = self._index.get_attribute_tokens(
            target_units
        )

        # Compute anti-attribute tokens (filtered to tracker's unit level)
        unit_level = self._targets.unit
        if self._anti_attribute_mode == "covered":
            covered = self._targets.covered_units
            self._current_anti_attribute_tokens = (
                self._index.get_anti_attribute_tokens(
                    covered, unit_level=unit_level
                )
            )
        else:  # frequency
            unit_counts = self._targets.tracker.phoneme_counts
            self._current_anti_attribute_tokens = (
                self._index.get_anti_attribute_tokens_by_frequency(
                    unit_counts,
                    threshold=self._frequency_threshold,
                    unit_level=unit_level,
                )
            )

        self._prepared = True

        logger.info(
            "DATGStrategy prepared: %d attribute tokens, %d anti-attribute tokens",
            len(self._current_attribute_tokens),
            len(self._current_anti_attribute_tokens),
        )

    def modify_logits(self, input_ids: Any, logits: Any) -> Any:
        """Apply logit modulation via the LogitModulator.

        Args:
            input_ids: Current token sequence (unused, part of ABC).
            logits: Raw logits tensor [batch, vocab_size].

        Returns:
            Modified logits tensor (same shape).

        Raises:
            RuntimeError: If prepare() has not been called.
        """
        if not self._prepared:
            raise RuntimeError(
                "DATGStrategy.modify_logits() called before prepare(). "
                "Call prepare() first."
            )

        assert self._current_attribute_tokens is not None
        assert self._current_anti_attribute_tokens is not None

        return self._modulator.modulate(
            logits=logits,
            attribute_token_ids=self._current_attribute_tokens,
            anti_attribute_token_ids=self._current_anti_attribute_tokens,
        )
