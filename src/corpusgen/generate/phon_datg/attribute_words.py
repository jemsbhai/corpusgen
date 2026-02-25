"""AttributeWordIndex: maps phonetic units to vocabulary token IDs.

Given a tokenizer's vocabulary and a G2P backend, this module
phonemizes every token and builds a bidirectional index:

- **unit → token IDs**: which tokens contain a given phoneme, diphone,
  or triphone.
- **token ID → units**: which phonetic units a given token covers.

This index is the foundation for Phon-DATG's logit modulation:
attribute words (tokens containing target units) get boosted,
anti-attribute words (tokens containing only already-covered units)
get penalized.

The index is built once (expensive — full vocabulary phonemization)
and then supports fast set lookups.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Isolated helper (mockable in tests)
# ---------------------------------------------------------------------------


def _phonemize_batch(texts: list[str], language: str = "en-us") -> list:
    """Phonemize a batch of texts via G2PManager. Isolated for mockability.

    Returns:
        List of G2PResult objects, each with a ``.phonemes`` attribute.
    """
    from corpusgen.g2p.manager import G2PManager

    g2p = G2PManager()
    return g2p.phonemize_batch(texts, language=language)


def _filter_by_level(units: set[str], unit_level: str | None) -> set[str]:
    """Filter a set of units to only those at a specific level.

    Args:
        units: Set of unit strings (phonemes, diphones, triphones).
        unit_level: ``"phoneme"``, ``"diphone"``, ``"triphone"``,
            or None (return all).

    Returns:
        Filtered set of units.
    """
    if unit_level is None:
        return units

    if unit_level == "phoneme":
        return {u for u in units if "-" not in u}
    elif unit_level == "diphone":
        return {u for u in units if u.count("-") == 1}
    elif unit_level == "triphone":
        return {u for u in units if u.count("-") == 2}
    else:
        return units


def _extract_units(phonemes: list[str]) -> set[str]:
    """Extract all phoneme, diphone, and triphone units from a phoneme list.

    Args:
        phonemes: Ordered list of phoneme symbols.

    Returns:
        Set of unit strings (phonemes, "X-Y" diphones, "X-Y-Z" triphones).
    """
    units: set[str] = set()

    # Phonemes
    units.update(phonemes)

    # Diphones
    for i in range(len(phonemes) - 1):
        units.add(f"{phonemes[i]}-{phonemes[i + 1]}")

    # Triphones
    for i in range(len(phonemes) - 2):
        units.add(f"{phonemes[i]}-{phonemes[i + 1]}-{phonemes[i + 2]}")

    return units


# ---------------------------------------------------------------------------
# AttributeWordIndex
# ---------------------------------------------------------------------------


class AttributeWordIndex:
    """Bidirectional index mapping phonetic units ↔ vocabulary token IDs.

    Built lazily from a tokenizer's vocabulary via G2P phonemization.
    Once built, supports fast lookups for attribute and anti-attribute
    token sets used by Phon-DATG logit modulation.

    Args:
        language: Language code for G2P conversion.
        batch_size: Number of tokens to phonemize per G2P batch call.
    """

    def __init__(
        self,
        language: str = "en-us",
        batch_size: int = 512,
    ) -> None:
        self._language = language
        self._batch_size = batch_size
        self._unit_to_tokens: dict[str, set[int]] | None = None
        self._token_to_units: dict[int, set[str]] | None = None
        self._built = False

    # -------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------

    @property
    def language(self) -> str:
        """Language code."""
        return self._language

    @property
    def is_built(self) -> bool:
        """Whether the index has been built."""
        return self._built

    @property
    def unit_to_tokens(self) -> dict[str, set[int]]:
        """Mapping of phonetic unit → set of token IDs (read-only copy).

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        self._ensure_built()
        assert self._unit_to_tokens is not None
        return {k: set(v) for k, v in self._unit_to_tokens.items()}

    @property
    def token_units(self) -> dict[int, set[str]]:
        """Mapping of token ID → set of phonetic units (read-only copy).

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        self._ensure_built()
        assert self._token_to_units is not None
        return {k: set(v) for k, v in self._token_to_units.items()}

    # -------------------------------------------------------------------
    # Build
    # -------------------------------------------------------------------

    def build(self, tokenizer: Any) -> None:
        """Build the index by phonemizing the tokenizer's full vocabulary.

        This is an expensive one-time operation. Subsequent calls are
        no-ops (idempotent).

        Args:
            tokenizer: A HuggingFace tokenizer (or any object with
                ``get_vocab()`` returning ``dict[str, int]`` and
                ``decode(token_id)`` returning ``str``).
        """
        if self._built:
            return

        vocab: dict[str, int] = tokenizer.get_vocab()
        logger.info(
            "Building AttributeWordIndex for %d tokens (language=%s)",
            len(vocab),
            self._language,
        )

        unit_to_tokens: dict[str, set[int]] = defaultdict(set)
        token_to_units: dict[int, set[str]] = {}

        # Process vocabulary in batches
        token_items = list(vocab.items())
        for batch_start in range(0, len(token_items), self._batch_size):
            batch = token_items[batch_start : batch_start + self._batch_size]
            token_strings = [token_str for token_str, _ in batch]
            token_ids = [token_id for _, token_id in batch]

            # Decode tokens to clean text for G2P
            decoded = [
                tokenizer.decode(tid, skip_special_tokens=True)
                for tid in token_ids
            ]

            # Phonemize the batch
            g2p_results = _phonemize_batch(decoded, language=self._language)

            for tid, g2p_result in zip(token_ids, g2p_results):
                phonemes = g2p_result.phonemes
                if not phonemes:
                    continue

                units = _extract_units(phonemes)
                token_to_units[tid] = units

                for unit in units:
                    unit_to_tokens[unit].add(tid)

        self._unit_to_tokens = dict(unit_to_tokens)
        self._token_to_units = token_to_units
        self._built = True

        logger.info(
            "AttributeWordIndex built: %d units, %d tokens indexed",
            len(self._unit_to_tokens),
            len(self._token_to_units),
        )

    # -------------------------------------------------------------------
    # Lookups
    # -------------------------------------------------------------------

    def get_attribute_tokens(self, target_units: list[str]) -> set[int]:
        """Get token IDs for words containing ANY of the target units.

        Args:
            target_units: Phonetic units to target.

        Returns:
            Set of token IDs whose decoded text contains at least one
            of the target units.

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        self._ensure_built()
        assert self._unit_to_tokens is not None

        if not target_units:
            return set()

        result: set[int] = set()
        for unit in target_units:
            result.update(self._unit_to_tokens.get(unit, set()))
        return result

    def get_anti_attribute_tokens(
        self,
        covered_units: set[str],
        unit_level: str | None = None,
    ) -> set[int]:
        """Get token IDs for words whose units are ALL already covered.

        These tokens contribute nothing new to coverage and can be
        penalized during generation to steer towards uncovered units.

        Args:
            covered_units: Set of phonetic units already covered.
            unit_level: If provided, only check units at this level:
                ``"phoneme"``, ``"diphone"``, or ``"triphone"``.
                If None, checks all unit levels.

        Returns:
            Set of token IDs whose phonetic units (at the specified
            level) are all in ``covered_units``.

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        self._ensure_built()
        assert self._token_to_units is not None

        if not covered_units:
            return set()

        result: set[int] = set()
        for tid, units in self._token_to_units.items():
            filtered = _filter_by_level(units, unit_level)
            if filtered and filtered <= covered_units:
                result.add(tid)
        return result

    def get_anti_attribute_tokens_by_frequency(
        self,
        unit_counts: dict[str, int],
        threshold: int,
        unit_level: str | None = None,
    ) -> set[int]:
        """Get token IDs for words whose units all exceed a frequency threshold.

        A fine-grained alternative to ``get_anti_attribute_tokens`` that
        uses actual coverage counts rather than a binary covered/uncovered
        distinction.

        Args:
            unit_counts: Mapping of phonetic unit → occurrence count.
            threshold: Minimum count. Tokens are included only if ALL
                their units have counts strictly greater than this value.
            unit_level: If provided, only check units at this level:
                ``"phoneme"``, ``"diphone"``, or ``"triphone"``.
                If None, checks all unit levels.

        Returns:
            Set of token IDs whose phonetic units (at the specified
            level) all have counts above ``threshold``.

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        self._ensure_built()
        assert self._token_to_units is not None

        result: set[int] = set()
        for tid, units in self._token_to_units.items():
            filtered = _filter_by_level(units, unit_level)
            if filtered and all(
                unit_counts.get(u, 0) > threshold for u in filtered
            ):
                result.add(tid)
        return result

    # -------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------

    def _ensure_built(self) -> None:
        """Raise if the index has not been built."""
        if not self._built:
            raise RuntimeError(
                "AttributeWordIndex has not been built yet. "
                "Call build(tokenizer) first."
            )
