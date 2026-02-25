"""CoverageTracker: tracks phoneme/diphone/triphone coverage state."""

from __future__ import annotations

from collections import defaultdict
from itertools import product


class CoverageTracker:
    """Tracks coverage of phonetic units against a target inventory.

    Supports phoneme, diphone, and triphone units. Maintains frequency
    counts and sentence-level provenance for each covered unit.

    Args:
        target_phonemes: List of phonemes in the target inventory.
        unit: Coverage unit type — "phoneme", "diphone", or "triphone".
    """

    _VALID_UNITS = ("phoneme", "diphone", "triphone")

    def __init__(self, target_phonemes: list[str], unit: str = "phoneme") -> None:
        if unit not in self._VALID_UNITS:
            raise ValueError(
                f"Invalid unit: {unit!r}. Must be one of {self._VALID_UNITS}"
            )
        self._unit = unit
        self._target_phonemes = list(target_phonemes)

        # Build the target set based on unit type
        if unit == "phoneme":
            self._target_set = set(target_phonemes)
        elif unit == "diphone":
            self._target_set = {
                f"{a}-{b}" for a, b in product(target_phonemes, repeat=2)
            }
        elif unit == "triphone":
            self._target_set = {
                f"{a}-{b}-{c}" for a, b, c in product(target_phonemes, repeat=3)
            }

        # State
        self._covered: set[str] = set()
        self._counts: dict[str, int] = defaultdict(int)
        self._sources: dict[str, list[int]] = defaultdict(list)

    @property
    def unit(self) -> str:
        """Coverage unit type."""
        return self._unit

    @property
    def target_size(self) -> int:
        """Number of units in the target inventory."""
        return len(self._target_set)

    @property
    def covered_count(self) -> int:
        """Number of target units covered so far."""
        return len(self._covered)

    @property
    def coverage(self) -> float:
        """Fraction of target units covered (0.0 to 1.0)."""
        if self.target_size == 0:
            return 1.0
        return self.covered_count / self.target_size

    @property
    def missing(self) -> set[str]:
        """Set of target units not yet covered."""
        return self._target_set - self._covered

    @property
    def phoneme_counts(self) -> dict[str, int]:
        """Per-unit occurrence counts (all occurrences, not just first)."""
        return dict(self._counts)

    @property
    def phoneme_sources(self) -> dict[str, list[int]]:
        """Maps each covered unit to the sentence indices where it appeared."""
        return dict(self._sources)

    def update(self, phonemes: list[str], sentence_index: int) -> None:
        """Update coverage state with phonemes from a sentence.

        Args:
            phonemes: List of phonemes extracted from the sentence.
            sentence_index: Index of the sentence in the corpus.
        """
        if self._unit == "phoneme":
            units = phonemes
        elif self._unit == "diphone":
            units = [
                f"{phonemes[i]}-{phonemes[i + 1]}"
                for i in range(len(phonemes) - 1)
            ]
        elif self._unit == "triphone":
            units = [
                f"{phonemes[i]}-{phonemes[i + 1]}-{phonemes[i + 2]}"
                for i in range(len(phonemes) - 2)
            ]
        else:
            units = []

        for u in units:
            if u in self._target_set:
                self._counts[u] += 1
                if u not in self._covered:
                    self._covered.add(u)
                if sentence_index not in self._sources[u]:
                    self._sources[u].append(sentence_index)

    def reset(self) -> None:
        """Reset all coverage state, keeping the target inventory."""
        self._covered.clear()
        self._counts.clear()
        self._sources.clear()
