"""SelectorBase: abstract base class for all sentence selection algorithms."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from corpusgen.select.result import SelectionResult
from corpusgen.weights import validate_unit_weights


class SelectorBase(ABC):
    """Abstract base class for sentence selection algorithms.

    All selectors receive pre-phonemized candidates and a target unit set,
    keeping the selection logic independent of G2P implementation.

    Args:
        unit: Coverage unit type — "phoneme", "diphone", or "triphone".
    """

    _VALID_UNITS = ("phoneme", "diphone", "triphone")

    def __init__(self, unit: str = "phoneme") -> None:
        if unit not in self._VALID_UNITS:
            raise ValueError(
                f"Invalid unit: {unit!r}. Must be one of {self._VALID_UNITS}"
            )
        self._unit = unit

    @property
    def unit(self) -> str:
        """Coverage unit type."""
        return self._unit

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """Short identifier for this algorithm (e.g. 'greedy', 'celf')."""
        ...

    @abstractmethod
    def select(
        self,
        candidates: list[str],
        candidate_phonemes: list[list[str]],
        target_units: set[str],
        max_sentences: int | None = None,
        target_coverage: float = 1.0,
        weights: dict[str, float] | None = None,
    ) -> SelectionResult:
        """Select sentences from candidates to maximize coverage of target units.

        Args:
            candidates: List of candidate sentences (raw text).
            candidate_phonemes: Pre-phonemized form of each candidate.
                candidate_phonemes[i] is the phoneme list for candidates[i].
            target_units: Set of units to cover (phonemes, diphones, or triphones).
            max_sentences: Maximum number of sentences to select (budget).
                None means no budget limit.
            target_coverage: Stop when this coverage fraction is reached.
                Defaults to 1.0 (full coverage).
            weights: Optional mapping from unit to weight for marginal gain.
                If None, all units are weighted equally (1.0).

        Returns:
            SelectionResult with selected sentences and coverage metrics.
        """
        ...

    def _extract_units(self, phonemes: list[str]) -> set[str]:
        """Extract coverage units from a phoneme sequence.

        Args:
            phonemes: List of phonemes from a single sentence.

        Returns:
            Set of units (phonemes, diphones, or triphones) present.
        """
        if self._unit == "phoneme":
            return set(phonemes)
        elif self._unit == "diphone":
            return {
                f"{phonemes[i]}-{phonemes[i + 1]}"
                for i in range(len(phonemes) - 1)
            }
        elif self._unit == "triphone":
            return {
                f"{phonemes[i]}-{phonemes[i + 1]}-{phonemes[i + 2]}"
                for i in range(len(phonemes) - 2)
            }
        return set()

    @staticmethod
    def _weighted_gain(
        new_units: set[str],
        weights: dict[str, float] | None,
    ) -> float:
        """Compute marginal gain, optionally weighted.

        Args:
            new_units: Set of newly covered units.
            weights: Optional unit-to-weight mapping.

        Returns:
            Sum of weights for new units, or count if unweighted.
        """
        if not new_units:
            return 0.0
        if weights is None:
            return float(len(new_units))
        return sum(weights.get(u, 1.0) for u in new_units)

    @staticmethod
    def _validate_limits(
        max_sentences: int | None,
        target_coverage: float,
    ) -> None:
        """Validate budget and stopping-threshold arguments."""
        if max_sentences is not None and max_sentences < 0:
            raise ValueError(
                f"max_sentences must be non-negative or None, got {max_sentences}"
            )
        if not 0.0 <= target_coverage <= 1.0:
            raise ValueError(
                f"target_coverage must be between 0.0 and 1.0, got {target_coverage}"
            )

    @staticmethod
    def _normalize_target_distribution(
        target_distribution: dict[str, float],
    ) -> dict[str, float]:
        """Validate and normalize a strictly positive finite distribution."""
        if not target_distribution:
            raise ValueError("target_distribution must be non-empty")
        if any(
            not math.isfinite(value) or value <= 0
            for value in target_distribution.values()
        ):
            raise ValueError(
                "All target_distribution values must be positive and finite"
            )
        total = sum(target_distribution.values())
        if not math.isfinite(total) or total <= 0:
            raise ValueError("target_distribution total must be positive and finite")
        return {
            unit: value / total
            for unit, value in target_distribution.items()
        }

    @classmethod
    def _validate_select_inputs(
        cls,
        candidates: list[str],
        candidate_phonemes: list[list[str]],
        max_sentences: int | None,
        target_coverage: float,
        weights: dict[str, float] | None,
    ) -> None:
        """Validate arguments common to every selector implementation."""
        if len(candidate_phonemes) != len(candidates):
            raise ValueError(
                f"candidate_phonemes length ({len(candidate_phonemes)}) must match "
                f"candidates length ({len(candidates)})"
            )
        cls._validate_limits(max_sentences, target_coverage)
        validate_unit_weights(weights)

    def _extract_unit_list(self, phonemes: list[str]) -> list[str]:
        """Extract coverage units preserving duplicates for frequency counting.

        Unlike ``_extract_units`` which returns a set, this preserves
        repeated occurrences for distributional analysis.

        Args:
            phonemes: List of phonemes from a single sentence.

        Returns:
            List of units with duplicates preserved.
        """
        if self._unit == "phoneme":
            return list(phonemes)
        elif self._unit == "diphone":
            return [
                f"{phonemes[i]}-{phonemes[i + 1]}"
                for i in range(len(phonemes) - 1)
            ]
        elif self._unit == "triphone":
            return [
                f"{phonemes[i]}-{phonemes[i + 1]}-{phonemes[i + 2]}"
                for i in range(len(phonemes) - 2)
            ]
        return []
