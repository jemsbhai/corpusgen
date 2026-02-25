"""GreedySelector: standard greedy Set Cover algorithm for sentence selection."""

from __future__ import annotations

import time

from corpusgen.select.base import SelectorBase
from corpusgen.select.result import SelectionResult


class GreedySelector(SelectorBase):
    """Greedy Set Cover sentence selector.

    At each iteration, selects the candidate that covers the most
    previously-uncovered target units. Achieves a ln(n)+1 approximation
    ratio to optimal (Chvátal, 1979), proven tight unless P=NP (Feige, 1998).

    This is the standard workhorse algorithm for corpus selection,
    used in FestVox, CMU ARCTIC, and many speech corpus projects.
    """

    @property
    def algorithm_name(self) -> str:
        return "greedy"

    def select(
        self,
        candidates: list[str],
        candidate_phonemes: list[list[str]],
        target_units: set[str],
        max_sentences: int | None = None,
        target_coverage: float = 1.0,
        weights: dict[str, float] | None = None,
    ) -> SelectionResult:
        start = time.perf_counter()

        # Edge case: empty target → instant full coverage
        if not target_units:
            return SelectionResult(
                selected_indices=[],
                selected_sentences=[],
                coverage=1.0,
                covered_units=set(),
                missing_units=set(),
                unit=self.unit,
                algorithm=self.algorithm_name,
                elapsed_seconds=time.perf_counter() - start,
                iterations=0,
                metadata={},
            )

        # Edge case: no candidates
        if not candidates:
            return SelectionResult(
                selected_indices=[],
                selected_sentences=[],
                coverage=0.0,
                covered_units=set(),
                missing_units=set(target_units),
                unit=self.unit,
                algorithm=self.algorithm_name,
                elapsed_seconds=time.perf_counter() - start,
                iterations=0,
                metadata={},
            )

        # Precompute units per candidate (only those in target)
        candidate_units: list[set[str]] = []
        for phonemes in candidate_phonemes:
            units = self._extract_units(phonemes)
            candidate_units.append(units & target_units)

        covered: set[str] = set()
        selected_indices: list[int] = []
        available = set(range(len(candidates)))
        iterations = 0

        target_count = len(target_units)
        coverage_threshold = target_coverage * target_count

        while available:
            iterations += 1

            # Check stopping conditions
            if len(covered) >= coverage_threshold:
                break
            if max_sentences is not None and len(selected_indices) >= max_sentences:
                break

            # Find candidate with maximum marginal gain
            best_idx = -1
            best_gain = 0.0
            for idx in available:
                new_units = candidate_units[idx] - covered
                gain = self._weighted_gain(new_units, weights)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx

            # No candidate adds anything new → stop
            if best_gain == 0:
                break

            # Select the best candidate
            selected_indices.append(best_idx)
            covered |= candidate_units[best_idx]
            available.discard(best_idx)

        elapsed = time.perf_counter() - start
        coverage = len(covered) / target_count if target_count > 0 else 1.0

        return SelectionResult(
            selected_indices=selected_indices,
            selected_sentences=[candidates[i] for i in selected_indices],
            coverage=coverage,
            covered_units=set(covered),
            missing_units=target_units - covered,
            unit=self.unit,
            algorithm=self.algorithm_name,
            elapsed_seconds=elapsed,
            iterations=iterations,
            metadata={},
        )

