"""StochasticGreedySelector: randomized subsampling for scalable selection."""

from __future__ import annotations

import math
import random
import time

from corpusgen.select.base import SelectorBase
from corpusgen.select.result import SelectionResult


class StochasticGreedySelector(SelectorBase):
    """Stochastic Greedy sentence selector for large-scale corpora.

    Instead of scanning all remaining candidates at each iteration,
    samples a random subset of size (n/k)·log(1/ε) and picks the
    best within that sample. Achieves (1-1/e-ε) approximation in
    O(n·log(1/ε)) time instead of O(n·k) for standard greedy.

    Reference:
        Mirzasoleiman, B., Badanidiyuru, A., Karbasi, A., Vondrák, J.,
        & Krause, A. (2015). Lazier than lazy greedy. AAAI/NeurIPS.

    Args:
        unit: Coverage unit type.
        epsilon: Approximation parameter in (0, 1]. Smaller ε → larger
            samples → closer to greedy quality but slower.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        unit: str = "phoneme",
        epsilon: float = 0.1,
        seed: int | None = None,
    ) -> None:
        super().__init__(unit=unit)
        if epsilon <= 0 or epsilon > 1:
            raise ValueError(
                f"epsilon must be in (0, 1], got {epsilon}"
            )
        self._epsilon = epsilon
        self._seed = seed

    @property
    def algorithm_name(self) -> str:
        return "stochastic"

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
        rng = random.Random(self._seed)

        # Edge case: empty target
        if not target_units:
            return self._make_result(
                [], candidates, target_units, set(), 1.0, 0,
                time.perf_counter() - start,
            )

        # Edge case: no candidates
        if not candidates:
            return self._make_result(
                [], candidates, target_units, set(), 0.0, 0,
                time.perf_counter() - start,
            )

        # Precompute units per candidate (only those in target)
        candidate_units: list[set[str]] = []
        for phonemes in candidate_phonemes:
            units = self._extract_units(phonemes)
            candidate_units.append(units & target_units)

        n = len(candidates)
        # Budget: either explicit or number of candidates
        k = max_sentences if max_sentences is not None else n

        # Sample size per iteration: (n/k) * log(1/ε)
        sample_size = max(1, int(math.ceil((n / k) * math.log(1.0 / self._epsilon))))

        covered: set[str] = set()
        selected_indices: list[int] = []
        available = list(range(n))
        target_count = len(target_units)
        coverage_threshold = target_coverage * target_count
        iterations = 0

        while available:
            # Check stopping conditions
            if len(covered) >= coverage_threshold:
                break
            if max_sentences is not None and len(selected_indices) >= max_sentences:
                break

            iterations += 1

            # Sample a random subset of available candidates
            actual_sample_size = min(sample_size, len(available))
            sample = rng.sample(available, actual_sample_size)

            # Find best in sample by marginal gain
            best_idx = -1
            best_gain = 0.0
            for idx in sample:
                new_units = candidate_units[idx] - covered
                gain = self._weighted_gain(new_units, weights)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx

            # No candidate in sample adds anything
            if best_gain == 0:
                # Check if ANY remaining candidate has gain > 0
                any_gain = False
                for idx in available:
                    if len(candidate_units[idx] - covered) > 0:
                        any_gain = True
                        break
                if not any_gain:
                    break
                # Otherwise continue sampling
                continue

            selected_indices.append(best_idx)
            covered |= candidate_units[best_idx]
            available.remove(best_idx)

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
            metadata={
                "epsilon": self._epsilon,
                "seed": self._seed,
                "sample_size": sample_size,
            },
        )

    def _make_result(
        self,
        selected_indices: list[int],
        candidates: list[str],
        target_units: set[str],
        covered: set[str],
        coverage: float,
        iterations: int,
        elapsed: float,
    ) -> SelectionResult:
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
            metadata={
                "epsilon": self._epsilon,
                "seed": self._seed,
                "sample_size": 0,
            },
        )
