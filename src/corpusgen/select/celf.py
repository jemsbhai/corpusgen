"""CELFSelector: Cost-Effective Lazy Forward selection algorithm."""

from __future__ import annotations

import heapq
import time

from corpusgen.select.base import SelectorBase
from corpusgen.select.result import SelectionResult


class CELFSelector(SelectorBase):
    """Cost-Effective Lazy Forward (CELF) sentence selector.

    Exploits the submodularity of coverage functions: marginal gains
    can only decrease as more candidates are selected. Instead of
    re-evaluating all candidates at each iteration, CELF uses a
    max-heap with lazy re-evaluation, achieving up to 700x speedup
    over naive greedy while producing identical results.

    Reference:
        Leskovec, J., Krause, A., Guestrin, C., Faloutsos, C.,
        VanBriesen, J., & Glance, N. (2007). Cost-effective outbreak
        detection in networks. KDD.
    """

    @property
    def algorithm_name(self) -> str:
        return "celf"

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

        # Edge case: empty target
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
                metadata={"evaluations": 0},
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
                metadata={"evaluations": 0},
            )

        # Precompute units per candidate (only those in target)
        candidate_units: list[set[str]] = []
        for phonemes in candidate_phonemes:
            units = self._extract_units(phonemes)
            candidate_units.append(units & target_units)

        covered: set[str] = set()
        selected_indices: list[int] = []
        target_count = len(target_units)
        coverage_threshold = target_coverage * target_count
        evaluations = 0
        iterations = 0

        # Initialize max-heap: gains are correct for round 1 (covered is empty)
        # Entries: (-gain, idx, eval_round)
        # Using heapq (min-heap), so negate gains for max behaviour.
        current_round = 1
        heap: list[tuple[float, int, int]] = []
        for idx in range(len(candidates)):
            gain = self._weighted_gain(candidate_units[idx], weights)
            evaluations += 1
            heapq.heappush(heap, (-gain, idx, current_round))

        while heap:
            # Check stopping conditions
            if len(covered) >= coverage_threshold:
                break
            if max_sentences is not None and len(selected_indices) >= max_sentences:
                break

            # Find the true best candidate for this round via lazy evaluation
            found = False
            while heap:
                neg_gain, idx, eval_round = heapq.heappop(heap)

                if eval_round == current_round:
                    # Gain is fresh for this round
                    if -neg_gain <= 0:
                        # No candidate adds anything — terminate
                        found = False
                        break
                    # Select this candidate
                    selected_indices.append(idx)
                    covered |= candidate_units[idx]
                    current_round += 1
                    iterations += 1
                    found = True
                    break
                else:
                    # Stale — recompute marginal gain
                    new_units = candidate_units[idx] - covered
                    new_gain = self._weighted_gain(new_units, weights)
                    evaluations += 1
                    heapq.heappush(heap, (-new_gain, idx, current_round))

            if not found:
                break

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
            metadata={"evaluations": evaluations},
        )
