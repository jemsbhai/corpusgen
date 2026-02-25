"""DistributionAwareSelector: frequency-distribution matching selection."""

from __future__ import annotations

import math
import time
from collections import Counter

from corpusgen.select.base import SelectorBase
from corpusgen.select.result import SelectionResult


class DistributionAwareSelector(SelectorBase):
    """Distribution-aware sentence selector.

    Goes beyond binary coverage to select sentences that bring the
    corpus phoneme frequency distribution as close as possible to a
    specified target distribution. Uses KL-divergence as the distance
    metric and greedily picks the candidate that minimises it at each
    step, with a fallback to maximum marginal coverage gain for
    tie-breaking and initial selection.

    This addresses a real limitation of pure Set Cover: a corpus that
    covers every phoneme once but with wildly skewed frequencies is
    suboptimal for TTS/ASR training (Alghamdi et al., 2021).

    Args:
        target_distribution: Mapping from phonetic unit to target
            frequency proportion. Will be normalized to sum to 1.0.
        unit: Coverage unit type.
    """

    def __init__(
        self,
        target_distribution: dict[str, float],
        unit: str = "phoneme",
    ) -> None:
        super().__init__(unit=unit)
        if not target_distribution:
            raise ValueError("target_distribution must be non-empty")
        if any(v <= 0 for v in target_distribution.values()):
            raise ValueError(
                "All target_distribution values must be positive"
            )
        # Normalize
        total = sum(target_distribution.values())
        self._target_dist = {
            k: v / total for k, v in target_distribution.items()
        }

    @property
    def algorithm_name(self) -> str:
        return "distribution"

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
                metadata={"kl_divergence": 0.0},
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
                metadata={"kl_divergence": float("inf")},
            )

        # Precompute units per candidate (all occurrences, not just unique)
        candidate_unit_lists: list[list[str]] = []
        candidate_unit_sets: list[set[str]] = []
        for phonemes in candidate_phonemes:
            units = self._extract_unit_list(phonemes)
            candidate_unit_lists.append(units)
            candidate_unit_sets.append(set(units) & target_units)

        # State
        corpus_counts: Counter[str] = Counter()
        covered: set[str] = set()
        selected_indices: list[int] = []
        available = set(range(len(candidates)))
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

            best_idx = -1
            best_score = float("inf")  # Lower KL is better
            best_gain = 0  # Tie-break by coverage gain

            for idx in available:
                # Compute hypothetical corpus counts if we add this candidate
                hypo_counts = corpus_counts + Counter(
                    u for u in candidate_unit_lists[idx] if u in target_units
                )
                gain = len(candidate_unit_sets[idx] - covered)

                if sum(hypo_counts.values()) == 0:
                    continue

                kl = self._kl_divergence(hypo_counts)

                # Prefer lower KL; break ties with higher weighted coverage gain
                wgain = self._weighted_gain(
                    candidate_unit_sets[idx] - covered, weights
                )
                if (kl < best_score) or (kl == best_score and wgain > best_gain):
                    best_score = kl
                    best_idx = idx
                    best_gain = wgain

            if best_idx == -1:
                break

            # Check if we're making any progress
            gain = len(candidate_unit_sets[best_idx] - covered)
            if gain == 0 and len(covered) >= coverage_threshold:
                break

            selected_indices.append(best_idx)
            corpus_counts += Counter(
                u for u in candidate_unit_lists[best_idx] if u in target_units
            )
            covered |= candidate_unit_sets[best_idx]
            available.discard(best_idx)

            # If no gain and no budget constraint, check if more is useful
            if gain == 0:
                # Still adding for distributional reasons — keep going
                # unless nothing left to improve
                any_remaining_gain = any(
                    len(candidate_unit_sets[i] - covered) > 0
                    for i in available
                )
                if not any_remaining_gain:
                    break

        elapsed = time.perf_counter() - start
        coverage = len(covered) / target_count if target_count > 0 else 1.0
        final_kl = self._kl_divergence(corpus_counts) if corpus_counts else float("inf")

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
            metadata={"kl_divergence": final_kl},
        )

    def _kl_divergence(self, counts: Counter[str]) -> float:
        """Compute KL(target || corpus) with Laplace smoothing.

        Args:
            counts: Observed unit frequency counts in corpus.

        Returns:
            KL divergence from target to observed distribution.
        """
        total = sum(counts.values())
        if total == 0:
            return float("inf")

        # All units in target distribution
        units = set(self._target_dist.keys())

        # Laplace smoothing: add 1 to every count to avoid log(0)
        smoothed_total = total + len(units)
        kl = 0.0
        for u in units:
            p = self._target_dist[u]  # target
            q = (counts.get(u, 0) + 1) / smoothed_total  # observed (smoothed)
            if p > 0:
                kl += p * math.log(p / q)
        return kl

