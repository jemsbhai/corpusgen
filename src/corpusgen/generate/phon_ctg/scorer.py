"""PhoneticScorer: evaluates text against phonetic targets for Phon-CTG.

Scores candidate text based on its contribution to covering remaining
phonetic targets. Supports a composite score combining coverage gain,
phonotactic legality, and fluency — mirroring the Phon-CTG reward function:

    R = w_cov · R_coverage + w_phono · R_phonotactic + w_fluency · R_fluency

Operates in two modes:
    - **Peek** (``score``/``score_batch``/``rank``): non-destructive evaluation
    - **Commit** (``score_and_commit``): scores then updates target inventory
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from corpusgen.generate.phon_ctg.targets import PhoneticTargetInventory


@dataclass
class ScoreResult:
    """Structured result from scoring a candidate sentence.

    Attributes:
        text: Original text (if provided).
        phonemes: Phoneme list used for scoring.
        coverage_gain: Number of new target units covered.
        weighted_coverage_gain: Sum of weights of new target units covered.
        phonotactic_score: Score from the phonotactic constraint hook (0.0 if no hook).
        fluency_score: Score from the fluency hook (0.0 if no hook).
        composite_score: Weighted combination of all score components.
        new_units: Set of target units newly covered by this candidate.
    """

    text: str | None
    phonemes: list[str]
    coverage_gain: int
    weighted_coverage_gain: float
    phonotactic_score: float
    fluency_score: float
    composite_score: float
    new_units: set[str]


class PhoneticScorer:
    """Evaluates candidate text against a dynamic phonetic target inventory.

    Args:
        targets: The PhoneticTargetInventory to score against.
        phonotactic_scorer: Optional callable (phonemes -> float) for
            phonotactic legality scoring.
        fluency_scorer: Optional callable (text -> float) for fluency scoring.
        coverage_weight: Weight for the coverage component in the composite score.
        phonotactic_weight: Weight for the phonotactic component.
        fluency_weight: Weight for the fluency component.
    """

    def __init__(
        self,
        targets: PhoneticTargetInventory,
        phonotactic_scorer: Callable[[list[str]], float] | None = None,
        fluency_scorer: Callable[[str | None], float] | None = None,
        coverage_weight: float = 1.0,
        phonotactic_weight: float = 0.0,
        fluency_weight: float = 0.0,
    ) -> None:
        self._targets = targets
        self._phonotactic_scorer = phonotactic_scorer
        self._fluency_scorer = fluency_scorer
        self._coverage_weight = coverage_weight
        self._phonotactic_weight = phonotactic_weight
        self._fluency_weight = fluency_weight

    # -------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------

    @property
    def targets(self) -> PhoneticTargetInventory:
        """The target inventory being scored against."""
        return self._targets

    @property
    def coverage_weight(self) -> float:
        """Weight for coverage component in composite score."""
        return self._coverage_weight

    @property
    def phonotactic_weight(self) -> float:
        """Weight for phonotactic component in composite score."""
        return self._phonotactic_weight

    @property
    def fluency_weight(self) -> float:
        """Weight for fluency component in composite score."""
        return self._fluency_weight

    # -------------------------------------------------------------------
    # Internal: compute units from phoneme list
    # -------------------------------------------------------------------

    def _extract_units(self, phonemes: list[str]) -> list[str]:
        """Extract coverage units from a phoneme sequence.

        Mirrors the logic in CoverageTracker.update but returns the
        unit list without modifying any state.
        """
        unit = self._targets.unit
        if unit == "phoneme":
            return phonemes
        elif unit == "diphone":
            return [
                f"{phonemes[i]}-{phonemes[i + 1]}"
                for i in range(len(phonemes) - 1)
            ]
        elif unit == "triphone":
            return [
                f"{phonemes[i]}-{phonemes[i + 1]}-{phonemes[i + 2]}"
                for i in range(len(phonemes) - 2)
            ]
        return []

    def _compute_new_units(self, phonemes: list[str]) -> set[str]:
        """Find which target units this phoneme sequence would newly cover."""
        units = self._extract_units(phonemes)
        target_set = self._targets.target_units
        already_covered = self._targets.covered_units
        return {u for u in units if u in target_set and u not in already_covered}

    # -------------------------------------------------------------------
    # Core scoring
    # -------------------------------------------------------------------

    def _score_internal(
        self,
        phonemes: list[str],
        text: str | None = None,
    ) -> ScoreResult:
        """Compute all score components for a candidate.

        Does not modify inventory state.
        """
        new_units = self._compute_new_units(phonemes)
        coverage_gain = len(new_units)

        # Weighted coverage gain using target inventory weights
        weighted_gain = sum(
            self._targets._get_weight(u) for u in new_units
        )

        # Phonotactic score
        if self._phonotactic_scorer is not None:
            phonotactic_score = self._phonotactic_scorer(phonemes)
        else:
            phonotactic_score = 0.0

        # Fluency score
        if self._fluency_scorer is not None:
            fluency_score = self._fluency_scorer(text)
        else:
            fluency_score = 0.0

        # Composite
        composite = (
            self._coverage_weight * weighted_gain
            + self._phonotactic_weight * phonotactic_score
            + self._fluency_weight * fluency_score
        )

        return ScoreResult(
            text=text,
            phonemes=phonemes,
            coverage_gain=coverage_gain,
            weighted_coverage_gain=weighted_gain,
            phonotactic_score=phonotactic_score,
            fluency_score=fluency_score,
            composite_score=composite,
            new_units=new_units,
        )

    # -------------------------------------------------------------------
    # Public API: peek mode
    # -------------------------------------------------------------------

    def score(
        self,
        phonemes: list[str],
        text: str | None = None,
    ) -> ScoreResult:
        """Score a candidate without modifying inventory state.

        Args:
            phonemes: Phoneme list for the candidate sentence.
            text: Optional raw text (passed to fluency hook if provided).

        Returns:
            ScoreResult with all score components.
        """
        return self._score_internal(phonemes=phonemes, text=text)

    def score_batch(
        self,
        candidates: list[dict],
    ) -> list[ScoreResult]:
        """Score multiple candidates without modifying inventory state.

        Each candidate is scored independently against the current state.

        Args:
            candidates: List of dicts, each with "phonemes" (required)
                and optionally "text".

        Returns:
            List of ScoreResult, one per candidate, in input order.
        """
        return [
            self._score_internal(
                phonemes=c["phonemes"],
                text=c.get("text"),
            )
            for c in candidates
        ]

    def rank(
        self,
        candidates: list[dict],
        top_k: int | None = None,
    ) -> list[ScoreResult]:
        """Score and rank candidates by composite score (descending).

        Args:
            candidates: List of dicts, each with "phonemes" (required)
                and optionally "text".
            top_k: If provided, return only the top-k results.

        Returns:
            List of ScoreResult sorted by composite_score descending.
        """
        results = self.score_batch(candidates)
        results.sort(key=lambda r: r.composite_score, reverse=True)
        if top_k is not None:
            return results[:top_k]
        return results

    # -------------------------------------------------------------------
    # Public API: commit mode
    # -------------------------------------------------------------------

    def score_and_commit(
        self,
        phonemes: list[str],
        sentence_index: int,
        text: str | None = None,
    ) -> ScoreResult:
        """Score a candidate then update the inventory with its coverage.

        Args:
            phonemes: Phoneme list for the candidate sentence.
            sentence_index: Index of the sentence in the corpus.
            text: Optional raw text (passed to fluency hook if provided).

        Returns:
            ScoreResult computed before the inventory update.
        """
        result = self._score_internal(phonemes=phonemes, text=text)
        self._targets.update(phonemes, sentence_index)
        return result
