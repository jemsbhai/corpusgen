"""PhoneticTargetInventory: dynamic phonetic target tracking for Phon-CTG.

Maintains a prioritized inventory of phonetic units (phonemes, diphones,
or triphones) that the generation framework aims to cover. Tracks what
has been covered, what remains, and which targets are highest priority.

Can operate in two modes:
    - **Standalone**: provide target_phonemes and unit; creates its own
      internal CoverageTracker.
    - **Wrap**: provide an existing CoverageTracker; takes ownership and
      builds priority logic on top.
"""

from __future__ import annotations

from corpusgen.coverage.tracker import CoverageTracker


class PhoneticTargetInventory:
    """Dynamic phonetic target inventory with weighted prioritization.

    Wraps a CoverageTracker and adds priority-based target selection,
    enabling generation backends to query which phonetic units to pursue
    next.

    Args:
        target_phonemes: List of phonemes for the target inventory.
            Mutually exclusive with ``tracker``.
        unit: Coverage unit type — "phoneme", "diphone", or "triphone".
            Ignored when ``tracker`` is provided (uses tracker's unit).
        tracker: An existing CoverageTracker to wrap. Mutually exclusive
            with ``target_phonemes``.
        weights: Optional mapping from unit string to priority weight.
            Higher weight = higher priority. Units not in the dict
            default to 1.0.
        max_target_size: Forwarded to CoverageTracker in standalone mode.
    """

    def __init__(
        self,
        target_phonemes: list[str] | None = None,
        unit: str = "phoneme",
        tracker: CoverageTracker | None = None,
        weights: dict[str, float] | None = None,
        max_target_size: int | None = None,
    ) -> None:
        # --- Validate mutually exclusive args ---
        if tracker is not None and target_phonemes is not None:
            raise ValueError(
                "Cannot provide both 'tracker' and 'target_phonemes'. "
                "Use one or the other."
            )
        if tracker is None and target_phonemes is None:
            raise ValueError(
                "Must provide either 'tracker' or 'target_phonemes'."
            )

        # --- Initialize tracker ---
        if tracker is not None:
            self._tracker = tracker
        else:
            kwargs: dict = {
                "target_phonemes": target_phonemes,
                "unit": unit,
            }
            if max_target_size is not None:
                kwargs["max_target_size"] = max_target_size
            self._tracker = CoverageTracker(**kwargs)

        # --- Store weights (default 1.0 for unspecified units) ---
        self._weights: dict[str, float] = dict(weights) if weights else {}

    # -------------------------------------------------------------------
    # Properties delegated to CoverageTracker
    # -------------------------------------------------------------------

    @property
    def tracker(self) -> CoverageTracker:
        """The underlying CoverageTracker instance."""
        return self._tracker

    @property
    def unit(self) -> str:
        """Coverage unit type."""
        return self._tracker.unit

    @property
    def target_size(self) -> int:
        """Number of units in the target inventory."""
        return self._tracker.target_size

    @property
    def target_units(self) -> set[str]:
        """Full set of target units."""
        return self._tracker.target_units

    @property
    def covered_count(self) -> int:
        """Number of target units covered so far."""
        return self._tracker.covered_count

    @property
    def covered_units(self) -> set[str]:
        """Set of target units covered so far."""
        return self._tracker.covered_units

    @property
    def coverage(self) -> float:
        """Fraction of target units covered (0.0 to 1.0)."""
        return self._tracker.coverage

    @property
    def missing(self) -> set[str]:
        """Set of target units not yet covered."""
        return self._tracker.missing

    # -------------------------------------------------------------------
    # Priority-based target selection
    # -------------------------------------------------------------------

    def _get_weight(self, unit: str) -> float:
        """Get the priority weight for a unit, defaulting to 1.0."""
        return self._weights.get(unit, 1.0)

    def next_targets(self, k: int) -> list[str]:
        """Return the top-k highest-priority uncovered units.

        Units are sorted by descending weight. Ties are broken by
        lexicographic order for determinism.

        Args:
            k: Maximum number of targets to return.

        Returns:
            List of unit strings, ordered by priority (highest first).
            May be shorter than k if fewer uncovered units remain.
        """
        if k <= 0:
            return []

        remaining = self.missing
        if not remaining:
            return []

        # Sort by weight descending, then alphabetically for stability
        ranked = sorted(
            remaining,
            key=lambda u: (-self._get_weight(u), u),
        )

        return ranked[:k]

    # -------------------------------------------------------------------
    # Update and reset
    # -------------------------------------------------------------------

    def update(self, phonemes: list[str], sentence_index: int) -> None:
        """Update coverage with phonemes from a generated sentence.

        Delegates to the underlying CoverageTracker.

        Args:
            phonemes: List of phonemes extracted from the sentence.
            sentence_index: Index of the sentence in the corpus.
        """
        self._tracker.update(phonemes, sentence_index)

    def reset(self) -> None:
        """Reset coverage state, preserving targets and weights."""
        self._tracker.reset()
