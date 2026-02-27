"""Coverage trajectory tracking for corpus selection and generation.

Computes step-by-step coverage snapshots from an ordered sequence of
phoneme lists against a target inventory.  This produces the classic
"coverage saturation curve" used in every corpus selection paper.

The computation is purely post-hoc — it works on any ordered phoneme
sequence from any source (selection results, generation results,
or arbitrary corpus orderings) without modifying the algorithms.

Usage::

    from corpusgen.evaluate.trajectory import compute_coverage_trajectory

    # From a SelectionResult:
    traj = compute_coverage_trajectory(
        [candidate_phonemes[i] for i in result.selected_indices],
        target_units=result.covered_units | result.missing_units,
        unit=result.unit,
    )

    # From a GenerationResult:
    traj = compute_coverage_trajectory(
        result.generated_phonemes,
        target_units=result.covered_units | result.missing_units,
        unit=result.unit,
    )

    # Plot the curve:
    import matplotlib.pyplot as plt
    plt.plot(range(len(traj.coverages)), traj.coverages)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class CoverageSnapshot:
    """A single point on the coverage trajectory.

    Represents the cumulative coverage state after adding one sentence.

    Attributes:
        sentence_index: Position of this sentence in the input sequence.
        coverage: Cumulative coverage fraction (0.0–1.0) after this
            sentence.
        covered_count: Cumulative number of target units covered.
        new_units_count: Number of new target units covered by this
            sentence (marginal gain).
        new_units: The specific new target units covered by this
            sentence, deduplicated and in first-occurrence order.
    """

    sentence_index: int
    coverage: float
    covered_count: int
    new_units_count: int
    new_units: list[str]


@dataclass(frozen=True)
class CoverageTrajectory:
    """Complete coverage trajectory over an ordered sentence sequence.

    Attributes:
        snapshots: One snapshot per sentence, in input order.
        unit: Coverage unit type ("phoneme", "diphone", "triphone").
        target_size: Total number of target units.
    """

    snapshots: list[CoverageSnapshot]
    unit: str
    target_size: int

    @property
    def coverages(self) -> list[float]:
        """Coverage fractions for easy plotting — one value per sentence."""
        return [s.coverage for s in self.snapshots]

    @property
    def gains(self) -> list[int]:
        """Marginal gains (new unit counts) per sentence."""
        return [s.new_units_count for s in self.snapshots]

    def to_dict(self) -> dict[str, Any]:
        """Export as a plain Python dict (JSON-safe)."""
        return {
            "unit": self.unit,
            "target_size": self.target_size,
            "snapshots": [asdict(s) for s in self.snapshots],
        }


def _extract_units(phonemes: list[str], unit: str) -> list[str]:
    """Extract coverage units from a phoneme sequence.

    Returns a list (with duplicates) so the caller can deduplicate
    while preserving first-occurrence order.

    Args:
        phonemes: Phoneme list for a single sentence.
        unit: One of "phoneme", "diphone", "triphone".

    Returns:
        List of unit strings.
    """
    if unit == "phoneme":
        return list(phonemes)
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
    else:
        raise ValueError(f"Invalid unit: {unit!r}")


def compute_coverage_trajectory(
    phoneme_sequences: list[list[str]],
    target_units: set[str],
    unit: str = "phoneme",
) -> CoverageTrajectory:
    """Compute a step-by-step coverage trajectory.

    For each sentence in order, records which new target units it
    covers and the cumulative coverage fraction.  This produces the
    data needed for the classic coverage saturation curve.

    Args:
        phoneme_sequences: Ordered list of phoneme lists.  Each inner
            list is the phoneme sequence for one sentence, in the order
            sentences were selected or generated.
        target_units: The full set of target units to measure against.
        unit: Coverage unit type — "phoneme", "diphone", or "triphone".
            Must match the format of ``target_units``.

    Returns:
        CoverageTrajectory with one snapshot per sentence.
    """
    target_size = len(target_units)
    covered: set[str] = set()
    snapshots: list[CoverageSnapshot] = []

    for idx, phonemes in enumerate(phoneme_sequences):
        sentence_units = _extract_units(phonemes, unit)

        # Find new target units from this sentence (deduplicated, ordered)
        new_units: list[str] = []
        seen_in_sentence: set[str] = set()
        for u in sentence_units:
            if u in target_units and u not in covered and u not in seen_in_sentence:
                new_units.append(u)
                seen_in_sentence.add(u)

        covered |= seen_in_sentence
        coverage = len(covered) / target_size if target_size > 0 else 1.0

        snapshots.append(
            CoverageSnapshot(
                sentence_index=idx,
                coverage=coverage,
                covered_count=len(covered),
                new_units_count=len(new_units),
                new_units=new_units,
            )
        )

    return CoverageTrajectory(
        snapshots=snapshots,
        unit=unit,
        target_size=target_size,
    )
