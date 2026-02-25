"""SelectionResult: immutable output from any sentence selection algorithm."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SelectionResult:
    """Immutable result of a sentence selection algorithm run.

    Attributes:
        selected_indices: Indices into the original candidate list.
        selected_sentences: The selected sentences (same order as indices).
        coverage: Final coverage ratio (0.0–1.0) of target units.
        covered_units: Set of target units that were covered.
        missing_units: Set of target units that remain uncovered.
        unit: Coverage unit type ("phoneme", "diphone", "triphone").
        algorithm: Name of the algorithm that produced this result.
        elapsed_seconds: Wall-clock time in seconds.
        iterations: Number of algorithm iterations/steps taken.
        metadata: Algorithm-specific extras (e.g. solver_status for ILP,
            pareto_front for NSGA-II, sample_size for Stochastic Greedy).
    """

    selected_indices: list[int]
    selected_sentences: list[str]
    coverage: float
    covered_units: set[str]
    missing_units: set[str]
    unit: str
    algorithm: str
    elapsed_seconds: float
    iterations: int
    metadata: dict = field(default_factory=dict)

    @property
    def num_selected(self) -> int:
        """Number of sentences selected."""
        return len(self.selected_indices)
