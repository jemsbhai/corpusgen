"""Phon-CTG Generation Loop: orchestrates backend, scorer, and targets.

The generation loop is the heart of Phon-CTG. It:
    1. Queries the PhoneticTargetInventory for uncovered units
    2. Asks a GenerationBackend to produce candidate sentences
    3. Uses the PhoneticScorer to evaluate and rank candidates
    4. Commits the best candidate and repeats until stopping criteria are met

Components:
    - **GenerationBackend**: ABC for pluggable generation engines
    - **StoppingCriteria**: configurable termination conditions
    - **GenerationResult**: structured output from a generation run
    - **GenerationLoop**: the orchestration loop itself
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from corpusgen.generate.phon_ctg.scorer import PhoneticScorer
from corpusgen.generate.phon_ctg.targets import PhoneticTargetInventory


logger = logging.getLogger(__name__)


class GenerationBackend(ABC):
    """Abstract base class for text generation backends.

    All backends must implement:
        - ``name``: identifier string for logging and results
        - ``generate(target_units, k)``: produce candidate sentences

    Each candidate is a dict with at minimum ``"phonemes"`` (list[str]),
    and optionally ``"text"`` (str).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier string."""

    @abstractmethod
    def generate(
        self,
        target_units: list[str],
        k: int = 5,
        **kwargs: Any,
    ) -> list[dict]:
        """Generate candidate sentences targeting specific phonetic units.

        Args:
            target_units: Highest-priority uncovered units to target.
            k: Number of candidates to generate.
            **kwargs: Backend-specific parameters.

        Returns:
            List of candidate dicts, each with at least ``"phonemes"``
            (list[str]) and optionally ``"text"`` (str).
        """


@dataclass
class StoppingCriteria:
    """Configurable stopping conditions for the generation loop.

    The loop terminates when ANY condition is met.

    Attributes:
        target_coverage: Stop when this coverage fraction is reached.
        max_sentences: Maximum number of sentences to accept.
        max_iterations: Maximum loop iterations (backend calls).
        timeout_seconds: Wall-clock time limit in seconds.
    """

    target_coverage: float = 1.0
    max_sentences: int | None = None
    max_iterations: int | None = None
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.target_coverage <= 1.0):
            raise ValueError(
                f"target_coverage must be in [0.0, 1.0], got {self.target_coverage}"
            )


@dataclass(frozen=True)
class GenerationResult:
    """Immutable result of a generation loop run.

    Attributes:
        generated_sentences: Accepted sentences in generation order.
        generated_phonemes: Phoneme lists for each accepted sentence.
        coverage: Final coverage ratio (0.0-1.0).
        covered_units: Set of target units covered.
        missing_units: Set of target units still uncovered.
        unit: Coverage unit type.
        backend: Name of the backend used.
        elapsed_seconds: Wall-clock time in seconds.
        iterations: Number of loop iterations completed.
        stop_reason: Why the loop terminated.
        metadata: Additional loop-specific data.
    """

    generated_sentences: list[str]
    generated_phonemes: list[list[str]]
    coverage: float
    covered_units: set[str]
    missing_units: set[str]
    unit: str
    backend: str
    elapsed_seconds: float
    iterations: int
    stop_reason: str
    metadata: dict = field(default_factory=dict)

    @property
    def num_generated(self) -> int:
        """Number of sentences generated."""
        return len(self.generated_sentences)


class GenerationLoop:
    """Orchestrates the Phon-CTG generation process.

    Connects a GenerationBackend, PhoneticTargetInventory, and
    PhoneticScorer into an iterative loop that generates sentences
    to maximize phonetic coverage.

    Args:
        backend: The generation backend to use.
        targets: The phonetic target inventory to cover.
        scorer: The scorer for evaluating candidates.
        stopping_criteria: When to stop generating. Defaults to
            full coverage with no other limits.
        candidates_per_iteration: How many candidates to request
            from the backend each iteration.
        on_progress: Optional callback invoked after each accepted
            sentence. Receives a dict with iteration info.
    """

    def __init__(
        self,
        backend: GenerationBackend,
        targets: PhoneticTargetInventory,
        scorer: PhoneticScorer,
        stopping_criteria: StoppingCriteria | None = None,
        candidates_per_iteration: int = 5,
        on_progress: Callable[[dict], None] | None = None,
    ) -> None:
        self._backend = backend
        self._targets = targets
        self._scorer = scorer
        self._stopping = stopping_criteria or StoppingCriteria()
        self._candidates_per_iteration = candidates_per_iteration
        self._on_progress = on_progress

    @property
    def backend(self) -> GenerationBackend:
        """The generation backend."""
        return self._backend

    @property
    def targets(self) -> PhoneticTargetInventory:
        """The phonetic target inventory."""
        return self._targets

    @property
    def scorer(self) -> PhoneticScorer:
        """The phonetic scorer."""
        return self._scorer

    @property
    def stopping_criteria(self) -> StoppingCriteria:
        """The stopping criteria."""
        return self._stopping

    @property
    def candidates_per_iteration(self) -> int:
        """Number of candidates requested per iteration."""
        return self._candidates_per_iteration

    def _check_stop(
        self,
        iterations: int,
        num_sentences: int,
        start_time: float,
    ) -> str | None:
        """Check if any stopping condition is met.

        Returns the stop reason string, or None to continue.
        """
        if self._targets.coverage >= self._stopping.target_coverage:
            return "target_coverage"

        if (
            self._stopping.max_sentences is not None
            and num_sentences >= self._stopping.max_sentences
        ):
            return "max_sentences"

        if (
            self._stopping.max_iterations is not None
            and iterations >= self._stopping.max_iterations
        ):
            return "max_iterations"

        if self._stopping.timeout_seconds is not None:
            elapsed = time.monotonic() - start_time
            if elapsed >= self._stopping.timeout_seconds:
                return "timeout"

        return None

    def run(self) -> GenerationResult:
        """Execute the generation loop.

        Returns:
            GenerationResult with all generated sentences and metrics.
        """
        start_time = time.monotonic()
        iterations = 0
        generated_sentences: list[str] = []
        generated_phonemes: list[list[str]] = []
        stop_reason = "target_coverage"

        # Check if already at target (e.g., empty inventory)
        initial_stop = self._check_stop(iterations, 0, start_time)
        if initial_stop is not None:
            elapsed = time.monotonic() - start_time
            logger.info(
                "Generation complete before first iteration: %s "
                "(coverage=%.2f%%)",
                initial_stop,
                self._targets.coverage * 100,
            )
            return self._build_result(
                generated_sentences,
                generated_phonemes,
                elapsed,
                iterations,
                initial_stop,
            )

        while True:
            iterations += 1

            # 1. Get highest-priority uncovered targets
            next_targets = self._targets.next_targets(
                k=self._candidates_per_iteration
            )

            # 2. Ask backend for candidates
            candidates = self._backend.generate(
                target_units=next_targets,
                k=self._candidates_per_iteration,
            )

            # 3. Handle empty response
            if not candidates:
                stop_reason = "backend_exhausted"
                logger.info(
                    "Backend exhausted at iteration %d (coverage=%.2f%%)",
                    iterations,
                    self._targets.coverage * 100,
                )
                break

            # 4. Rank candidates by composite score
            ranked = self._scorer.rank(candidates)

            # 5. Find best candidate with positive gain
            best = None
            for candidate in ranked:
                if candidate.coverage_gain > 0:
                    best = candidate
                    break

            # 6. No candidate adds coverage
            if best is None:
                stop_reason = "backend_exhausted"
                logger.info(
                    "No candidates with positive gain at iteration %d "
                    "(coverage=%.2f%%)",
                    iterations,
                    self._targets.coverage * 100,
                )
                break

            # 7. Commit the best candidate
            sentence_index = len(generated_sentences)
            self._scorer.score_and_commit(
                phonemes=best.phonemes,
                sentence_index=sentence_index,
                text=best.text,
            )

            text = best.text or ""
            generated_sentences.append(text)
            generated_phonemes.append(best.phonemes)

            logger.info(
                "Iteration %d: accepted (gain=%d, coverage=%.2f%%)",
                iterations,
                best.coverage_gain,
                self._targets.coverage * 100,
            )

            # 8. Callback
            if self._on_progress is not None:
                self._on_progress({
                    "iteration": iterations,
                    "coverage": self._targets.coverage,
                    "sentence": text,
                    "coverage_gain": best.coverage_gain,
                    "covered_count": self._targets.covered_count,
                    "target_size": self._targets.target_size,
                })

            # 9. Check stopping conditions
            stop = self._check_stop(
                iterations, len(generated_sentences), start_time
            )
            if stop is not None:
                stop_reason = stop
                logger.info(
                    "Stopping: %s at iteration %d (coverage=%.2f%%)",
                    stop_reason,
                    iterations,
                    self._targets.coverage * 100,
                )
                break

        elapsed = time.monotonic() - start_time
        return self._build_result(
            generated_sentences,
            generated_phonemes,
            elapsed,
            iterations,
            stop_reason,
        )

    def _build_result(
        self,
        generated_sentences: list[str],
        generated_phonemes: list[list[str]],
        elapsed: float,
        iterations: int,
        stop_reason: str,
    ) -> GenerationResult:
        """Construct the final GenerationResult."""
        return GenerationResult(
            generated_sentences=generated_sentences,
            generated_phonemes=generated_phonemes,
            coverage=self._targets.coverage,
            covered_units=self._targets.covered_units,
            missing_units=self._targets.missing,
            unit=self._targets.unit,
            backend=self._backend.name,
            elapsed_seconds=elapsed,
            iterations=iterations,
            stop_reason=stop_reason,
        )
