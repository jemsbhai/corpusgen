"""Tests for Phon-CTG Generation Loop — orchestrates backend + scorer + targets."""

import time
import logging

import pytest

from corpusgen.generate.phon_ctg.targets import PhoneticTargetInventory
from corpusgen.generate.phon_ctg.scorer import PhoneticScorer
from corpusgen.generate.phon_ctg.loop import (
    GenerationBackend,
    GenerationResult,
    GenerationLoop,
    StoppingCriteria,
)


# ---------------------------------------------------------------------------
# Fake backend for testing
# ---------------------------------------------------------------------------


class FakeBackend(GenerationBackend):
    """A test backend that returns pre-configured sentences.

    Each call to generate() pops from a queue of prepared responses.
    If the queue is empty, returns empty list.
    """

    def __init__(self, responses: list[list[dict]] | None = None):
        self._responses = list(responses) if responses else []
        self._call_count = 0
        self._received_targets: list[list[str]] = []

    @property
    def name(self) -> str:
        return "fake"

    def generate(
        self, target_units: list[str], k: int = 5, **kwargs
    ) -> list[dict]:
        self._call_count += 1
        self._received_targets.append(list(target_units))
        if self._responses:
            return self._responses.pop(0)
        return []


class InfiniteBackend(GenerationBackend):
    """Backend that generates candidates covering one target at a time.

    Each call returns a candidate containing the first requested target unit,
    ensuring the loop always has positive gain and doesn't exit early.
    """

    def __init__(self):
        self._call_count = 0

    @property
    def name(self) -> str:
        return "infinite"

    def generate(
        self, target_units: list[str], k: int = 5, **kwargs
    ) -> list[dict]:
        self._call_count += 1
        if not target_units:
            return []
        # Return a candidate that covers the first target unit
        unit = target_units[0]
        # For phoneme units, the phoneme list is just the unit itself
        # For diphone/triphone, split on '-'
        phonemes = unit.split("-")
        return [{"text": f"sentence_{self._call_count}", "phonemes": phonemes}]


# ---------------------------------------------------------------------------
# GenerationBackend ABC
# ---------------------------------------------------------------------------


class TestGenerationBackendABC:
    """GenerationBackend defines the interface for all backends."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            GenerationBackend()

    def test_must_implement_generate(self):
        class Incomplete(GenerationBackend):
            @property
            def name(self):
                return "incomplete"

        with pytest.raises(TypeError):
            Incomplete()

    def test_must_implement_name(self):
        class Incomplete(GenerationBackend):
            def generate(self, target_units, k=5, **kwargs):
                return []

        with pytest.raises(TypeError):
            Incomplete()

    def test_valid_subclass(self):
        backend = FakeBackend()
        assert isinstance(backend, GenerationBackend)
        assert backend.name == "fake"


# ---------------------------------------------------------------------------
# StoppingCriteria
# ---------------------------------------------------------------------------


class TestStoppingCriteria:
    """StoppingCriteria configuration for the generation loop."""

    def test_default_values(self):
        criteria = StoppingCriteria()
        assert criteria.target_coverage == 1.0
        assert criteria.max_sentences is None
        assert criteria.max_iterations is None
        assert criteria.timeout_seconds is None

    def test_custom_values(self):
        criteria = StoppingCriteria(
            target_coverage=0.95,
            max_sentences=100,
            max_iterations=50,
            timeout_seconds=60.0,
        )
        assert criteria.target_coverage == 0.95
        assert criteria.max_sentences == 100
        assert criteria.max_iterations == 50
        assert criteria.timeout_seconds == 60.0

    def test_invalid_coverage_too_high(self):
        with pytest.raises(ValueError):
            StoppingCriteria(target_coverage=1.5)

    def test_invalid_coverage_negative(self):
        with pytest.raises(ValueError):
            StoppingCriteria(target_coverage=-0.1)


# ---------------------------------------------------------------------------
# GenerationResult
# ---------------------------------------------------------------------------


class TestGenerationResult:
    """GenerationResult is a structured output from the loop."""

    def test_has_required_fields(self):
        result = GenerationResult(
            generated_sentences=["hello", "world"],
            generated_phonemes=[["h", "ɛ", "l", "oʊ"], ["w", "ɜː", "l", "d"]],
            coverage=0.8,
            covered_units={"h", "ɛ"},
            missing_units={"p"},
            unit="phoneme",
            backend="fake",
            elapsed_seconds=1.5,
            iterations=3,
            stop_reason="target_coverage",
            metadata={},
        )
        assert result.num_generated == 2
        assert result.coverage == 0.8
        assert result.stop_reason == "target_coverage"

    def test_stop_reasons(self):
        """Valid stop reasons for documentation."""
        valid_reasons = [
            "target_coverage",
            "max_sentences",
            "max_iterations",
            "timeout",
            "backend_exhausted",
        ]
        for reason in valid_reasons:
            result = GenerationResult(
                generated_sentences=[],
                generated_phonemes=[],
                coverage=0.0,
                covered_units=set(),
                missing_units=set(),
                unit="phoneme",
                backend="fake",
                elapsed_seconds=0.0,
                iterations=0,
                stop_reason=reason,
                metadata={},
            )
            assert result.stop_reason == reason


# ---------------------------------------------------------------------------
# GenerationLoop: construction
# ---------------------------------------------------------------------------


class TestLoopConstruction:
    """GenerationLoop setup."""

    def test_basic_creation(self):
        targets = PhoneticTargetInventory(
            target_phonemes=["p", "b", "t"],
            unit="phoneme",
        )
        scorer = PhoneticScorer(targets=targets)
        backend = FakeBackend()
        loop = GenerationLoop(
            backend=backend,
            targets=targets,
            scorer=scorer,
        )
        assert loop.backend is backend
        assert loop.targets is targets
        assert loop.scorer is scorer

    def test_default_stopping_criteria(self):
        targets = PhoneticTargetInventory(
            target_phonemes=["p", "b"],
            unit="phoneme",
        )
        scorer = PhoneticScorer(targets=targets)
        backend = FakeBackend()
        loop = GenerationLoop(
            backend=backend,
            targets=targets,
            scorer=scorer,
        )
        assert loop.stopping_criteria.target_coverage == 1.0

    def test_custom_stopping_criteria(self):
        targets = PhoneticTargetInventory(
            target_phonemes=["p", "b"],
            unit="phoneme",
        )
        scorer = PhoneticScorer(targets=targets)
        backend = FakeBackend()
        criteria = StoppingCriteria(target_coverage=0.9, max_iterations=10)
        loop = GenerationLoop(
            backend=backend,
            targets=targets,
            scorer=scorer,
            stopping_criteria=criteria,
        )
        assert loop.stopping_criteria.target_coverage == 0.9
        assert loop.stopping_criteria.max_iterations == 10


# ---------------------------------------------------------------------------
# GenerationLoop: run — basic flow
# ---------------------------------------------------------------------------


class TestLoopRun:
    """Core generation loop behavior."""

    def test_achieves_full_coverage(self):
        targets = PhoneticTargetInventory(
            target_phonemes=["p", "b", "t"],
            unit="phoneme",
        )
        scorer = PhoneticScorer(targets=targets)
        backend = FakeBackend(responses=[
            # Iteration 1: two candidates, first covers p and b
            [
                {"text": "pab", "phonemes": ["p", "b"]},
                {"text": "tab", "phonemes": ["t", "b"]},
            ],
            # Iteration 2: one candidate covers t
            [
                {"text": "tat", "phonemes": ["t", "d"]},
            ],
        ])
        loop = GenerationLoop(
            backend=backend,
            targets=targets,
            scorer=scorer,
        )
        result = loop.run()
        assert isinstance(result, GenerationResult)
        assert result.coverage == pytest.approx(1.0)
        assert result.stop_reason == "target_coverage"
        assert result.num_generated == 2

    def test_returns_generated_sentences(self):
        targets = PhoneticTargetInventory(
            target_phonemes=["p", "b"],
            unit="phoneme",
        )
        scorer = PhoneticScorer(targets=targets)
        backend = FakeBackend(responses=[
            [{"text": "pab", "phonemes": ["p", "b"]}],
        ])
        loop = GenerationLoop(
            backend=backend,
            targets=targets,
            scorer=scorer,
        )
        result = loop.run()
        assert "pab" in result.generated_sentences

    def test_returns_generated_phonemes(self):
        targets = PhoneticTargetInventory(
            target_phonemes=["p", "b"],
            unit="phoneme",
        )
        scorer = PhoneticScorer(targets=targets)
        backend = FakeBackend(responses=[
            [{"text": "pab", "phonemes": ["p", "b"]}],
        ])
        loop = GenerationLoop(
            backend=backend,
            targets=targets,
            scorer=scorer,
        )
        result = loop.run()
        assert ["p", "b"] in result.generated_phonemes

    def test_selects_best_candidate_per_iteration(self):
        targets = PhoneticTargetInventory(
            target_phonemes=["p", "b", "t", "d", "k"],
            unit="phoneme",
        )
        scorer = PhoneticScorer(targets=targets)
        backend = FakeBackend(responses=[
            [
                {"text": "one", "phonemes": ["p"]},            # gain 1
                {"text": "best", "phonemes": ["p", "b", "t"]}, # gain 3 — best
                {"text": "two", "phonemes": ["p", "b"]},       # gain 2
            ],
            [
                {"text": "rest", "phonemes": ["d", "k"]},
            ],
        ])
        loop = GenerationLoop(
            backend=backend,
            targets=targets,
            scorer=scorer,
        )
        result = loop.run()
        # Should have picked "best" first, then "rest"
        assert result.generated_sentences[0] == "best"

    def test_passes_target_units_to_backend(self):
        targets = PhoneticTargetInventory(
            target_phonemes=["p", "b", "t"],
            unit="phoneme",
        )
        scorer = PhoneticScorer(targets=targets)
        backend = FakeBackend(responses=[
            [{"text": "all", "phonemes": ["p", "b", "t"]}],
        ])
        loop = GenerationLoop(
            backend=backend,
            targets=targets,
            scorer=scorer,
        )
        loop.run()
        # Backend should have received target units on first call
        assert len(backend._received_targets) >= 1
        first_targets = backend._received_targets[0]
        assert set(first_targets) == {"p", "b", "t"}


# ---------------------------------------------------------------------------
# GenerationLoop: stopping criteria
# ---------------------------------------------------------------------------


class TestLoopStopping:
    """Various stopping conditions."""

    def test_stops_at_target_coverage(self):
        targets = PhoneticTargetInventory(
            target_phonemes=["p", "b", "t", "d"],
            unit="phoneme",
        )
        scorer = PhoneticScorer(targets=targets)
        backend = FakeBackend(responses=[
            [{"text": "s1", "phonemes": ["p", "b"]}],       # 50%
            [{"text": "s2", "phonemes": ["t"]}],             # 75%
            [{"text": "s3", "phonemes": ["d"]}],             # 100% — should not reach
        ])
        criteria = StoppingCriteria(target_coverage=0.75)
        loop = GenerationLoop(
            backend=backend,
            targets=targets,
            scorer=scorer,
            stopping_criteria=criteria,
        )
        result = loop.run()
        assert result.coverage >= 0.75
        assert result.stop_reason == "target_coverage"

    def test_stops_at_max_sentences(self):
        targets = PhoneticTargetInventory(
            target_phonemes=["p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z"],
            unit="phoneme",
        )
        scorer = PhoneticScorer(targets=targets)
        backend = InfiniteBackend()
        criteria = StoppingCriteria(max_sentences=3)
        loop = GenerationLoop(
            backend=backend,
            targets=targets,
            scorer=scorer,
            stopping_criteria=criteria,
        )
        result = loop.run()
        assert result.num_generated <= 3
        assert result.stop_reason == "max_sentences"

    def test_stops_at_max_iterations(self):
        targets = PhoneticTargetInventory(
            target_phonemes=["p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z"],
            unit="phoneme",
        )
        scorer = PhoneticScorer(targets=targets)
        backend = InfiniteBackend()
        criteria = StoppingCriteria(max_iterations=5)
        loop = GenerationLoop(
            backend=backend,
            targets=targets,
            scorer=scorer,
            stopping_criteria=criteria,
        )
        result = loop.run()
        assert result.iterations <= 5
        assert result.stop_reason == "max_iterations"

    def test_stops_at_timeout(self):
        # Large inventory so coverage target is never hit
        phonemes = [f"ph{i}" for i in range(1000)]
        targets = PhoneticTargetInventory(
            target_phonemes=phonemes,
            unit="phoneme",
        )
        scorer = PhoneticScorer(targets=targets)
        backend = InfiniteBackend()
        criteria = StoppingCriteria(timeout_seconds=0.1)
        loop = GenerationLoop(
            backend=backend,
            targets=targets,
            scorer=scorer,
            stopping_criteria=criteria,
        )
        result = loop.run()
        assert result.stop_reason == "timeout"
        assert result.elapsed_seconds <= 1.0  # generous upper bound

    def test_stops_when_backend_exhausted(self):
        targets = PhoneticTargetInventory(
            target_phonemes=["p", "b", "t", "d", "k"],
            unit="phoneme",
        )
        scorer = PhoneticScorer(targets=targets)
        # Backend returns candidates once, then empty
        backend = FakeBackend(responses=[
            [{"text": "s1", "phonemes": ["p", "b"]}],
            [],  # exhausted
        ])
        loop = GenerationLoop(
            backend=backend,
            targets=targets,
            scorer=scorer,
        )
        result = loop.run()
        assert result.stop_reason == "backend_exhausted"
        assert result.coverage < 1.0

    def test_stops_when_no_gain_candidates(self):
        """If backend returns candidates but none add new coverage, stop."""
        targets = PhoneticTargetInventory(
            target_phonemes=["p", "b"],
            unit="phoneme",
        )
        scorer = PhoneticScorer(targets=targets)
        # First response covers everything, second has no new units
        backend = FakeBackend(responses=[
            [{"text": "s1", "phonemes": ["p", "b"]}],
        ])
        loop = GenerationLoop(
            backend=backend,
            targets=targets,
            scorer=scorer,
        )
        result = loop.run()
        assert result.coverage == pytest.approx(1.0)
        assert result.stop_reason == "target_coverage"


# ---------------------------------------------------------------------------
# GenerationLoop: result metadata
# ---------------------------------------------------------------------------


class TestLoopResultMetadata:
    """Result contains correct metadata."""

    def test_elapsed_seconds(self):
        targets = PhoneticTargetInventory(
            target_phonemes=["p"],
            unit="phoneme",
        )
        scorer = PhoneticScorer(targets=targets)
        backend = FakeBackend(responses=[
            [{"text": "s", "phonemes": ["p"]}],
        ])
        loop = GenerationLoop(
            backend=backend,
            targets=targets,
            scorer=scorer,
        )
        result = loop.run()
        assert result.elapsed_seconds >= 0.0

    def test_iteration_count(self):
        targets = PhoneticTargetInventory(
            target_phonemes=["p", "b", "t"],
            unit="phoneme",
        )
        scorer = PhoneticScorer(targets=targets)
        backend = FakeBackend(responses=[
            [{"text": "s1", "phonemes": ["p"]}],
            [{"text": "s2", "phonemes": ["b"]}],
            [{"text": "s3", "phonemes": ["t"]}],
        ])
        loop = GenerationLoop(
            backend=backend,
            targets=targets,
            scorer=scorer,
        )
        result = loop.run()
        assert result.iterations == 3

    def test_backend_name_in_result(self):
        targets = PhoneticTargetInventory(
            target_phonemes=["p"],
            unit="phoneme",
        )
        scorer = PhoneticScorer(targets=targets)
        backend = FakeBackend(responses=[
            [{"text": "s", "phonemes": ["p"]}],
        ])
        loop = GenerationLoop(
            backend=backend,
            targets=targets,
            scorer=scorer,
        )
        result = loop.run()
        assert result.backend == "fake"

    def test_unit_in_result(self):
        targets = PhoneticTargetInventory(
            target_phonemes=["p", "b"],
            unit="diphone",
        )
        scorer = PhoneticScorer(targets=targets)
        backend = FakeBackend(responses=[
            [{"text": "s", "phonemes": ["p", "b", "p", "b"]}],
        ])
        criteria = StoppingCriteria(max_iterations=1)
        loop = GenerationLoop(
            backend=backend,
            targets=targets,
            scorer=scorer,
            stopping_criteria=criteria,
        )
        result = loop.run()
        assert result.unit == "diphone"


# ---------------------------------------------------------------------------
# GenerationLoop: callbacks
# ---------------------------------------------------------------------------


class TestLoopCallbacks:
    """Progress callbacks during generation."""

    def test_callback_called_each_iteration(self):
        targets = PhoneticTargetInventory(
            target_phonemes=["p", "b", "t"],
            unit="phoneme",
        )
        scorer = PhoneticScorer(targets=targets)
        backend = FakeBackend(responses=[
            [{"text": "s1", "phonemes": ["p"]}],
            [{"text": "s2", "phonemes": ["b"]}],
            [{"text": "s3", "phonemes": ["t"]}],
        ])
        callback_log = []

        def on_progress(info):
            callback_log.append(info)

        loop = GenerationLoop(
            backend=backend,
            targets=targets,
            scorer=scorer,
            on_progress=on_progress,
        )
        loop.run()
        assert len(callback_log) == 3

    def test_callback_receives_progress_info(self):
        targets = PhoneticTargetInventory(
            target_phonemes=["p", "b"],
            unit="phoneme",
        )
        scorer = PhoneticScorer(targets=targets)
        backend = FakeBackend(responses=[
            [{"text": "s1", "phonemes": ["p", "b"]}],
        ])
        callback_log = []

        def on_progress(info):
            callback_log.append(info)

        loop = GenerationLoop(
            backend=backend,
            targets=targets,
            scorer=scorer,
            on_progress=on_progress,
        )
        loop.run()
        assert len(callback_log) == 1
        info = callback_log[0]
        assert "iteration" in info
        assert "coverage" in info
        assert "sentence" in info
        assert "coverage_gain" in info


# ---------------------------------------------------------------------------
# GenerationLoop: logging
# ---------------------------------------------------------------------------


class TestLoopLogging:
    """Loop emits log messages."""

    def test_logs_iteration_progress(self, caplog):
        targets = PhoneticTargetInventory(
            target_phonemes=["p", "b"],
            unit="phoneme",
        )
        scorer = PhoneticScorer(targets=targets)
        backend = FakeBackend(responses=[
            [{"text": "s1", "phonemes": ["p", "b"]}],
        ])
        loop = GenerationLoop(
            backend=backend,
            targets=targets,
            scorer=scorer,
        )
        with caplog.at_level(logging.INFO, logger="corpusgen.generate.phon_ctg.loop"):
            loop.run()
        assert len(caplog.records) >= 1

    def test_logs_stop_reason(self, caplog):
        targets = PhoneticTargetInventory(
            target_phonemes=["p", "b"],
            unit="phoneme",
        )
        scorer = PhoneticScorer(targets=targets)
        backend = FakeBackend(responses=[
            [{"text": "s1", "phonemes": ["p", "b"]}],
        ])
        loop = GenerationLoop(
            backend=backend,
            targets=targets,
            scorer=scorer,
        )
        with caplog.at_level(logging.INFO, logger="corpusgen.generate.phon_ctg.loop"):
            loop.run()
        # At least one log message should mention the stop reason
        all_messages = " ".join(r.message for r in caplog.records)
        assert "target_coverage" in all_messages or "coverage" in all_messages.lower()


# ---------------------------------------------------------------------------
# GenerationLoop: candidates_per_iteration
# ---------------------------------------------------------------------------


class TestLoopCandidatesPerIteration:
    """Configurable number of candidates requested per iteration."""

    def test_default_candidates_per_iteration(self):
        targets = PhoneticTargetInventory(
            target_phonemes=["p"],
            unit="phoneme",
        )
        scorer = PhoneticScorer(targets=targets)
        backend = FakeBackend(responses=[
            [{"text": "s", "phonemes": ["p"]}],
        ])
        loop = GenerationLoop(
            backend=backend,
            targets=targets,
            scorer=scorer,
        )
        assert loop.candidates_per_iteration == 5  # sensible default

    def test_custom_candidates_per_iteration(self):
        targets = PhoneticTargetInventory(
            target_phonemes=["p"],
            unit="phoneme",
        )
        scorer = PhoneticScorer(targets=targets)
        backend = FakeBackend(responses=[
            [{"text": "s", "phonemes": ["p"]}],
        ])
        loop = GenerationLoop(
            backend=backend,
            targets=targets,
            scorer=scorer,
            candidates_per_iteration=10,
        )
        assert loop.candidates_per_iteration == 10


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestLoopEdgeCases:
    """Boundary conditions."""

    def test_empty_target_inventory(self):
        """No targets needed — should return immediately."""
        targets = PhoneticTargetInventory(
            target_phonemes=[],
            unit="phoneme",
        )
        scorer = PhoneticScorer(targets=targets)
        backend = FakeBackend()
        loop = GenerationLoop(
            backend=backend,
            targets=targets,
            scorer=scorer,
        )
        result = loop.run()
        assert result.coverage == 1.0
        assert result.num_generated == 0
        assert result.stop_reason == "target_coverage"

    def test_backend_returns_empty_first_call(self):
        targets = PhoneticTargetInventory(
            target_phonemes=["p"],
            unit="phoneme",
        )
        scorer = PhoneticScorer(targets=targets)
        backend = FakeBackend(responses=[])
        loop = GenerationLoop(
            backend=backend,
            targets=targets,
            scorer=scorer,
        )
        result = loop.run()
        assert result.stop_reason == "backend_exhausted"
        assert result.num_generated == 0
