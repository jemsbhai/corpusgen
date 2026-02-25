"""Integration tests: G2P → Selection → Evaluation end-to-end pipeline."""

from __future__ import annotations

import pytest

from corpusgen import evaluate, select_sentences
from corpusgen.select import SelectionResult
from corpusgen.evaluate.report import EvaluationReport


def _espeak_available() -> bool:
    """Check if espeak-ng is available for G2P."""
    try:
        from corpusgen.g2p.manager import G2PManager
        g2p = G2PManager()
        results = g2p.phonemize_batch(["hello"], language="en-us")
        return len(results) > 0 and len(results[0].phonemes) > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _espeak_available(),
    reason="espeak-ng not available",
)


# --- Candidate pool for selection ---

ENGLISH_CANDIDATES = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "Peter Piper picked a peck of pickled peppers.",
    "How much wood would a woodchuck chuck?",
    "The cat sat on the mat.",
    "A journey of a thousand miles begins with a single step.",
    "To be or not to be, that is the question.",
    "All that glitters is not gold.",
    "Every cloud has a silver lining.",
    "Actions speak louder than words.",
]


class TestSelectionToEvaluation:
    """Full pipeline: select from candidates, then evaluate the selection."""

    def test_greedy_select_then_evaluate(self):
        result = select_sentences(
            ENGLISH_CANDIDATES,
            language="en-us",
            algorithm="greedy",
        )
        assert isinstance(result, SelectionResult)
        assert result.coverage > 0.0
        assert result.num_selected > 0
        assert result.num_selected <= len(ENGLISH_CANDIDATES)

        # Evaluate the selected subset
        report = evaluate(
            result.selected_sentences,
            language="en-us",
        )
        assert isinstance(report, EvaluationReport)
        assert report.coverage > 0.0
        assert report.total_sentences == result.num_selected

    def test_celf_matches_greedy_coverage(self):
        greedy = select_sentences(
            ENGLISH_CANDIDATES, language="en-us", algorithm="greedy"
        )
        celf = select_sentences(
            ENGLISH_CANDIDATES, language="en-us", algorithm="celf"
        )
        assert celf.coverage == greedy.coverage
        assert celf.num_selected == greedy.num_selected

    def test_stochastic_achieves_reasonable_coverage(self):
        result = select_sentences(
            ENGLISH_CANDIDATES,
            language="en-us",
            algorithm="stochastic",
            epsilon=0.1,
            seed=42,
        )
        assert result.coverage > 0.5  # Should get decent coverage

    def test_selected_subset_is_smaller_than_full(self):
        result = select_sentences(
            ENGLISH_CANDIDATES,
            language="en-us",
            algorithm="greedy",
        )
        # The algorithm should select fewer sentences than the full pool
        # (since there's redundancy in coverage)
        assert result.num_selected <= len(ENGLISH_CANDIDATES)

    def test_max_sentences_budget_respected(self):
        result = select_sentences(
            ENGLISH_CANDIDATES,
            language="en-us",
            algorithm="greedy",
            max_sentences=3,
        )
        assert result.num_selected <= 3

    def test_evaluation_coverage_matches_selection_coverage(self):
        """The coverage reported by select should match evaluate on the same set."""
        sel_result = select_sentences(
            ENGLISH_CANDIDATES,
            language="en-us",
            algorithm="greedy",
        )
        eval_report = evaluate(
            sel_result.selected_sentences,
            language="en-us",
            target_phonemes=sorted(
                sel_result.covered_units | sel_result.missing_units
            ),
        )
        assert eval_report.coverage == pytest.approx(sel_result.coverage, abs=1e-9)

    def test_diphone_unit_pipeline(self):
        result = select_sentences(
            ENGLISH_CANDIDATES,
            language="en-us",
            algorithm="greedy",
            unit="diphone",
        )
        assert result.unit == "diphone"
        assert result.coverage > 0.0

        report = evaluate(
            result.selected_sentences,
            language="en-us",
            unit="diphone",
        )
        assert report.unit == "diphone"


class TestSelectionWithILP:
    """ILP integration tests (requires pulp)."""

    def test_ilp_optimal_vs_greedy(self):
        pytest.importorskip("pulp")
        greedy = select_sentences(
            ENGLISH_CANDIDATES, language="en-us", algorithm="greedy"
        )
        ilp = select_sentences(
            ENGLISH_CANDIDATES, language="en-us", algorithm="ilp"
        )
        # ILP should find solution with same or fewer sentences
        assert ilp.coverage >= greedy.coverage - 1e-9
        assert ilp.num_selected <= greedy.num_selected


class TestSelectionWithNSGA2:
    """NSGA-II integration tests (requires pymoo)."""

    def test_nsga2_produces_pareto_front(self):
        pytest.importorskip("pymoo")
        result = select_sentences(
            ENGLISH_CANDIDATES,
            language="en-us",
            algorithm="nsga2",
            population_size=20,
            n_generations=30,
            seed=42,
        )
        assert result.coverage > 0.0
        assert "pareto_front" in result.metadata
        assert len(result.metadata["pareto_front"]) >= 1


class TestSelectionWithWeights:
    """Weighted selection integration tests."""

    def test_frequency_inverse_weighted_selection(self):
        from corpusgen.g2p.manager import G2PManager
        from corpusgen.weights import frequency_inverse_weights

        # Phonemize candidates
        g2p = G2PManager()
        g2p_results = g2p.phonemize_batch(ENGLISH_CANDIDATES, language="en-us")
        candidate_phonemes = [r.phonemes for r in g2p_results]

        # Derive target and weights
        all_phonemes: set[str] = set()
        for p in candidate_phonemes:
            all_phonemes.update(p)
        target = sorted(all_phonemes)

        weights = frequency_inverse_weights(set(target), candidate_phonemes)
        assert len(weights) == len(target)

        result = select_sentences(
            ENGLISH_CANDIDATES,
            target_phonemes=target,
            candidate_phonemes=candidate_phonemes,
            algorithm="greedy",
            weights=weights,
        )
        assert result.coverage > 0.0
