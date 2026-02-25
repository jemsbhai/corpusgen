"""Tests for NSGA2Selector: multi-objective Pareto optimization."""

from __future__ import annotations

import pytest

pymoo = pytest.importorskip("pymoo")

from corpusgen.select.nsga2 import NSGA2Selector
from corpusgen.select.result import SelectionResult


@pytest.fixture
def phoneme_target() -> set[str]:
    return {"a", "b", "c", "d", "e"}


@pytest.fixture
def candidates() -> list[str]:
    return ["s0", "s1", "s2", "s3", "s4"]


@pytest.fixture
def candidate_phonemes() -> list[list[str]]:
    return [
        ["a", "b"],
        ["c", "d", "e"],
        ["a", "c"],
        ["b", "d"],
        ["a", "b", "c", "d", "e"],
    ]


class TestNSGA2Selector:
    """Tests for the NSGA-II multi-objective selection algorithm."""

    def test_returns_selection_result(
        self, candidates, candidate_phonemes, phoneme_target
    ):
        selector = NSGA2Selector(population_size=20, n_generations=30, seed=42)
        result = selector.select(candidates, candidate_phonemes, phoneme_target)
        assert isinstance(result, SelectionResult)

    def test_algorithm_name(self):
        selector = NSGA2Selector()
        assert selector.algorithm_name == "nsga2"

    def test_finds_full_coverage(
        self, candidates, candidate_phonemes, phoneme_target
    ):
        """With candidate 4 covering everything, NSGA-II should find it."""
        selector = NSGA2Selector(population_size=20, n_generations=50, seed=42)
        result = selector.select(candidates, candidate_phonemes, phoneme_target)
        assert result.coverage == 1.0

    def test_deterministic_with_seed(
        self, candidates, candidate_phonemes, phoneme_target
    ):
        s1 = NSGA2Selector(population_size=20, n_generations=30, seed=123)
        s2 = NSGA2Selector(population_size=20, n_generations=30, seed=123)
        r1 = s1.select(candidates, candidate_phonemes, phoneme_target)
        r2 = s2.select(candidates, candidate_phonemes, phoneme_target)
        assert r1.selected_indices == r2.selected_indices
        assert r1.coverage == r2.coverage

    def test_metadata_has_pareto_front(
        self, candidates, candidate_phonemes, phoneme_target
    ):
        selector = NSGA2Selector(population_size=20, n_generations=30, seed=42)
        result = selector.select(candidates, candidate_phonemes, phoneme_target)
        assert "pareto_front" in result.metadata
        front = result.metadata["pareto_front"]
        assert isinstance(front, list)
        assert len(front) >= 1
        # Each entry should have coverage, n_sentences, and indices
        for entry in front:
            assert "coverage" in entry
            assert "n_sentences" in entry
            assert "selected_indices" in entry

    def test_pareto_front_is_nondominated(
        self, candidates, candidate_phonemes, phoneme_target
    ):
        """No solution on the front should be dominated by another."""
        selector = NSGA2Selector(population_size=30, n_generations=50, seed=42)
        result = selector.select(candidates, candidate_phonemes, phoneme_target)
        front = result.metadata["pareto_front"]
        for i, a in enumerate(front):
            for j, b in enumerate(front):
                if i == j:
                    continue
                # a should NOT be dominated by b
                # dominated = b is better or equal on all objectives and
                # strictly better on at least one
                b_dominates = (
                    b["coverage"] >= a["coverage"]
                    and b["n_sentences"] <= a["n_sentences"]
                    and (
                        b["coverage"] > a["coverage"]
                        or b["n_sentences"] < a["n_sentences"]
                    )
                )
                assert not b_dominates, (
                    f"Solution {i} is dominated by {j}: {a} vs {b}"
                )

    def test_selects_highest_coverage_from_front(
        self, candidates, candidate_phonemes, phoneme_target
    ):
        """Returned result should be the Pareto-front solution with highest coverage."""
        selector = NSGA2Selector(population_size=20, n_generations=30, seed=42)
        result = selector.select(candidates, candidate_phonemes, phoneme_target)
        front = result.metadata["pareto_front"]
        max_cov = max(e["coverage"] for e in front)
        assert result.coverage == max_cov

    def test_max_sentences_budget(
        self, candidates, candidate_phonemes, phoneme_target
    ):
        selector = NSGA2Selector(population_size=20, n_generations=30, seed=42)
        result = selector.select(
            candidates, candidate_phonemes, phoneme_target, max_sentences=2
        )
        assert result.num_selected <= 2

    def test_empty_candidates(self, phoneme_target):
        selector = NSGA2Selector(population_size=20, n_generations=30, seed=42)
        result = selector.select([], [], phoneme_target)
        assert result.coverage == 0.0
        assert result.num_selected == 0

    def test_empty_target(self):
        selector = NSGA2Selector(population_size=20, n_generations=30, seed=42)
        result = selector.select(["s0"], [["a"]], set())
        assert result.coverage == 1.0
        assert result.num_selected == 0

    def test_impossible_full_coverage(self):
        target = {"a", "b", "x"}
        cands = ["s0"]
        phonemes = [["a", "b"]]
        selector = NSGA2Selector(population_size=20, n_generations=30, seed=42)
        result = selector.select(cands, phonemes, target)
        assert result.missing_units == {"x"}

    def test_with_target_distribution(self):
        """Three objectives: coverage, count, and KL-divergence."""
        target_dist = {"a": 0.5, "b": 0.5}
        selector = NSGA2Selector(
            target_distribution=target_dist,
            population_size=20,
            n_generations=30,
            seed=42,
        )
        cands = ["s0", "s1"]
        phonemes = [["a", "b"], ["a", "a"]]
        result = selector.select(cands, phonemes, {"a", "b"})
        assert result.coverage == 1.0
        # Pareto front entries should include kl_divergence
        for entry in result.metadata["pareto_front"]:
            assert "kl_divergence" in entry

    def test_diphone_unit(self):
        target = {"a-b", "b-c"}
        cands = ["s0"]
        phonemes = [["a", "b", "c"]]
        selector = NSGA2Selector(
            unit="diphone", population_size=20, n_generations=30, seed=42
        )
        result = selector.select(cands, phonemes, target)
        assert result.coverage == 1.0
        assert result.unit == "diphone"

    def test_indices_and_sentences_match(
        self, candidates, candidate_phonemes, phoneme_target
    ):
        selector = NSGA2Selector(population_size=20, n_generations=30, seed=42)
        result = selector.select(candidates, candidate_phonemes, phoneme_target)
        for idx, sent in zip(result.selected_indices, result.selected_sentences):
            assert sent == candidates[idx]

    def test_population_size_validation(self):
        with pytest.raises(ValueError, match="population_size"):
            NSGA2Selector(population_size=1)

    def test_n_generations_validation(self):
        with pytest.raises(ValueError, match="n_generations"):
            NSGA2Selector(n_generations=0)
