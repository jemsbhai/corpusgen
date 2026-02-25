"""Tests for StochasticGreedySelector: randomized subsampling for scalability."""

from __future__ import annotations

import pytest

from corpusgen.select.stochastic import StochasticGreedySelector
from corpusgen.select.greedy import GreedySelector
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


class TestStochasticGreedySelector:
    """Tests for the stochastic greedy selection algorithm."""

    def test_returns_selection_result(
        self, candidates, candidate_phonemes, phoneme_target
    ):
        selector = StochasticGreedySelector(epsilon=0.1, seed=42)
        result = selector.select(candidates, candidate_phonemes, phoneme_target)
        assert isinstance(result, SelectionResult)

    def test_algorithm_name(self):
        selector = StochasticGreedySelector()
        assert selector.algorithm_name == "stochastic"

    def test_achieves_full_coverage_when_possible(
        self, candidates, candidate_phonemes, phoneme_target
    ):
        """With candidate 4 covering everything, stochastic should find it."""
        selector = StochasticGreedySelector(epsilon=0.1, seed=42)
        result = selector.select(candidates, candidate_phonemes, phoneme_target)
        assert result.coverage == 1.0
        assert result.missing_units == set()

    def test_deterministic_with_seed(
        self, candidates, candidate_phonemes, phoneme_target
    ):
        """Same seed produces identical results."""
        s1 = StochasticGreedySelector(epsilon=0.1, seed=123)
        s2 = StochasticGreedySelector(epsilon=0.1, seed=123)
        r1 = s1.select(candidates, candidate_phonemes, phoneme_target)
        r2 = s2.select(candidates, candidate_phonemes, phoneme_target)
        assert r1.selected_indices == r2.selected_indices
        assert r1.coverage == r2.coverage

    def test_different_seeds_may_differ(self):
        """Different seeds may produce different selection orders.

        We use a scenario where there are ties that randomization can break.
        """
        target = {"a", "b", "c", "d", "e", "f"}
        cands = [f"s{i}" for i in range(6)]
        phonemes = [
            ["a", "b"],
            ["c", "d"],
            ["e", "f"],
            ["a", "c", "e"],
            ["b", "d", "f"],
            ["a", "d"],
        ]
        results = set()
        for seed in range(20):
            s = StochasticGreedySelector(epsilon=0.5, seed=seed)
            r = s.select(cands, phonemes, target)
            results.add(tuple(r.selected_indices))
        # With epsilon=0.5, small sample sizes should produce some variation
        # (not guaranteed, but very likely with 20 seeds and 6 candidates)
        # At minimum all should achieve full coverage
        for seed in range(20):
            s = StochasticGreedySelector(epsilon=0.5, seed=seed)
            r = s.select(cands, phonemes, target)
            assert r.coverage == 1.0

    def test_epsilon_controls_sample_size(self):
        """Metadata should report the sample_size used."""
        selector = StochasticGreedySelector(epsilon=0.1, seed=42)
        target = {"a", "b", "c"}
        cands = [f"s{i}" for i in range(100)]
        phonemes = [["a"] for _ in range(100)]
        result = selector.select(cands, phonemes, target)
        assert "sample_size" in result.metadata
        assert result.metadata["sample_size"] >= 1

    def test_epsilon_zero_raises(self):
        """Epsilon must be positive."""
        with pytest.raises(ValueError, match="epsilon"):
            StochasticGreedySelector(epsilon=0.0)

    def test_epsilon_negative_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            StochasticGreedySelector(epsilon=-0.5)

    def test_epsilon_above_one_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            StochasticGreedySelector(epsilon=1.5)

    def test_max_sentences_budget(
        self, candidates, candidate_phonemes, phoneme_target
    ):
        cands = candidates[:4]
        phons = candidate_phonemes[:4]
        selector = StochasticGreedySelector(epsilon=0.1, seed=42)
        result = selector.select(cands, phons, phoneme_target, max_sentences=1)
        assert result.num_selected == 1

    def test_target_coverage_early_stop(
        self, candidates, candidate_phonemes, phoneme_target
    ):
        cands = candidates[:4]
        phons = candidate_phonemes[:4]
        selector = StochasticGreedySelector(epsilon=0.1, seed=42)
        result = selector.select(cands, phons, phoneme_target, target_coverage=0.6)
        assert result.coverage >= 0.6

    def test_empty_candidates(self, phoneme_target):
        selector = StochasticGreedySelector(epsilon=0.1, seed=42)
        result = selector.select([], [], phoneme_target)
        assert result.coverage == 0.0
        assert result.num_selected == 0

    def test_empty_target(self):
        selector = StochasticGreedySelector(epsilon=0.1, seed=42)
        result = selector.select(["s1"], [["a"]], set())
        assert result.coverage == 1.0
        assert result.num_selected == 0

    def test_impossible_full_coverage(self):
        target = {"a", "b", "x"}
        cands = ["s1"]
        phonemes = [["a", "b"]]
        selector = StochasticGreedySelector(epsilon=0.1, seed=42)
        result = selector.select(cands, phonemes, target)
        assert result.missing_units == {"x"}

    def test_diphone_unit(self):
        target = {"a-b", "b-c"}
        cands = ["s1"]
        phonemes = [["a", "b", "c"]]
        selector = StochasticGreedySelector(unit="diphone", epsilon=0.1, seed=42)
        result = selector.select(cands, phonemes, target)
        assert result.coverage == 1.0
        assert result.unit == "diphone"

    def test_metadata_includes_epsilon_and_seed(
        self, candidates, candidate_phonemes, phoneme_target
    ):
        selector = StochasticGreedySelector(epsilon=0.3, seed=99)
        result = selector.select(candidates, candidate_phonemes, phoneme_target)
        assert result.metadata["epsilon"] == 0.3
        assert result.metadata["seed"] == 99

    def test_indices_and_sentences_match(
        self, candidates, candidate_phonemes, phoneme_target
    ):
        selector = StochasticGreedySelector(epsilon=0.1, seed=42)
        result = selector.select(candidates, candidate_phonemes, phoneme_target)
        for idx, sent in zip(result.selected_indices, result.selected_sentences):
            assert sent == candidates[idx]
