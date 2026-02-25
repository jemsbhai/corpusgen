"""Tests for weighted selection across all algorithms."""

from __future__ import annotations

import pytest

from corpusgen.select import (
    GreedySelector,
    CELFSelector,
    StochasticGreedySelector,
    DistributionAwareSelector,
    select_sentences,
    SelectionResult,
)


@pytest.fixture
def weighted_scenario():
    """Scenario where weights change which candidate is preferred.

    Target: {a, b, c, d}
    s0 covers {a, b, c} — 3 units
    s1 covers {d}        — 1 unit

    Unweighted: greedy picks s0 first (gain=3).
    Weighted with d=10.0, others=1.0: s1 first (gain=10 vs 3).
    """
    return {
        "candidates": ["s0", "s1"],
        "candidate_phonemes": [["a", "b", "c"], ["d"]],
        "target_units": {"a", "b", "c", "d"},
        "weights": {"a": 1.0, "b": 1.0, "c": 1.0, "d": 10.0},
    }


class TestGreedyWeighted:
    def test_weights_change_selection_order(self, weighted_scenario):
        s = weighted_scenario
        selector = GreedySelector()
        result = selector.select(
            s["candidates"], s["candidate_phonemes"],
            s["target_units"], weights=s["weights"],
        )
        # s1 (d, gain=10) should be picked before s0 (a+b+c, gain=3)
        assert result.selected_indices[0] == 1

    def test_none_weights_is_unweighted(self, weighted_scenario):
        s = weighted_scenario
        selector = GreedySelector()
        result = selector.select(
            s["candidates"], s["candidate_phonemes"],
            s["target_units"], weights=None,
        )
        # Unweighted: s0 (3 units) first
        assert result.selected_indices[0] == 0


class TestCELFWeighted:
    def test_weights_change_selection_order(self, weighted_scenario):
        s = weighted_scenario
        selector = CELFSelector()
        result = selector.select(
            s["candidates"], s["candidate_phonemes"],
            s["target_units"], weights=s["weights"],
        )
        assert result.selected_indices[0] == 1

    def test_matches_greedy_weighted(self, weighted_scenario):
        """CELF with weights must produce same result as greedy with weights."""
        s = weighted_scenario
        greedy = GreedySelector().select(
            s["candidates"], s["candidate_phonemes"],
            s["target_units"], weights=s["weights"],
        )
        celf = CELFSelector().select(
            s["candidates"], s["candidate_phonemes"],
            s["target_units"], weights=s["weights"],
        )
        assert celf.selected_indices == greedy.selected_indices
        assert celf.coverage == greedy.coverage


class TestStochasticWeighted:
    def test_weights_influence_selection(self, weighted_scenario):
        s = weighted_scenario
        selector = StochasticGreedySelector(epsilon=0.1, seed=42)
        result = selector.select(
            s["candidates"], s["candidate_phonemes"],
            s["target_units"], weights=s["weights"],
        )
        # With only 2 candidates, sample will include both, so d=10 wins
        assert result.selected_indices[0] == 1


class TestDistributionWeighted:
    def test_accepts_weights(self):
        target_dist = {"a": 0.5, "b": 0.5}
        selector = DistributionAwareSelector(target_distribution=target_dist)
        result = selector.select(
            ["s0"], [["a", "b"]], {"a", "b"},
            weights={"a": 1.0, "b": 1.0},
        )
        assert result.coverage == 1.0


class TestILPWeighted:
    def test_weights_change_selection(self):
        pulp = pytest.importorskip("pulp")
        from corpusgen.select import ILPSelector

        # Same scenario: without weights, ILP picks s0 (fewer sentences
        # for 3 units). With heavy weight on d, both are needed anyway,
        # but the objective value changes.
        target = {"a", "b", "c", "d"}
        cands = ["s0", "s1"]
        phonemes = [["a", "b", "c"], ["d"]]
        weights = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 10.0}

        selector = ILPSelector()
        result = selector.select(
            cands, phonemes, target, weights=weights,
        )
        # Both needed for full coverage regardless of weights
        assert result.coverage == 1.0


class TestNSGA2Weighted:
    def test_accepts_weights(self):
        pymoo = pytest.importorskip("pymoo")
        from corpusgen.select import NSGA2Selector

        selector = NSGA2Selector(population_size=10, n_generations=10, seed=42)
        result = selector.select(
            ["s0", "s1"],
            [["a", "b"], ["c", "d"]],
            {"a", "b", "c", "d"},
            weights={"a": 1.0, "b": 1.0, "c": 1.0, "d": 5.0},
        )
        assert isinstance(result, SelectionResult)


class TestDispatcherWeighted:
    def test_weights_passed_through(self):
        result = select_sentences(
            ["s0", "s1"],
            target_phonemes=["a", "b", "c", "d"],
            candidate_phonemes=[["a", "b", "c"], ["d"]],
            algorithm="greedy",
            weights={"a": 1.0, "b": 1.0, "c": 1.0, "d": 10.0},
        )
        assert result.selected_indices[0] == 1
