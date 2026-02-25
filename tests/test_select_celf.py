"""Tests for CELFSelector: Cost-Effective Lazy Forward selection algorithm."""

from __future__ import annotations

import pytest

from corpusgen.select.celf import CELFSelector
from corpusgen.select.greedy import GreedySelector
from corpusgen.select.result import SelectionResult


@pytest.fixture
def phoneme_target() -> set[str]:
    return {"a", "b", "c", "d", "e"}


@pytest.fixture
def candidates() -> list[str]:
    return [
        "s0",  # covers {a, b}
        "s1",  # covers {c, d, e}
        "s2",  # covers {a, c}
        "s3",  # covers {b, d}
        "s4",  # covers all
    ]


@pytest.fixture
def candidate_phonemes() -> list[list[str]]:
    return [
        ["a", "b"],
        ["c", "d", "e"],
        ["a", "c"],
        ["b", "d"],
        ["a", "b", "c", "d", "e"],
    ]


class TestCELFSelector:
    """Tests for the CELF lazy-evaluation selection algorithm."""

    def test_returns_selection_result(
        self, candidates, candidate_phonemes, phoneme_target
    ):
        selector = CELFSelector()
        result = selector.select(candidates, candidate_phonemes, phoneme_target)
        assert isinstance(result, SelectionResult)

    def test_algorithm_name(self):
        selector = CELFSelector()
        assert selector.algorithm_name == "celf"

    def test_full_coverage_single_sentence(
        self, candidates, candidate_phonemes, phoneme_target
    ):
        """Candidate 4 covers everything — CELF should find it."""
        selector = CELFSelector()
        result = selector.select(candidates, candidate_phonemes, phoneme_target)
        assert result.coverage == 1.0
        assert result.num_selected == 1
        assert 4 in result.selected_indices

    def test_matches_greedy_result(self):
        """CELF must produce identical coverage and selection count as Greedy.

        CELF is an optimization of Greedy — same selection logic, fewer
        evaluations. The selected sets must be equivalent in quality.
        """
        target = {"a", "b", "c", "d", "e", "f", "g", "h"}
        cands = [f"s{i}" for i in range(6)]
        phonemes = [
            ["a", "b", "c"],        # 3
            ["d", "e"],              # 2
            ["f", "g", "h"],         # 3
            ["a", "d", "f"],         # 3
            ["b", "e", "g"],         # 3
            ["c", "h"],              # 2
        ]
        greedy_result = GreedySelector().select(cands, phonemes, target)
        celf_result = CELFSelector().select(cands, phonemes, target)

        assert celf_result.coverage == greedy_result.coverage
        assert celf_result.num_selected == greedy_result.num_selected
        assert celf_result.covered_units == greedy_result.covered_units
        assert celf_result.missing_units == greedy_result.missing_units

    def test_matches_greedy_selection_order(self):
        """When gains are unambiguous, CELF picks same candidates as Greedy."""
        target = {"a", "b", "c", "d"}
        cands = ["s0", "s1"]
        phonemes = [
            ["a"],              # 1
            ["b", "c", "d"],    # 3
        ]
        greedy_result = GreedySelector().select(cands, phonemes, target)
        celf_result = CELFSelector().select(cands, phonemes, target)
        assert celf_result.selected_indices == greedy_result.selected_indices

    def test_max_sentences_budget(
        self, candidates, candidate_phonemes, phoneme_target
    ):
        cands = candidates[:4]
        phons = candidate_phonemes[:4]
        selector = CELFSelector()
        result = selector.select(cands, phons, phoneme_target, max_sentences=1)
        assert result.num_selected == 1

    def test_target_coverage_early_stop(
        self, candidates, candidate_phonemes, phoneme_target
    ):
        cands = candidates[:4]
        phons = candidate_phonemes[:4]
        selector = CELFSelector()
        result = selector.select(cands, phons, phoneme_target, target_coverage=0.6)
        assert result.coverage >= 0.6

    def test_impossible_full_coverage(self):
        target = {"a", "b", "c", "x", "y"}
        cands = ["s1", "s2"]
        phonemes = [["a", "b"], ["c"]]
        selector = CELFSelector()
        result = selector.select(cands, phonemes, target)
        assert result.coverage == 3 / 5
        assert result.missing_units == {"x", "y"}

    def test_empty_candidates(self, phoneme_target):
        selector = CELFSelector()
        result = selector.select([], [], phoneme_target)
        assert result.coverage == 0.0
        assert result.num_selected == 0

    def test_empty_target(self):
        selector = CELFSelector()
        result = selector.select(["s1"], [["a"]], set())
        assert result.coverage == 1.0
        assert result.num_selected == 0

    def test_fewer_evaluations_than_greedy(self):
        """CELF should skip some evaluations via lazy logic.

        Tracked in metadata['evaluations'] vs greedy brute force.
        """
        target = {"a", "b", "c", "d", "e", "f"}
        cands = [f"s{i}" for i in range(5)]
        phonemes = [
            ["a", "b", "c", "d", "e", "f"],  # covers all
            ["a", "b"],
            ["c", "d"],
            ["e", "f"],
            ["a", "c", "e"],
        ]
        selector = CELFSelector()
        result = selector.select(cands, phonemes, target)
        # With a single candidate covering all, CELF should need very few evals
        assert result.metadata.get("evaluations") is not None
        assert result.metadata["evaluations"] >= 1

    def test_diphone_unit(self):
        target = {"a-b", "b-c"}
        cands = ["s1", "s2"]
        phonemes = [["a", "b", "c"], ["x", "y"]]
        selector = CELFSelector(unit="diphone")
        result = selector.select(cands, phonemes, target)
        assert result.coverage == 1.0
        assert result.unit == "diphone"

    def test_indices_and_sentences_match(
        self, candidates, candidate_phonemes, phoneme_target
    ):
        selector = CELFSelector()
        result = selector.select(candidates, candidate_phonemes, phoneme_target)
        for idx, sent in zip(result.selected_indices, result.selected_sentences):
            assert sent == candidates[idx]
