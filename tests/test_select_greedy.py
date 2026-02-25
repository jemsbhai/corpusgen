"""Tests for GreedySelector: standard greedy Set Cover algorithm."""

from __future__ import annotations

import pytest

from corpusgen.select.greedy import GreedySelector
from corpusgen.select.result import SelectionResult


@pytest.fixture
def phoneme_target() -> set[str]:
    """Simple phoneme target set."""
    return {"a", "b", "c", "d", "e"}


@pytest.fixture
def candidates() -> list[str]:
    """Candidate sentences."""
    return [
        "sentence covering a b",      # 0: covers {a, b}
        "sentence covering c d e",    # 1: covers {c, d, e}
        "sentence covering a c",      # 2: covers {a, c}
        "sentence covering b d",      # 3: covers {b, d}
        "sentence covering a b c d e",  # 4: covers all
    ]


@pytest.fixture
def candidate_phonemes() -> list[list[str]]:
    """Pre-phonemized candidates matching the candidates fixture."""
    return [
        ["a", "b"],          # 0
        ["c", "d", "e"],     # 1
        ["a", "c"],          # 2
        ["b", "d"],          # 3
        ["a", "b", "c", "d", "e"],  # 4
    ]


class TestGreedySelector:
    """Tests for the greedy Set Cover selection algorithm."""

    def test_returns_selection_result(
        self, candidates, candidate_phonemes, phoneme_target
    ):
        selector = GreedySelector()
        result = selector.select(candidates, candidate_phonemes, phoneme_target)
        assert isinstance(result, SelectionResult)

    def test_algorithm_name(self):
        selector = GreedySelector()
        assert selector.algorithm_name == "greedy"

    def test_full_coverage_single_sentence(
        self, candidates, candidate_phonemes, phoneme_target
    ):
        """Candidate 4 covers everything — greedy should pick it first."""
        selector = GreedySelector()
        result = selector.select(candidates, candidate_phonemes, phoneme_target)
        assert result.coverage == 1.0
        assert result.missing_units == set()
        assert 4 in result.selected_indices
        assert result.num_selected == 1

    def test_full_coverage_multiple_sentences(self, phoneme_target):
        """When no single candidate covers all, greedy needs multiple."""
        cands = ["s1", "s2", "s3"]
        phonemes = [
            ["a", "b", "c"],   # covers 3
            ["d", "e"],         # covers 2
            ["a", "d"],         # covers 2 (redundant)
        ]
        selector = GreedySelector()
        result = selector.select(cands, phonemes, phoneme_target)
        assert result.coverage == 1.0
        # Should pick s1 (3 new) then s2 (2 new) = full coverage in 2
        assert result.num_selected == 2
        assert 0 in result.selected_indices
        assert 1 in result.selected_indices

    def test_greedy_picks_largest_marginal_gain(self):
        """Greedy should always pick the candidate with most new units."""
        target = {"a", "b", "c", "d"}
        cands = ["s1", "s2"]
        phonemes = [
            ["a"],              # covers 1
            ["b", "c", "d"],    # covers 3
        ]
        selector = GreedySelector()
        result = selector.select(cands, phonemes, target)
        # s2 first (3 new), then s1 (1 new)
        assert result.selected_indices[0] == 1
        assert result.selected_indices[1] == 0

    def test_max_sentences_budget(
        self, candidates, candidate_phonemes, phoneme_target
    ):
        """Respect the max_sentences budget constraint."""
        # Exclude candidate 4 so greedy needs multiple
        cands = candidates[:4]
        phons = candidate_phonemes[:4]
        selector = GreedySelector()
        result = selector.select(cands, phons, phoneme_target, max_sentences=1)
        assert result.num_selected == 1
        # With budget=1, cannot get full coverage
        assert result.coverage < 1.0

    def test_target_coverage_early_stop(
        self, candidates, candidate_phonemes, phoneme_target
    ):
        """Stop early when target_coverage threshold is reached."""
        # Exclude candidate 4
        cands = candidates[:4]
        phons = candidate_phonemes[:4]
        selector = GreedySelector()
        result = selector.select(
            cands, phons, phoneme_target, target_coverage=0.6
        )
        # 3/5 = 0.6, so picking one candidate with 3 units should suffice
        assert result.coverage >= 0.6

    def test_impossible_full_coverage(self):
        """When candidates cannot cover all targets, return partial coverage."""
        target = {"a", "b", "c", "x", "y"}
        cands = ["s1", "s2"]
        phonemes = [["a", "b"], ["c"]]
        selector = GreedySelector()
        result = selector.select(cands, phonemes, target)
        assert result.coverage == 3 / 5
        assert result.missing_units == {"x", "y"}
        assert result.covered_units == {"a", "b", "c"}

    def test_empty_candidates(self, phoneme_target):
        """Empty candidate list returns zero coverage."""
        selector = GreedySelector()
        result = selector.select([], [], phoneme_target)
        assert result.coverage == 0.0
        assert result.num_selected == 0
        assert result.missing_units == phoneme_target

    def test_empty_target(self):
        """Empty target set means instant full coverage."""
        selector = GreedySelector()
        result = selector.select(["s1"], [["a"]], set())
        assert result.coverage == 1.0
        assert result.num_selected == 0

    def test_elapsed_seconds_positive(
        self, candidates, candidate_phonemes, phoneme_target
    ):
        selector = GreedySelector()
        result = selector.select(candidates, candidate_phonemes, phoneme_target)
        assert result.elapsed_seconds >= 0.0

    def test_iterations_tracked(
        self, candidates, candidate_phonemes, phoneme_target
    ):
        selector = GreedySelector()
        result = selector.select(candidates, candidate_phonemes, phoneme_target)
        assert result.iterations >= 1

    def test_diphone_unit(self):
        """Greedy works with diphone units."""
        target = {"a-b", "b-c"}
        cands = ["s1", "s2"]
        phonemes = [["a", "b", "c"], ["x", "y"]]
        selector = GreedySelector(unit="diphone")
        result = selector.select(cands, phonemes, target)
        assert result.coverage == 1.0
        assert result.unit == "diphone"
        assert result.num_selected == 1
        assert 0 in result.selected_indices

    def test_triphone_unit(self):
        """Greedy works with triphone units."""
        target = {"a-b-c"}
        cands = ["s1"]
        phonemes = [["a", "b", "c"]]
        selector = GreedySelector(unit="triphone")
        result = selector.select(cands, phonemes, target)
        assert result.coverage == 1.0
        assert result.unit == "triphone"

    def test_no_duplicate_selections(self):
        """Each candidate should be selected at most once."""
        target = {"a", "b"}
        cands = ["s1"]
        phonemes = [["a", "b"]]
        selector = GreedySelector()
        result = selector.select(cands, phonemes, target)
        assert len(result.selected_indices) == len(set(result.selected_indices))

    def test_indices_and_sentences_match(
        self, candidates, candidate_phonemes, phoneme_target
    ):
        """selected_sentences[i] must correspond to candidates[selected_indices[i]]."""
        selector = GreedySelector()
        result = selector.select(candidates, candidate_phonemes, phoneme_target)
        for idx, sent in zip(result.selected_indices, result.selected_sentences):
            assert sent == candidates[idx]
