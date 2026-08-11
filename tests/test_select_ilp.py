"""Tests for ILPSelector: exact Integer Linear Programming solver."""

from __future__ import annotations

import warnings

import pytest

import corpusgen.select.ilp as ilp_module
from corpusgen.select.greedy import GreedySelector
from corpusgen.select.ilp import ILPSelector
from corpusgen.select.result import SelectionResult

pulp = pytest.importorskip("pulp")


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


class TestILPSelector:
    """Tests for the ILP exact selection algorithm."""

    def test_pulp_33_apis_do_not_emit_deprecation_warnings(self):
        selector = ILPSelector()
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            result = selector.select(["s0"], [["a"]], {"a"})

        assert result.coverage == 1.0

    def test_binary_variable_falls_back_to_pulp_2_api(self, monkeypatch):
        sentinel = object()
        calls = []

        def fake_lp_variable(name, *, cat):
            calls.append((name, cat))
            return sentinel

        monkeypatch.setattr(ilp_module.pulp, "LpVariable", fake_lp_variable)

        result = ilp_module._add_binary_variable(object(), "x_0")

        assert result is sentinel
        assert calls == [("x_0", ilp_module.pulp.LpBinary)]

    def test_prefers_available_coin_solver(self, monkeypatch):
        class AvailableCoinSolver:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def available(self):
                return True

        def fail_bundled_solver(**kwargs):
            pytest.fail(f"bundled solver should not be constructed: {kwargs}")

        monkeypatch.setattr(ilp_module.pulp, "COIN_CMD", AvailableCoinSolver)
        monkeypatch.setattr(ilp_module.pulp, "PULP_CBC_CMD", fail_bundled_solver)

        solver = ilp_module._create_cbc_solver(12.5)

        assert isinstance(solver, AvailableCoinSolver)
        assert solver.kwargs == {"msg": False, "timeLimit": 12.5}

    def test_unavailable_coin_solver_uses_quiet_bundled_fallback(self, monkeypatch):
        class UnavailableCoinSolver:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def available(self):
                return False

        class BundledSolver:
            def __init__(self, **kwargs):
                warnings.warn(
                    "PULP_CBC_CMD is deprecated and will be removed in PuLP 4.0. "
                    "Install CBC separately.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                self.kwargs = kwargs

        monkeypatch.setattr(ilp_module.pulp, "COIN_CMD", UnavailableCoinSolver)
        monkeypatch.setattr(ilp_module.pulp, "PULP_CBC_CMD", BundledSolver)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solver = ilp_module._create_cbc_solver(None)

        assert isinstance(solver, BundledSolver)
        assert solver.kwargs == {"msg": False, "timeLimit": None}
        assert caught == []

    def test_returns_selection_result(
        self, candidates, candidate_phonemes, phoneme_target
    ):
        selector = ILPSelector()
        result = selector.select(candidates, candidate_phonemes, phoneme_target)
        assert isinstance(result, SelectionResult)

    def test_algorithm_name(self):
        selector = ILPSelector()
        assert selector.algorithm_name == "ilp"

    def test_finds_optimal_single_sentence(
        self, candidates, candidate_phonemes, phoneme_target
    ):
        """Candidate 4 covers everything — ILP should find it optimally."""
        selector = ILPSelector()
        result = selector.select(candidates, candidate_phonemes, phoneme_target)
        assert result.coverage == 1.0
        assert result.num_selected == 1
        assert 4 in result.selected_indices

    def test_finds_optimal_minimum_set(self):
        """ILP must find the minimum-cardinality covering set."""
        target = {"a", "b", "c", "d"}
        cands = ["s0", "s1", "s2", "s3"]
        phonemes = [
            ["a", "b", "c", "d"],   # covers all in 1
            ["a", "b"],              # needs partner
            ["c", "d"],              # partner for s1
            ["a", "c"],              # suboptimal
        ]
        selector = ILPSelector()
        result = selector.select(cands, phonemes, target)
        assert result.coverage == 1.0
        assert result.num_selected == 1  # Optimal: just s0
        assert 0 in result.selected_indices

    def test_optimal_beats_or_ties_greedy(self):
        """ILP should find a set at most as large as greedy's.

        This is the key property: ILP is the ground truth baseline.
        """
        target = {"a", "b", "c", "d", "e", "f"}
        cands = [f"s{i}" for i in range(6)]
        phonemes = [
            ["a", "b"],         # 0
            ["c", "d"],         # 1
            ["e", "f"],         # 2
            ["a", "c", "e"],    # 3: covers 3, greedy picks this first
            ["b", "d", "f"],    # 4: covers remaining 3
            ["a", "b", "c", "d", "e", "f"],  # 5: covers all in 1
        ]
        greedy_result = GreedySelector().select(cands, phonemes, target)
        ilp_result = ILPSelector().select(cands, phonemes, target)
        assert ilp_result.coverage == 1.0
        assert ilp_result.num_selected <= greedy_result.num_selected

    def test_ilp_finds_strictly_better_than_greedy(self):
        """Classic case where greedy is suboptimal.

        Target: {1,2,3,4,5,6}
        S0 = {1,2,3,4}  (4 elements)
        S1 = {1,2,5}    (3 elements — greedy won't pick this first)
        S2 = {3,4,6}    (3 elements — greedy won't pick this first)

        Greedy picks S0 (gain=4), then needs S1 or similar for 5,
        then another for 6 → 3 sentences.
        Optimal is S1 + S2 = 2 sentences.
        """
        target = {"1", "2", "3", "4", "5", "6"}
        cands = ["s0", "s1", "s2"]
        phonemes = [
            ["1", "2", "3", "4"],
            ["1", "2", "5"],
            ["3", "4", "6"],
        ]
        greedy_result = GreedySelector().select(cands, phonemes, target)
        ilp_result = ILPSelector().select(cands, phonemes, target)
        assert ilp_result.coverage == 1.0
        assert ilp_result.num_selected == 2
        assert ilp_result.num_selected < greedy_result.num_selected

    def test_max_sentences_budget(
        self, candidates, candidate_phonemes, phoneme_target
    ):
        cands = candidates[:4]
        phons = candidate_phonemes[:4]
        selector = ILPSelector()
        result = selector.select(cands, phons, phoneme_target, max_sentences=1)
        assert result.num_selected <= 1

    def test_target_coverage_threshold(self):
        """ILP respects target_coverage by requiring enough units covered."""
        target = {"a", "b", "c", "d", "e"}
        cands = ["s0", "s1", "s2"]
        phonemes = [["a", "b", "c"], ["d"], ["e"]]
        selector = ILPSelector()
        result = selector.select(cands, phonemes, target, target_coverage=0.6)
        assert result.coverage >= 0.6

    def test_impossible_full_coverage(self):
        target = {"a", "b", "x"}
        cands = ["s1"]
        phonemes = [["a", "b"]]
        selector = ILPSelector()
        result = selector.select(cands, phonemes, target)
        assert result.missing_units == {"x"}
        assert result.coverage == pytest.approx(2 / 3)

    def test_empty_candidates(self, phoneme_target):
        selector = ILPSelector()
        result = selector.select([], [], phoneme_target)
        assert result.coverage == 0.0
        assert result.num_selected == 0

    def test_empty_target(self):
        selector = ILPSelector()
        result = selector.select(["s1"], [["a"]], set())
        assert result.coverage == 1.0
        assert result.num_selected == 0

    def test_metadata_includes_solver_status(
        self, candidates, candidate_phonemes, phoneme_target
    ):
        selector = ILPSelector()
        result = selector.select(candidates, candidate_phonemes, phoneme_target)
        assert "solver_status" in result.metadata

    def test_diphone_unit(self):
        target = {"a-b", "b-c"}
        cands = ["s1"]
        phonemes = [["a", "b", "c"]]
        selector = ILPSelector(unit="diphone")
        result = selector.select(cands, phonemes, target)
        assert result.coverage == 1.0
        assert result.unit == "diphone"

    def test_indices_and_sentences_match(
        self, candidates, candidate_phonemes, phoneme_target
    ):
        selector = ILPSelector()
        result = selector.select(candidates, candidate_phonemes, phoneme_target)
        for idx, sent in zip(result.selected_indices, result.selected_sentences):
            assert sent == candidates[idx]
