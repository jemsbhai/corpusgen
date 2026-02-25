"""Tests for select_sentences() top-level dispatcher."""

from __future__ import annotations

import pytest

from corpusgen.select import select_sentences, SelectionResult


class TestSelectSentencesDispatcher:
    """Tests for the select_sentences() top-level API."""

    @pytest.fixture
    def simple_candidates(self) -> list[str]:
        return ["hello world", "foo bar baz", "the cat sat"]

    def test_returns_selection_result(self, simple_candidates):
        result = select_sentences(
            simple_candidates,
            language="en-us",
            algorithm="greedy",
        )
        assert isinstance(result, SelectionResult)

    def test_default_algorithm_is_greedy(self, simple_candidates):
        result = select_sentences(simple_candidates, language="en-us")
        assert result.algorithm == "greedy"

    def test_greedy_algorithm(self, simple_candidates):
        result = select_sentences(
            simple_candidates, language="en-us", algorithm="greedy"
        )
        assert result.algorithm == "greedy"

    def test_celf_algorithm(self, simple_candidates):
        result = select_sentences(
            simple_candidates, language="en-us", algorithm="celf"
        )
        assert result.algorithm == "celf"

    def test_stochastic_algorithm(self, simple_candidates):
        result = select_sentences(
            simple_candidates,
            language="en-us",
            algorithm="stochastic",
            epsilon=0.3,
            seed=42,
        )
        assert result.algorithm == "stochastic"

    def test_invalid_algorithm_raises(self, simple_candidates):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            select_sentences(
                simple_candidates, language="en-us", algorithm="bogus"
            )

    def test_max_sentences_passed_through(self, simple_candidates):
        result = select_sentences(
            simple_candidates,
            language="en-us",
            algorithm="greedy",
            max_sentences=1,
        )
        assert result.num_selected <= 1

    def test_target_coverage_passed_through(self, simple_candidates):
        result = select_sentences(
            simple_candidates,
            language="en-us",
            algorithm="greedy",
            target_coverage=0.5,
        )
        assert result.coverage >= 0.5

    def test_pre_phonemized_skips_g2p(self):
        """When candidate_phonemes is provided, G2P is skipped."""
        cands = ["s0", "s1"]
        phonemes = [["a", "b"], ["c", "d"]]
        target = ["a", "b", "c", "d"]
        result = select_sentences(
            cands,
            target_phonemes=target,
            candidate_phonemes=phonemes,
            algorithm="greedy",
        )
        assert result.coverage == 1.0
        assert result.algorithm == "greedy"

    def test_pre_phonemized_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="candidate_phonemes"):
            select_sentences(
                ["s0", "s1"],
                target_phonemes=["a"],
                candidate_phonemes=[["a"]],  # length 1 vs 2 candidates
                algorithm="greedy",
            )

    def test_target_phonemes_none_derives_from_corpus(self):
        """When target_phonemes is None, derive from all candidates."""
        cands = ["s0", "s1"]
        phonemes = [["a", "b"], ["c"]]
        result = select_sentences(
            cands,
            target_phonemes=None,
            candidate_phonemes=phonemes,
            algorithm="greedy",
        )
        # Derived target is {a, b, c} — all covered
        assert result.coverage == 1.0

    def test_unit_phoneme(self):
        cands = ["s0"]
        phonemes = [["a", "b"]]
        result = select_sentences(
            cands,
            target_phonemes=["a", "b"],
            candidate_phonemes=phonemes,
            unit="phoneme",
        )
        assert result.unit == "phoneme"

    def test_unit_diphone(self):
        cands = ["s0"]
        phonemes = [["a", "b", "c"]]
        result = select_sentences(
            cands,
            target_phonemes=["a", "b", "c"],
            candidate_phonemes=phonemes,
            unit="diphone",
        )
        assert result.unit == "diphone"

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError, match="Invalid unit"):
            select_sentences(
                ["s0"],
                target_phonemes=["a"],
                candidate_phonemes=[["a"]],
                unit="quadphone",
            )

    def test_empty_candidates(self):
        result = select_sentences(
            [],
            target_phonemes=["a"],
            candidate_phonemes=[],
            algorithm="greedy",
        )
        assert result.num_selected == 0
        assert result.coverage == 0.0

    def test_kwargs_forwarded_to_stochastic(self):
        """epsilon and seed should reach the StochasticGreedySelector."""
        cands = ["s0", "s1"]
        phonemes = [["a"], ["b"]]
        r1 = select_sentences(
            cands,
            target_phonemes=["a", "b"],
            candidate_phonemes=phonemes,
            algorithm="stochastic",
            epsilon=0.5,
            seed=42,
        )
        r2 = select_sentences(
            cands,
            target_phonemes=["a", "b"],
            candidate_phonemes=phonemes,
            algorithm="stochastic",
            epsilon=0.5,
            seed=42,
        )
        assert r1.selected_indices == r2.selected_indices

    def test_kwargs_forwarded_to_distribution(self):
        cands = ["s0", "s1"]
        phonemes = [["a", "b"], ["c", "d"]]
        result = select_sentences(
            cands,
            target_phonemes=["a", "b", "c", "d"],
            candidate_phonemes=phonemes,
            algorithm="distribution",
            target_distribution={"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25},
        )
        assert result.algorithm == "distribution"
        assert "kl_divergence" in result.metadata


class TestSelectSentencesILP:
    """ILP-specific dispatcher tests (requires pulp)."""

    def test_ilp_algorithm(self):
        pytest.importorskip("pulp")
        cands = ["s0", "s1"]
        phonemes = [["a", "b"], ["c", "d"]]
        result = select_sentences(
            cands,
            target_phonemes=["a", "b", "c", "d"],
            candidate_phonemes=phonemes,
            algorithm="ilp",
        )
        assert result.algorithm == "ilp"


class TestSelectSentencesNSGA2:
    """NSGA-II-specific dispatcher tests (requires pymoo)."""

    def test_nsga2_algorithm(self):
        pytest.importorskip("pymoo")
        cands = ["s0", "s1"]
        phonemes = [["a", "b"], ["c", "d"]]
        result = select_sentences(
            cands,
            target_phonemes=["a", "b", "c", "d"],
            candidate_phonemes=phonemes,
            algorithm="nsga2",
            population_size=10,
            n_generations=10,
            seed=42,
        )
        assert result.algorithm == "nsga2"
        assert "pareto_front" in result.metadata
