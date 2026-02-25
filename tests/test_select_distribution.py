"""Tests for DistributionAwareSelector: frequency-distribution matching."""

from __future__ import annotations

import pytest

from corpusgen.select.distribution import DistributionAwareSelector
from corpusgen.select.greedy import GreedySelector
from corpusgen.select.result import SelectionResult


@pytest.fixture
def uniform_target() -> dict[str, float]:
    """Uniform distribution over 4 phonemes."""
    return {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}


@pytest.fixture
def skewed_target() -> dict[str, float]:
    """Skewed distribution: 'a' should dominate."""
    return {"a": 0.7, "b": 0.1, "c": 0.1, "d": 0.1}


class TestDistributionAwareSelector:
    """Tests for distribution-aware selection algorithm."""

    def test_returns_selection_result(self, uniform_target):
        selector = DistributionAwareSelector(target_distribution=uniform_target)
        cands = ["s0", "s1"]
        phonemes = [["a", "b"], ["c", "d"]]
        result = selector.select(
            cands, phonemes, set(uniform_target.keys())
        )
        assert isinstance(result, SelectionResult)

    def test_algorithm_name(self, uniform_target):
        selector = DistributionAwareSelector(target_distribution=uniform_target)
        assert selector.algorithm_name == "distribution"

    def test_achieves_full_coverage(self, uniform_target):
        cands = ["s0", "s1"]
        phonemes = [["a", "b", "c"], ["c", "d"]]
        selector = DistributionAwareSelector(target_distribution=uniform_target)
        result = selector.select(cands, phonemes, set(uniform_target.keys()))
        assert result.coverage == 1.0

    def test_prefers_distribution_match(self, skewed_target):
        """Should prefer candidates that bring distribution closer to target.

        With target a=0.7, b=0.1, c=0.1, d=0.1:
        - s0 has many 'a's → better distributional match
        - s1 has equal mix → worse match for this target
        """
        cands = ["s0", "s1"]
        phonemes = [
            ["a", "a", "a", "a", "a", "a", "a", "b", "c", "d"],  # 70% a
            ["a", "b", "c", "d", "a", "b", "c", "d", "a", "b"],  # 30% a
        ]
        selector = DistributionAwareSelector(target_distribution=skewed_target)
        target_units = set(skewed_target.keys())
        result = selector.select(cands, phonemes, target_units, max_sentences=1)
        # s0 should be preferred as it matches the skewed distribution better
        assert 0 in result.selected_indices

    def test_metadata_includes_kl_divergence(self, uniform_target):
        cands = ["s0"]
        phonemes = [["a", "b", "c", "d"]]
        selector = DistributionAwareSelector(target_distribution=uniform_target)
        result = selector.select(cands, phonemes, set(uniform_target.keys()))
        assert "kl_divergence" in result.metadata

    def test_kl_divergence_zero_for_perfect_match(self):
        """If selected corpus perfectly matches target, KL should be ~0."""
        target = {"a": 0.5, "b": 0.5}
        cands = ["s0"]
        phonemes = [["a", "b"]]
        selector = DistributionAwareSelector(target_distribution=target)
        result = selector.select(cands, phonemes, {"a", "b"})
        assert result.metadata["kl_divergence"] == pytest.approx(0.0, abs=1e-9)

    def test_max_sentences_budget(self, uniform_target):
        cands = ["s0", "s1", "s2"]
        phonemes = [["a", "b"], ["c", "d"], ["a", "c"]]
        selector = DistributionAwareSelector(target_distribution=uniform_target)
        result = selector.select(
            cands, phonemes, set(uniform_target.keys()), max_sentences=1
        )
        assert result.num_selected <= 1

    def test_target_coverage_early_stop(self, uniform_target):
        cands = ["s0", "s1", "s2"]
        phonemes = [["a", "b"], ["c", "d"], ["a", "c"]]
        selector = DistributionAwareSelector(target_distribution=uniform_target)
        result = selector.select(
            cands, phonemes, set(uniform_target.keys()), target_coverage=0.5
        )
        assert result.coverage >= 0.5

    def test_empty_candidates(self, uniform_target):
        selector = DistributionAwareSelector(target_distribution=uniform_target)
        result = selector.select([], [], set(uniform_target.keys()))
        assert result.coverage == 0.0
        assert result.num_selected == 0

    def test_empty_target(self, uniform_target):
        selector = DistributionAwareSelector(target_distribution=uniform_target)
        result = selector.select(["s0"], [["a"]], set())
        assert result.coverage == 1.0
        assert result.num_selected == 0

    def test_impossible_full_coverage(self, uniform_target):
        cands = ["s0"]
        phonemes = [["a", "b"]]
        selector = DistributionAwareSelector(target_distribution=uniform_target)
        result = selector.select(cands, phonemes, set(uniform_target.keys()))
        assert result.missing_units == {"c", "d"}

    def test_distribution_must_be_nonempty(self):
        with pytest.raises(ValueError, match="target_distribution"):
            DistributionAwareSelector(target_distribution={})

    def test_distribution_values_must_be_positive(self):
        with pytest.raises(ValueError, match="positive"):
            DistributionAwareSelector(target_distribution={"a": -0.1, "b": 1.1})

    def test_distribution_auto_normalizes(self):
        """Non-normalized distributions should be accepted and normalized."""
        target = {"a": 2.0, "b": 2.0}
        selector = DistributionAwareSelector(target_distribution=target)
        cands = ["s0"]
        phonemes = [["a", "b"]]
        result = selector.select(cands, phonemes, {"a", "b"})
        # Should work fine; internally treated as 0.5/0.5
        assert result.coverage == 1.0

    def test_diphone_unit(self):
        target = {"a-b": 0.5, "b-c": 0.5}
        cands = ["s0"]
        phonemes = [["a", "b", "c"]]
        selector = DistributionAwareSelector(
            target_distribution=target, unit="diphone"
        )
        result = selector.select(cands, phonemes, {"a-b", "b-c"})
        assert result.coverage == 1.0
        assert result.unit == "diphone"

    def test_indices_and_sentences_match(self, uniform_target):
        cands = ["s0", "s1", "s2"]
        phonemes = [["a", "b"], ["c", "d"], ["a", "c"]]
        selector = DistributionAwareSelector(target_distribution=uniform_target)
        result = selector.select(cands, phonemes, set(uniform_target.keys()))
        for idx, sent in zip(result.selected_indices, result.selected_sentences):
            assert sent == cands[idx]
