"""Tests for the weights module: phoneme weighting strategies."""

from __future__ import annotations

import pytest

from corpusgen.weights import uniform_weights, frequency_inverse_weights, linguistic_class_weights


class TestUniformWeights:
    """All units get equal weight of 1.0."""

    def test_basic(self):
        units = {"a", "b", "c"}
        w = uniform_weights(units)
        assert w == {"a": 1.0, "b": 1.0, "c": 1.0}

    def test_empty(self):
        assert uniform_weights(set()) == {}

    def test_single(self):
        assert uniform_weights({"x"}) == {"x": 1.0}


class TestFrequencyInverseWeights:
    """Rare units get higher weights based on corpus frequencies."""

    def test_basic(self):
        """Less frequent phonemes get higher weight."""
        corpus_phonemes = [
            ["a", "a", "a", "b"],  # a appears 3x, b appears 1x
            ["a", "c"],             # c appears 1x
        ]
        target = {"a", "b", "c"}
        w = frequency_inverse_weights(target, corpus_phonemes)
        assert w["b"] > w["a"]  # b is rarer → higher weight
        assert w["c"] > w["a"]  # c is rarer → higher weight
        assert w["b"] == pytest.approx(w["c"])  # same frequency → same weight

    def test_all_values_positive(self):
        corpus_phonemes = [["a", "b", "c"]]
        target = {"a", "b", "c"}
        w = frequency_inverse_weights(target, corpus_phonemes)
        assert all(v > 0 for v in w.values())

    def test_unseen_unit_gets_max_weight(self):
        """Units not in corpus at all should get the highest weight."""
        corpus_phonemes = [["a", "a", "a"]]
        target = {"a", "x"}
        w = frequency_inverse_weights(target, corpus_phonemes)
        assert w["x"] > w["a"]

    def test_empty_corpus(self):
        """All units get equal weight when corpus is empty."""
        target = {"a", "b"}
        w = frequency_inverse_weights(target, [])
        assert w["a"] == w["b"]
        assert all(v > 0 for v in w.values())

    def test_empty_target(self):
        assert frequency_inverse_weights(set(), [["a"]]) == {}

    def test_weights_are_normalized(self):
        """Weights should sum to len(target) (average weight = 1.0)."""
        corpus_phonemes = [["a", "a", "b"], ["b", "c"]]
        target = {"a", "b", "c"}
        w = frequency_inverse_weights(target, corpus_phonemes)
        assert sum(w.values()) == pytest.approx(len(target), rel=1e-6)


class TestLinguisticClassWeights:
    """Weight by phoneme class: vowels, consonants, etc."""

    def test_basic(self):
        target = {"a", "p", "s"}
        w = linguistic_class_weights(target)
        assert set(w.keys()) == target
        assert all(v > 0 for v in w.values())

    def test_empty(self):
        assert linguistic_class_weights(set()) == {}

    def test_custom_class_weights(self):
        """Users can provide custom weights per phoneme class."""
        target = {"a", "p"}
        custom = {"vowel": 2.0, "consonant": 1.0}
        w = linguistic_class_weights(target, class_weights=custom)
        # 'a' is a vowel, 'p' is a consonant
        assert w["a"] > w["p"]

    def test_unknown_phoneme_gets_default_weight(self):
        """Phonemes not classifiable get weight 1.0."""
        target = {"ʘ"}  # A click — may not be classified
        w = linguistic_class_weights(target)
        assert "ʘ" in w
        assert w["ʘ"] > 0

    def test_all_values_positive(self):
        target = {"a", "e", "i", "p", "t", "k", "s", "ʃ"}
        w = linguistic_class_weights(target)
        assert all(v > 0 for v in w.values())
