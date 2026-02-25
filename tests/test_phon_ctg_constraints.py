"""Tests for Phonotactic Constraint Module — validates phoneme sequence legality."""

import pytest

from corpusgen.generate.phon_ctg.constraints import (
    PhonotacticConstraint,
    NgramPhonotacticModel,
)


# ---------------------------------------------------------------------------
# ABC / Protocol
# ---------------------------------------------------------------------------


class TestPhonotacticConstraintABC:
    """PhonotacticConstraint defines the interface all implementations follow."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            PhonotacticConstraint()

    def test_subclass_must_implement_score(self):
        class Incomplete(PhonotacticConstraint):
            def fit(self, phoneme_sequences):
                pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_subclass_must_implement_fit(self):
        class Incomplete(PhonotacticConstraint):
            def score(self, phonemes):
                return 1.0

        with pytest.raises(TypeError):
            Incomplete()

    def test_valid_subclass(self):
        class Valid(PhonotacticConstraint):
            def score(self, phonemes):
                return 1.0

            def fit(self, phoneme_sequences):
                pass

        obj = Valid()
        assert obj.score(["p", "b"]) == 1.0

    def test_callable_protocol(self):
        """PhonotacticConstraint.score should be usable as PhoneticScorer's hook."""
        class Valid(PhonotacticConstraint):
            def score(self, phonemes):
                return 0.75

            def fit(self, phoneme_sequences):
                pass

        obj = Valid()
        # The scorer expects Callable[[list[str]], float]
        result = obj.score(["p", "b", "t"])
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# NgramPhonotacticModel: construction
# ---------------------------------------------------------------------------


class TestNgramConstruction:
    """NgramPhonotacticModel creation and configuration."""

    def test_default_order(self):
        model = NgramPhonotacticModel()
        assert model.order == 2  # bigram by default

    def test_custom_order(self):
        model = NgramPhonotacticModel(order=3)
        assert model.order == 3

    def test_invalid_order_zero(self):
        with pytest.raises(ValueError, match="[Oo]rder"):
            NgramPhonotacticModel(order=0)

    def test_invalid_order_negative(self):
        with pytest.raises(ValueError, match="[Oo]rder"):
            NgramPhonotacticModel(order=-1)

    def test_is_phonotactic_constraint(self):
        model = NgramPhonotacticModel()
        assert isinstance(model, PhonotacticConstraint)

    def test_not_fitted_initially(self):
        model = NgramPhonotacticModel()
        assert not model.is_fitted


# ---------------------------------------------------------------------------
# NgramPhonotacticModel: fit with pre-phonemized data
# ---------------------------------------------------------------------------


class TestNgramFitPhonemes:
    """Training the n-gram model from pre-phonemized sequences."""

    def test_fit_with_phoneme_sequences(self):
        model = NgramPhonotacticModel(order=2)
        sequences = [
            ["p", "æ", "t"],       # pat
            ["b", "æ", "t"],       # bat
            ["k", "æ", "t"],       # cat
        ]
        model.fit(sequences)
        assert model.is_fitted

    def test_fit_empty_sequences_raises(self):
        model = NgramPhonotacticModel()
        with pytest.raises(ValueError, match="[Ee]mpty"):
            model.fit([])

    def test_fit_filters_empty_inner_sequences(self):
        model = NgramPhonotacticModel()
        sequences = [
            ["p", "æ", "t"],
            [],                    # empty — should be skipped
            ["b", "æ", "t"],
        ]
        model.fit(sequences)
        assert model.is_fitted

    def test_fit_all_empty_raises(self):
        model = NgramPhonotacticModel()
        with pytest.raises(ValueError, match="[Nn]o valid"):
            model.fit([[], []])


# ---------------------------------------------------------------------------
# NgramPhonotacticModel: fit from raw text
# ---------------------------------------------------------------------------


class TestNgramFitText:
    """Training the n-gram model from raw text + G2P."""

    @pytest.fixture
    def english_sentences(self):
        return [
            "The cat sat on the mat.",
            "A big dog ran fast.",
            "She picked the red hat.",
        ]

    def test_fit_from_text(self, english_sentences):
        model = NgramPhonotacticModel(order=2)
        model.fit_from_text(english_sentences, language="en-us")
        assert model.is_fitted

    def test_fit_from_text_empty_raises(self):
        model = NgramPhonotacticModel()
        with pytest.raises(ValueError, match="[Ee]mpty"):
            model.fit_from_text([], language="en-us")

    def test_fit_from_text_default_language(self, english_sentences):
        model = NgramPhonotacticModel()
        model.fit_from_text(english_sentences)  # should default to en-us
        assert model.is_fitted


# ---------------------------------------------------------------------------
# NgramPhonotacticModel: score
# ---------------------------------------------------------------------------


class TestNgramScore:
    """Scoring phoneme sequences for phonotactic legality."""

    @pytest.fixture
    def fitted_model(self):
        model = NgramPhonotacticModel(order=2)
        # Train on simple English-like sequences
        sequences = [
            ["s", "t", "ɹ", "ɪ", "ŋ"],      # string
            ["s", "p", "ɹ", "ɪ", "ŋ"],       # spring
            ["s", "t", "æ", "k"],              # stack
            ["p", "l", "æ", "n"],              # plan
            ["b", "l", "æ", "k"],              # black
            ["k", "l", "æ", "s"],              # class
            ["t", "ɹ", "æ", "p"],              # trap
            ["k", "ɹ", "æ", "b"],              # crab
        ]
        model.fit(sequences)
        return model

    def test_score_returns_float(self, fitted_model):
        result = fitted_model.score(["s", "t", "æ", "k"])
        assert isinstance(result, float)

    def test_score_range_zero_to_one(self, fitted_model):
        result = fitted_model.score(["s", "t", "æ", "k"])
        assert 0.0 <= result <= 1.0

    def test_seen_sequence_scores_higher(self, fitted_model):
        """Sequences similar to training data should score higher."""
        seen_score = fitted_model.score(["s", "t", "ɹ", "ɪ", "ŋ"])
        # A sequence with unusual bigrams should score lower
        unusual_score = fitted_model.score(["ŋ", "ŋ", "ŋ", "ŋ"])
        assert seen_score > unusual_score

    def test_score_not_fitted_raises(self):
        model = NgramPhonotacticModel()
        with pytest.raises(RuntimeError, match="[Nn]ot fitted"):
            model.score(["p", "æ", "t"])

    def test_score_empty_phonemes(self, fitted_model):
        result = fitted_model.score([])
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_score_single_phoneme(self, fitted_model):
        """Single phoneme — no bigrams possible, should handle gracefully."""
        result = fitted_model.score(["p"])
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_score_with_unseen_phonemes(self, fitted_model):
        """Phonemes not in training data should get low but valid score."""
        result = fitted_model.score(["ʔ", "ɣ", "ʕ"])
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# NgramPhonotacticModel: trigram order
# ---------------------------------------------------------------------------


class TestNgramTrigram:
    """N-gram model with order=3 (trigram)."""

    def test_trigram_fit_and_score(self):
        model = NgramPhonotacticModel(order=3)
        sequences = [
            ["s", "t", "ɹ", "ɪ", "ŋ"],
            ["s", "p", "ɹ", "ɪ", "ŋ"],
            ["p", "l", "æ", "n"],
        ]
        model.fit(sequences)
        result = model.score(["s", "t", "ɹ", "ɪ", "ŋ"])
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# NgramPhonotacticModel: smoothing
# ---------------------------------------------------------------------------


class TestNgramSmoothing:
    """Smoothing prevents zero probabilities for unseen n-grams."""

    def test_unseen_ngram_does_not_give_zero(self):
        model = NgramPhonotacticModel(order=2)
        sequences = [
            ["p", "æ", "t"],
            ["b", "æ", "t"],
        ]
        model.fit(sequences)
        # "z-ʒ" is an unseen bigram — should not be 0.0 due to smoothing
        result = model.score(["z", "ʒ"])
        assert result > 0.0

    def test_custom_smoothing(self):
        model = NgramPhonotacticModel(order=2, smoothing=0.1)
        sequences = [["p", "æ", "t"]]
        model.fit(sequences)
        result = model.score(["z", "ʒ"])
        assert result > 0.0


# ---------------------------------------------------------------------------
# NgramPhonotacticModel: vocabulary
# ---------------------------------------------------------------------------


class TestNgramVocabulary:
    """Vocabulary tracking from training data."""

    def test_vocabulary_after_fit(self):
        model = NgramPhonotacticModel()
        sequences = [
            ["p", "æ", "t"],
            ["b", "æ", "d"],
        ]
        model.fit(sequences)
        vocab = model.vocabulary
        assert isinstance(vocab, set)
        assert "p" in vocab
        assert "æ" in vocab
        assert "t" in vocab
        assert "b" in vocab
        assert "d" in vocab

    def test_vocabulary_before_fit(self):
        model = NgramPhonotacticModel()
        assert model.vocabulary == set()


# ---------------------------------------------------------------------------
# Integration: scorer hook compatibility
# ---------------------------------------------------------------------------


class TestScorerIntegration:
    """NgramPhonotacticModel works as a PhoneticScorer phonotactic_scorer hook."""

    def test_as_callable_hook(self):
        """The .score method has the right signature for PhoneticScorer."""
        model = NgramPhonotacticModel()
        model.fit([["p", "æ", "t"], ["b", "æ", "t"]])

        # Simulates what PhoneticScorer does internally
        phonemes = ["p", "æ", "t"]
        result = model.score(phonemes)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestNgramSerialization:
    """Save and load fitted models."""

    def test_to_dict_and_from_dict(self):
        model = NgramPhonotacticModel(order=2, smoothing=0.05)
        model.fit([["p", "æ", "t"], ["b", "æ", "t"]])

        data = model.to_dict()
        assert isinstance(data, dict)

        restored = NgramPhonotacticModel.from_dict(data)
        assert restored.order == 2
        assert restored.is_fitted

        # Scores should be identical
        phonemes = ["p", "æ", "t"]
        assert restored.score(phonemes) == pytest.approx(model.score(phonemes))

    def test_to_dict_not_fitted(self):
        model = NgramPhonotacticModel()
        data = model.to_dict()
        restored = NgramPhonotacticModel.from_dict(data)
        assert not restored.is_fitted
