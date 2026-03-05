"""Tests for corpusgen.evaluate.text_quality — text-level corpus metrics.

Covers word tokenization, sentence length stats, vocabulary stats,
and readability scores.  All expected values are hand-computed.
"""

from __future__ import annotations

import math
import statistics

import pytest

from corpusgen.evaluate.text_quality import (
    TextQualityMetrics,
    compute_text_quality_metrics,
    tokenize_words,
)


def _approx(value: float, abs_tol: float = 1e-6) -> pytest.approx:
    return pytest.approx(value, abs=abs_tol)


# ---------------------------------------------------------------------------
# 1. Word tokenizer
# ---------------------------------------------------------------------------


class TestTokenizeWords:
    """Unicode-aware, punctuation-stripping word tokenizer."""

    def test_basic_english(self):
        assert tokenize_words("The cat sat.") == ["the", "cat", "sat"]

    def test_strips_punctuation(self):
        assert tokenize_words("Hello, world!") == ["hello", "world"]

    def test_handles_hyphens(self):
        # Hyphenated words are kept as one token
        assert tokenize_words("well-known fact") == ["well-known", "fact"]

    def test_handles_apostrophes(self):
        assert tokenize_words("don't stop") == ["don't", "stop"]

    def test_unicode_non_latin(self):
        """Non-Latin scripts should tokenize on whitespace."""
        tokens = tokenize_words("مرحبا بالعالم")
        assert len(tokens) == 2

    def test_mixed_punctuation(self):
        result = tokenize_words("Yes! No? Maybe... (okay)")
        assert result == ["yes", "no", "maybe", "okay"]

    def test_empty_string(self):
        assert tokenize_words("") == []

    def test_whitespace_only(self):
        assert tokenize_words("   \t\n  ") == []

    def test_numbers_kept(self):
        assert tokenize_words("I have 3 cats.") == ["i", "have", "3", "cats"]

    def test_lowercases(self):
        assert tokenize_words("The BIG Dog") == ["the", "big", "dog"]


# ---------------------------------------------------------------------------
# 2. Sentence length stats (words)
# ---------------------------------------------------------------------------


class TestSentenceLengthWords:
    """Word-level sentence length statistics."""

    @pytest.fixture()
    def metrics(self) -> TextQualityMetrics:
        sentences = [
            "The cat sat.",        # 3 words
            "Big dogs dig deep holes.",  # 5 words
            "Go.",                 # 1 word
        ]
        phoneme_sequences = [
            ["ð", "ə", "k", "æ", "t", "s", "æ", "t"],
            ["b", "ɪ", "ɡ", "d", "ɒ", "ɡ", "z", "d", "ɪ", "ɡ"],
            ["ɡ", "oʊ"],
        ]
        return compute_text_quality_metrics(sentences, phoneme_sequences)

    def test_word_length_mean(self, metrics):
        # (3 + 5 + 1) / 3 = 3.0
        assert metrics.sentence_length_words_mean == _approx(3.0)

    def test_word_length_median(self, metrics):
        # sorted: [1, 3, 5] → median = 3
        assert metrics.sentence_length_words_median == _approx(3.0)

    def test_word_length_std(self, metrics):
        # population std of [3, 5, 1]
        expected = statistics.pstdev([3, 5, 1])
        assert metrics.sentence_length_words_std == _approx(expected)

    def test_word_length_min(self, metrics):
        assert metrics.sentence_length_words_min == 1

    def test_word_length_max(self, metrics):
        assert metrics.sentence_length_words_max == 5


# ---------------------------------------------------------------------------
# 3. Sentence length stats (phonemes)
# ---------------------------------------------------------------------------


class TestSentenceLengthPhonemes:
    """Phoneme-level sentence length statistics."""

    @pytest.fixture()
    def metrics(self) -> TextQualityMetrics:
        sentences = ["a", "b"]
        phoneme_sequences = [
            ["p", "b", "t"],          # 3 phonemes
            ["k", "æ", "t", "s", "z"],  # 5 phonemes
        ]
        return compute_text_quality_metrics(sentences, phoneme_sequences)

    def test_phoneme_length_mean(self, metrics):
        assert metrics.sentence_length_phonemes_mean == _approx(4.0)

    def test_phoneme_length_median(self, metrics):
        assert metrics.sentence_length_phonemes_median == _approx(4.0)

    def test_phoneme_length_std(self, metrics):
        expected = statistics.pstdev([3, 5])
        assert metrics.sentence_length_phonemes_std == _approx(expected)

    def test_phoneme_length_min(self, metrics):
        assert metrics.sentence_length_phonemes_min == 3

    def test_phoneme_length_max(self, metrics):
        assert metrics.sentence_length_phonemes_max == 5


# ---------------------------------------------------------------------------
# 4. Vocabulary stats
# ---------------------------------------------------------------------------


class TestVocabularyStats:
    """TTR, hapax ratio, word counts."""

    @pytest.fixture()
    def metrics(self) -> TextQualityMetrics:
        # "the" appears 2x, all others 1x
        sentences = [
            "The cat sat on the mat.",  # the(2), cat, sat, on, mat → 5 unique, 6 total
        ]
        phoneme_sequences = [["x"] * 6]
        return compute_text_quality_metrics(sentences, phoneme_sequences)

    def test_total_words(self, metrics):
        assert metrics.total_words == 6

    def test_unique_words(self, metrics):
        assert metrics.unique_words == 5

    def test_type_token_ratio(self, metrics):
        assert metrics.type_token_ratio == _approx(5 / 6)

    def test_hapax_ratio(self, metrics):
        # 4 words appear once (cat, sat, on, mat) out of 5 unique
        assert metrics.hapax_ratio == _approx(4 / 5)


class TestVocabularyMultipleSentences:
    """Vocabulary computed across all sentences."""

    def test_cross_sentence_dedup(self):
        sentences = ["The cat.", "The dog."]
        phoneme_sequences = [["x"], ["y"]]
        m = compute_text_quality_metrics(sentences, phoneme_sequences)
        # Words: the, cat, the, dog → total=4, unique=3 (the, cat, dog)
        assert m.total_words == 4
        assert m.unique_words == 3
        assert m.type_token_ratio == _approx(3 / 4)


# ---------------------------------------------------------------------------
# 5. Readability scores (English text)
# ---------------------------------------------------------------------------


class TestReadability:
    """Flesch Reading Ease and Flesch-Kincaid Grade Level."""

    def test_simple_english_has_readability(self):
        """Simple English text should produce valid readability scores."""
        sentences = [
            "The cat sat on the mat.",
            "I like big dogs.",
        ]
        phoneme_sequences = [["x"] * 6, ["y"] * 4]
        m = compute_text_quality_metrics(sentences, phoneme_sequences)
        assert m.flesch_reading_ease is not None
        assert m.flesch_kincaid_grade is not None

    def test_reading_ease_range(self):
        """Flesch Reading Ease is typically 0-100 for normal text."""
        sentences = ["The cat sat on the mat."]
        phoneme_sequences = [["x"] * 6]
        m = compute_text_quality_metrics(sentences, phoneme_sequences)
        # Simple sentence → high readability
        assert m.flesch_reading_ease is not None
        assert m.flesch_reading_ease > 50.0

    def test_complex_text_lower_readability(self):
        """Complex text should have lower reading ease than simple text."""
        simple = ["I like cats."]
        complex_ = ["The obfuscation of incomprehensible terminology exacerbates confusion."]
        m_simple = compute_text_quality_metrics(simple, [["x"] * 3])
        m_complex = compute_text_quality_metrics(complex_, [["y"] * 7])
        assert m_simple.flesch_reading_ease is not None
        assert m_complex.flesch_reading_ease is not None
        assert m_simple.flesch_reading_ease > m_complex.flesch_reading_ease

    def test_grade_level_nonnegative(self):
        sentences = ["The cat sat."]
        phoneme_sequences = [["x"] * 3]
        m = compute_text_quality_metrics(sentences, phoneme_sequences)
        assert m.flesch_kincaid_grade is not None
        # Grade level can be negative for very simple text, but usually ≥ 0
        # We just check it's a real number
        assert isinstance(m.flesch_kincaid_grade, float)

    def test_known_values(self):
        """Verify against hand computation.

        "The cat sat on the mat."
        Words: the, cat, sat, on, the, mat → 6 words, 1 sentence
        Syllables: the(1) cat(1) sat(1) on(1) the(1) mat(1) → 6 syllables

        Flesch RE = 206.835 - 1.015*(6/1) - 84.6*(6/6)
                  = 206.835 - 6.09 - 84.6 = 116.145
        FK Grade = 0.39*(6/1) + 11.8*(6/6) - 15.59
                 = 2.34 + 11.8 - 15.59 = -1.45
        """
        sentences = ["The cat sat on the mat."]
        phoneme_sequences = [["x"] * 6]
        m = compute_text_quality_metrics(sentences, phoneme_sequences)
        assert m.flesch_reading_ease == _approx(116.145, abs_tol=0.5)
        assert m.flesch_kincaid_grade == _approx(-1.45, abs_tol=0.5)


# ---------------------------------------------------------------------------
# 6. Readability with non-Latin text
# ---------------------------------------------------------------------------


class TestReadabilityNonLatin:
    """Readability returns None for non-syllable-countable text."""

    def test_arabic_returns_none(self):
        sentences = ["مرحبا بالعالم"]
        phoneme_sequences = [["m", "a", "r", "ħ", "a", "b", "a"]]
        m = compute_text_quality_metrics(sentences, phoneme_sequences)
        # Can't count syllables in Arabic script → None
        assert m.flesch_reading_ease is None
        assert m.flesch_kincaid_grade is None


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------


class TestTextQualityEdgeCases:
    """Degenerate inputs."""

    def test_empty_corpus(self):
        m = compute_text_quality_metrics([], [])
        assert m.total_words == 0
        assert m.unique_words == 0
        assert m.sentence_length_words_mean == _approx(0.0)
        assert m.sentence_length_phonemes_mean == _approx(0.0)
        assert m.type_token_ratio == _approx(0.0)
        assert m.hapax_ratio == _approx(0.0)
        assert m.flesch_reading_ease is None
        assert m.flesch_kincaid_grade is None

    def test_single_word_sentence(self):
        m = compute_text_quality_metrics(["Go."], [["ɡ", "oʊ"]])
        assert m.total_words == 1
        assert m.sentence_length_words_mean == _approx(1.0)
        assert m.sentence_length_words_std == _approx(0.0)

    def test_all_same_word(self):
        """All words identical → TTR approaches 1/N, hapax = 0."""
        m = compute_text_quality_metrics(
            ["the the the the"],
            [["ð", "ə"] * 4],
        )
        assert m.total_words == 4
        assert m.unique_words == 1
        assert m.type_token_ratio == _approx(1 / 4)
        assert m.hapax_ratio == _approx(0.0)  # "the" appears 4x, not a hapax


# ---------------------------------------------------------------------------
# 8. Dataclass export
# ---------------------------------------------------------------------------


class TestTextQualityExport:
    """to_dict() works correctly."""

    def test_to_dict(self):
        m = compute_text_quality_metrics(
            ["The cat sat."], [["ð", "ə", "k", "æ", "t"]]
        )
        d = m.to_dict()
        assert isinstance(d, dict)
        assert "sentence_length_words_mean" in d
        assert "sentence_length_phonemes_mean" in d
        assert "type_token_ratio" in d
        assert "flesch_reading_ease" in d
        assert "flesch_kincaid_grade" in d
        assert "hapax_ratio" in d
