"""Tests for corpusgen.evaluate.error_rates — WER, CER, PER, SER.

All expected values are hand-computed from the standard edit distance
formula: ErrorRate = (S + D + I) / N  where N = len(reference).
"""

from __future__ import annotations

import pytest

from corpusgen.evaluate.error_rates import (
    ErrorRateResult,
    SentenceErrorDetail,
    compute_error_rates,
    edit_distance,
    word_error_rate,
    character_error_rate,
    phoneme_error_rate,
    sentence_error_rate,
)


def _approx(value: float, abs_tol: float = 1e-9):
    return pytest.approx(value, abs=abs_tol)


# ---------------------------------------------------------------------------
# 1. Edit distance (Levenshtein)
# ---------------------------------------------------------------------------


class TestEditDistance:
    """Core Levenshtein edit distance."""

    def test_identical(self):
        assert edit_distance(["a", "b", "c"], ["a", "b", "c"]) == 0

    def test_single_substitution(self):
        assert edit_distance(["a", "b", "c"], ["a", "x", "c"]) == 1

    def test_single_deletion(self):
        # ref=[a,b,c], hyp=[a,c] → delete b → 1
        assert edit_distance(["a", "b", "c"], ["a", "c"]) == 1

    def test_single_insertion(self):
        # ref=[a,c], hyp=[a,b,c] → insert b → 1
        assert edit_distance(["a", "c"], ["a", "b", "c"]) == 1

    def test_empty_reference(self):
        assert edit_distance([], ["a", "b"]) == 2

    def test_empty_hypothesis(self):
        assert edit_distance(["a", "b"], []) == 2

    def test_both_empty(self):
        assert edit_distance([], []) == 0

    def test_completely_different(self):
        assert edit_distance(["a", "b"], ["c", "d"]) == 2

    def test_strings_as_sequences(self):
        """Should work with strings (character sequences)."""
        assert edit_distance("kitten", "sitting") == 3


# ---------------------------------------------------------------------------
# 2. Word Error Rate
# ---------------------------------------------------------------------------


class TestWordErrorRate:
    """WER = (S + D + I) / N at word level."""

    def test_perfect_match(self):
        assert word_error_rate("the cat sat", "the cat sat") == _approx(0.0)

    def test_one_substitution(self):
        # ref=3 words, 1 sub → 1/3
        assert word_error_rate("the cat sat", "the dog sat") == _approx(1 / 3)

    def test_one_deletion(self):
        # ref=3 words, hyp=2 words, 1 del → 1/3
        assert word_error_rate("the cat sat", "the sat") == _approx(1 / 3)

    def test_one_insertion(self):
        # ref=3 words, hyp=4 words, 1 ins → 1/3
        assert word_error_rate("the cat sat", "the big cat sat") == _approx(1 / 3)

    def test_all_wrong(self):
        # ref=2 words, hyp=2 words, 2 subs → 2/2 = 1.0
        assert word_error_rate("hello world", "foo bar") == _approx(1.0)

    def test_wer_can_exceed_one(self):
        """WER > 1.0 is possible when insertions dominate."""
        # ref=1 word, hyp=3 words → 1 sub + 2 ins = 3, WER = 3/1 = 3.0
        result = word_error_rate("hello", "foo bar baz")
        assert result > 1.0

    def test_empty_reference(self):
        """Empty reference → WER = 0.0 if hyp also empty, else undefined.

        Convention: return float('inf') for non-empty hyp, 0.0 for empty hyp.
        """
        assert word_error_rate("", "") == _approx(0.0)
        assert word_error_rate("", "hello") == float("inf")

    def test_case_sensitive(self):
        """WER should be case-insensitive by default."""
        assert word_error_rate("The Cat", "the cat") == _approx(0.0)


# ---------------------------------------------------------------------------
# 3. Character Error Rate
# ---------------------------------------------------------------------------


class TestCharacterErrorRate:
    """CER = edit_distance(ref_chars, hyp_chars) / len(ref_chars)."""

    def test_perfect_match(self):
        assert character_error_rate("hello", "hello") == _approx(0.0)

    def test_one_char_sub(self):
        # "cat" vs "car" → 1 sub / 3 chars
        assert character_error_rate("cat", "car") == _approx(1 / 3)

    def test_empty_reference(self):
        assert character_error_rate("", "") == _approx(0.0)
        assert character_error_rate("", "a") == float("inf")


# ---------------------------------------------------------------------------
# 4. Phoneme Error Rate
# ---------------------------------------------------------------------------


class TestPhonemeErrorRate:
    """PER = edit_distance(ref_phonemes, hyp_phonemes) / len(ref_phonemes)."""

    def test_perfect_match(self):
        ref = ["k", "æ", "t"]
        hyp = ["k", "æ", "t"]
        assert phoneme_error_rate(ref, hyp) == _approx(0.0)

    def test_one_substitution(self):
        ref = ["k", "æ", "t"]
        hyp = ["k", "ɑ", "t"]
        assert phoneme_error_rate(ref, hyp) == _approx(1 / 3)

    def test_deletion(self):
        ref = ["k", "æ", "t"]
        hyp = ["k", "t"]
        assert phoneme_error_rate(ref, hyp) == _approx(1 / 3)

    def test_insertion(self):
        ref = ["k", "æ", "t"]
        hyp = ["k", "æ", "t", "s"]
        assert phoneme_error_rate(ref, hyp) == _approx(1 / 3)

    def test_empty_reference(self):
        assert phoneme_error_rate([], []) == _approx(0.0)
        assert phoneme_error_rate([], ["k"]) == float("inf")


# ---------------------------------------------------------------------------
# 5. Sentence Error Rate
# ---------------------------------------------------------------------------


class TestSentenceErrorRate:
    """SER = fraction of sentences with any error."""

    def test_all_correct(self):
        refs = ["the cat", "big dog"]
        hyps = ["the cat", "big dog"]
        assert sentence_error_rate(refs, hyps) == _approx(0.0)

    def test_all_wrong(self):
        refs = ["the cat", "big dog"]
        hyps = ["a cat", "small dog"]
        assert sentence_error_rate(refs, hyps) == _approx(1.0)

    def test_half_wrong(self):
        refs = ["the cat", "big dog"]
        hyps = ["the cat", "small dog"]
        assert sentence_error_rate(refs, hyps) == _approx(0.5)

    def test_case_insensitive(self):
        refs = ["The Cat"]
        hyps = ["the cat"]
        assert sentence_error_rate(refs, hyps) == _approx(0.0)

    def test_empty(self):
        assert sentence_error_rate([], []) == _approx(0.0)


# ---------------------------------------------------------------------------
# 6. Corpus-level compute_error_rates
# ---------------------------------------------------------------------------


class TestComputeErrorRates:
    """Aggregate function returning structured ErrorRateResult."""

    @pytest.fixture
    def result(self) -> ErrorRateResult:
        refs = ["the cat sat", "big dogs dig"]
        hyps = ["the cat sat", "big dog dig"]  # 1 sub in sentence 2
        ref_phonemes = [
            ["ð", "ə", "k", "æ", "t", "s", "æ", "t"],
            ["b", "ɪ", "ɡ", "d", "ɒ", "ɡ", "z", "d", "ɪ", "ɡ"],
        ]
        hyp_phonemes = [
            ["ð", "ə", "k", "æ", "t", "s", "æ", "t"],
            ["b", "ɪ", "ɡ", "d", "ɒ", "ɡ", "d", "ɪ", "ɡ"],  # missing z
        ]
        return compute_error_rates(refs, hyps, ref_phonemes, hyp_phonemes)

    def test_return_type(self, result):
        assert isinstance(result, ErrorRateResult)

    def test_wer(self, result):
        # Sentence 1: 0 errors / 3 words, Sentence 2: 1 error / 3 words
        # Corpus WER = total errors / total ref words = 1 / 6
        assert result.wer == _approx(1 / 6)

    def test_ser(self, result):
        # 1 of 2 sentences has errors
        assert result.ser == _approx(0.5)

    def test_per_sentence_details(self, result):
        assert len(result.details) == 2
        assert result.details[0].wer == _approx(0.0)
        assert result.details[1].wer > 0.0

    def test_cer_present(self, result):
        assert isinstance(result.cer, float)
        assert result.cer >= 0.0

    def test_per_present(self, result):
        assert isinstance(result.per, float)
        assert result.per >= 0.0

    def test_to_dict(self, result):
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "wer" in d
        assert "cer" in d
        assert "per" in d
        assert "ser" in d
        assert "details" in d

    def test_without_phonemes(self):
        """Should work without phoneme lists (PER = None)."""
        refs = ["the cat"]
        hyps = ["a cat"]
        result = compute_error_rates(refs, hyps)
        assert result.wer > 0.0
        assert result.per is None
        assert result.cer >= 0.0


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------


class TestErrorRateEdgeCases:

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            compute_error_rates(["a"], ["b", "c"])

    def test_empty_corpus(self):
        result = compute_error_rates([], [])
        assert result.wer == _approx(0.0)
        assert result.cer == _approx(0.0)
        assert result.ser == _approx(0.0)
        assert result.per is None
        assert result.details == []
