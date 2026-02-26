"""Tests for Phon-DATG helper functions: _extract_units and _filter_by_level.

These are the foundation of the AttributeWordIndex. _extract_units determines
what phonetic units (phonemes, diphones, triphones) are indexed for each token.
_filter_by_level controls which units are considered during anti-attribute
computation. Correctness here directly impacts coverage metrics.
"""

import pytest

from corpusgen.generate.phon_datg.attribute_words import (
    _extract_units,
    _filter_by_level,
)


# ===========================================================================
# _extract_units
# ===========================================================================


class TestExtractUnitsPhonemes:
    """Phoneme-level extraction (individual symbols)."""

    def test_single_phoneme(self):
        result = _extract_units(["k"])
        assert "k" in result

    def test_two_phonemes(self):
        result = _extract_units(["k", "æ"])
        assert "k" in result
        assert "æ" in result

    def test_three_phonemes(self):
        result = _extract_units(["k", "æ", "t"])
        assert {"k", "æ", "t"} <= result

    def test_duplicate_phonemes_deduplicated(self):
        """Repeated phonemes appear only once in the set."""
        result = _extract_units(["k", "æ", "k"])
        assert "k" in result
        # Set semantics — no duplicates by definition


class TestExtractUnitsDiphones:
    """Diphone extraction (adjacent pairs)."""

    def test_two_phonemes_produces_one_diphone(self):
        result = _extract_units(["k", "æ"])
        assert "k-æ" in result

    def test_three_phonemes_produces_two_diphones(self):
        result = _extract_units(["k", "æ", "t"])
        assert "k-æ" in result
        assert "æ-t" in result

    def test_four_phonemes_produces_three_diphones(self):
        result = _extract_units(["s", "k", "w", "ɛ"])
        assert "s-k" in result
        assert "k-w" in result
        assert "w-ɛ" in result

    def test_single_phoneme_no_diphones(self):
        result = _extract_units(["k"])
        diphones = {u for u in result if u.count("-") == 1}
        assert diphones == set()

    def test_diphone_order_is_sequential(self):
        """Diphones reflect left-to-right phoneme order."""
        result = _extract_units(["t", "k"])
        assert "t-k" in result
        assert "k-t" not in result


class TestExtractUnitsTriphones:
    """Triphone extraction (adjacent triples)."""

    def test_three_phonemes_produces_one_triphone(self):
        result = _extract_units(["k", "æ", "t"])
        assert "k-æ-t" in result

    def test_four_phonemes_produces_two_triphones(self):
        result = _extract_units(["s", "k", "æ", "t"])
        assert "s-k-æ" in result
        assert "k-æ-t" in result

    def test_two_phonemes_no_triphones(self):
        result = _extract_units(["k", "æ"])
        triphones = {u for u in result if u.count("-") == 2}
        assert triphones == set()

    def test_single_phoneme_no_triphones(self):
        result = _extract_units(["k"])
        triphones = {u for u in result if u.count("-") == 2}
        assert triphones == set()

    def test_triphone_order_is_sequential(self):
        result = _extract_units(["k", "æ", "t"])
        assert "k-æ-t" in result
        assert "t-æ-k" not in result


class TestExtractUnitsCompleteness:
    """Total unit counts match expected combinatorics."""

    def test_empty_input(self):
        result = _extract_units([])
        assert result == set()

    def test_one_phoneme_total_count(self):
        """1 phoneme → 1 phoneme, 0 diphones, 0 triphones = 1."""
        result = _extract_units(["k"])
        assert len(result) == 1

    def test_two_phonemes_total_count(self):
        """2 phonemes → 2 phonemes, 1 diphone, 0 triphones = 3."""
        result = _extract_units(["k", "æ"])
        assert len(result) == 3

    def test_three_unique_phonemes_total_count(self):
        """3 unique phonemes → 3 + 2 + 1 = 6."""
        result = _extract_units(["k", "æ", "t"])
        assert len(result) == 6

    def test_four_unique_phonemes_total_count(self):
        """4 unique phonemes → 4 + 3 + 2 = 9."""
        result = _extract_units(["s", "k", "æ", "t"])
        assert len(result) == 9

    def test_repeated_phonemes_reduce_count(self):
        """Repeated phonemes produce fewer unique units.

        [k, æ, k] → phonemes: {k, æ} (2)
                     diphones: {k-æ, æ-k} (2)
                     triphones: {k-æ-k} (1)
                     total: 5
        """
        result = _extract_units(["k", "æ", "k"])
        assert len(result) == 5

    def test_all_same_phoneme(self):
        """[k, k, k] → phonemes: {k}, diphones: {k-k}, triphones: {k-k-k} = 3."""
        result = _extract_units(["k", "k", "k"])
        assert result == {"k", "k-k", "k-k-k"}


class TestExtractUnitsIPA:
    """Handles real IPA symbols, including multi-character phonemes."""

    def test_ipa_vowels(self):
        result = _extract_units(["ɪ", "ə", "ɛ"])
        assert {"ɪ", "ə", "ɛ"} <= result

    def test_ipa_affricates(self):
        """Multi-char IPA symbols like tʃ treated as single phonemes."""
        result = _extract_units(["tʃ", "ɪ", "p"])
        assert "tʃ" in result
        assert "tʃ-ɪ" in result
        assert "tʃ-ɪ-p" in result

    def test_long_vowels(self):
        result = _extract_units(["iː", "p"])
        assert "iː" in result
        assert "iː-p" in result


# ===========================================================================
# _filter_by_level
# ===========================================================================


class TestFilterByLevelNone:
    """unit_level=None returns all units unfiltered."""

    def test_returns_all_units(self):
        units = {"k", "æ", "k-æ", "k-æ-t"}
        result = _filter_by_level(units, None)
        assert result == units

    def test_empty_set(self):
        result = _filter_by_level(set(), None)
        assert result == set()


class TestFilterByLevelPhoneme:
    """unit_level='phoneme' keeps only units without hyphens."""

    def test_keeps_phonemes(self):
        units = {"k", "æ", "t", "k-æ", "æ-t", "k-æ-t"}
        result = _filter_by_level(units, "phoneme")
        assert result == {"k", "æ", "t"}

    def test_excludes_diphones(self):
        units = {"k-æ", "æ-t"}
        result = _filter_by_level(units, "phoneme")
        assert result == set()

    def test_excludes_triphones(self):
        units = {"k-æ-t"}
        result = _filter_by_level(units, "phoneme")
        assert result == set()

    def test_empty_set(self):
        result = _filter_by_level(set(), "phoneme")
        assert result == set()

    def test_phonemes_only_input(self):
        units = {"k", "æ", "t"}
        result = _filter_by_level(units, "phoneme")
        assert result == units


class TestFilterByLevelDiphone:
    """unit_level='diphone' keeps only units with exactly one hyphen."""

    def test_keeps_diphones(self):
        units = {"k", "æ", "k-æ", "æ-t", "k-æ-t"}
        result = _filter_by_level(units, "diphone")
        assert result == {"k-æ", "æ-t"}

    def test_excludes_phonemes(self):
        units = {"k", "æ"}
        result = _filter_by_level(units, "diphone")
        assert result == set()

    def test_excludes_triphones(self):
        units = {"k-æ-t"}
        result = _filter_by_level(units, "diphone")
        assert result == set()

    def test_empty_set(self):
        result = _filter_by_level(set(), "diphone")
        assert result == set()


class TestFilterByLevelTriphone:
    """unit_level='triphone' keeps only units with exactly two hyphens."""

    def test_keeps_triphones(self):
        units = {"k", "k-æ", "k-æ-t", "s-k-æ"}
        result = _filter_by_level(units, "triphone")
        assert result == {"k-æ-t", "s-k-æ"}

    def test_excludes_phonemes_and_diphones(self):
        units = {"k", "æ", "k-æ"}
        result = _filter_by_level(units, "triphone")
        assert result == set()

    def test_empty_set(self):
        result = _filter_by_level(set(), "triphone")
        assert result == set()


class TestFilterByLevelUnknown:
    """Unknown unit_level returns all units (no filtering)."""

    def test_unknown_level_returns_all(self):
        units = {"k", "k-æ", "k-æ-t"}
        result = _filter_by_level(units, "quadphone")
        assert result == units


class TestFilterByLevelEdgeCases:
    """Edge cases for the hyphen-counting heuristic."""

    def test_multi_char_phoneme_no_hyphen(self):
        """Multi-character IPA symbols like 'tʃ' have no hyphen → phoneme."""
        units = {"tʃ", "tʃ-ɪ"}
        result = _filter_by_level(units, "phoneme")
        assert result == {"tʃ"}

    def test_multi_char_in_diphone(self):
        """Diphone of multi-char phonemes still has exactly one hyphen."""
        units = {"tʃ-ɪ", "tʃ"}
        result = _filter_by_level(units, "diphone")
        assert result == {"tʃ-ɪ"}

    def test_multi_char_in_triphone(self):
        units = {"tʃ-ɪ-p", "tʃ-ɪ"}
        result = _filter_by_level(units, "triphone")
        assert result == {"tʃ-ɪ-p"}
