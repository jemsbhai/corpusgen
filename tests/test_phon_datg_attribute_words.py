"""Tests for AttributeWordIndex — phonetic unit → token ID mapping."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from corpusgen.generate.phon_datg.attribute_words import AttributeWordIndex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_tokenizer(vocab: dict[str, int]) -> MagicMock:
    """Create a mock tokenizer with a known vocabulary.

    Args:
        vocab: Mapping of token string → token ID.
    """
    tok = MagicMock()
    tok.get_vocab.return_value = dict(vocab)
    # batch_decode: given list of token IDs, return corresponding strings
    id_to_str = {v: k for k, v in vocab.items()}
    tok.batch_decode.side_effect = lambda ids, **kw: [
        id_to_str.get(i, "") for i in ids
    ]
    tok.decode.side_effect = lambda i, **kw: id_to_str.get(i, "")
    return tok


def _mock_g2p_batch(texts: list[str], language: str = "en-us") -> list:
    """Simulated G2P results for a small test vocabulary.

    Maps known words to phoneme lists. Unknown words get empty phonemes.
    """
    word_phonemes = {
        "cat": ["k", "æ", "t"],
        "kit": ["k", "ɪ", "t"],
        "sat": ["s", "æ", "t"],
        "the": ["ð", "ə"],
        "ship": ["ʃ", "ɪ", "p"],
        "thin": ["θ", "ɪ", "n"],
        "square": ["s", "k", "w", "ɛ", "ɹ"],
        "quick": ["k", "w", "ɪ", "k"],
        "a": ["ə"],
        "": [],
    }
    results = []
    for text in texts:
        cleaned = text.strip().lower().lstrip("▁")  # handle sentencepiece prefix
        phonemes = word_phonemes.get(cleaned, [])
        result = MagicMock()
        result.phonemes = phonemes
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """AttributeWordIndex creation."""

    def test_basic_creation(self):
        index = AttributeWordIndex()
        assert isinstance(index, AttributeWordIndex)

    def test_default_language(self):
        index = AttributeWordIndex()
        assert index.language == "en-us"

    def test_custom_language(self):
        index = AttributeWordIndex(language="fr-fr")
        assert index.language == "fr-fr"

    def test_not_built_initially(self):
        index = AttributeWordIndex()
        assert not index.is_built


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


class TestBuild:
    """Building the index from a tokenizer vocabulary."""

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_build_sets_is_built(self, mock_g2p):
        mock_g2p.side_effect = _mock_g2p_batch
        vocab = {"cat": 0, "sat": 1, "the": 2}
        tok = _mock_tokenizer(vocab)

        index = AttributeWordIndex()
        index.build(tok)

        assert index.is_built

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_build_is_idempotent(self, mock_g2p):
        """Second build() call does not re-phonemize."""
        mock_g2p.side_effect = _mock_g2p_batch
        vocab = {"cat": 0}
        tok = _mock_tokenizer(vocab)

        index = AttributeWordIndex()
        index.build(tok)
        index.build(tok)

        # Only called once
        assert mock_g2p.call_count == 1

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_build_populates_unit_to_tokens(self, mock_g2p):
        mock_g2p.side_effect = _mock_g2p_batch
        vocab = {"cat": 0, "sat": 1}
        tok = _mock_tokenizer(vocab)

        index = AttributeWordIndex()
        index.build(tok)

        mapping = index.unit_to_tokens
        assert isinstance(mapping, dict)
        # "cat" has k, æ, t; "sat" has s, æ, t
        assert 0 in mapping.get("k", set())  # cat → k
        assert 1 in mapping.get("s", set())  # sat → s
        assert 0 in mapping.get("æ", set())  # cat → æ
        assert 1 in mapping.get("æ", set())  # sat → æ

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_build_includes_diphones(self, mock_g2p):
        mock_g2p.side_effect = _mock_g2p_batch
        vocab = {"cat": 0}
        tok = _mock_tokenizer(vocab)

        index = AttributeWordIndex()
        index.build(tok)

        mapping = index.unit_to_tokens
        # "cat" → [k, æ, t] → diphones: k-æ, æ-t
        assert 0 in mapping.get("k-æ", set())
        assert 0 in mapping.get("æ-t", set())

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_build_includes_triphones(self, mock_g2p):
        mock_g2p.side_effect = _mock_g2p_batch
        vocab = {"cat": 0}
        tok = _mock_tokenizer(vocab)

        index = AttributeWordIndex()
        index.build(tok)

        mapping = index.unit_to_tokens
        # "cat" → [k, æ, t] → triphone: k-æ-t
        assert 0 in mapping.get("k-æ-t", set())

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_build_skips_empty_phonemes(self, mock_g2p):
        """Tokens that G2P can't phonemize are excluded."""
        mock_g2p.side_effect = _mock_g2p_batch
        vocab = {"cat": 0, "": 1}  # empty token
        tok = _mock_tokenizer(vocab)

        index = AttributeWordIndex()
        index.build(tok)

        mapping = index.unit_to_tokens
        # Token 1 should not appear anywhere
        for token_ids in mapping.values():
            assert 1 not in token_ids

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_build_empty_vocab(self, mock_g2p):
        mock_g2p.side_effect = _mock_g2p_batch
        tok = _mock_tokenizer({})

        index = AttributeWordIndex()
        index.build(tok)

        assert index.is_built
        assert index.unit_to_tokens == {}


# ---------------------------------------------------------------------------
# get_attribute_tokens
# ---------------------------------------------------------------------------


class TestGetAttributeTokens:
    """Look up token IDs for target phonetic units."""

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_returns_tokens_for_target(self, mock_g2p):
        mock_g2p.side_effect = _mock_g2p_batch
        vocab = {"cat": 0, "sat": 1, "ship": 2}
        tok = _mock_tokenizer(vocab)

        index = AttributeWordIndex()
        index.build(tok)

        # Target ʃ — only "ship" contains it
        result = index.get_attribute_tokens(["ʃ"])
        assert 2 in result
        assert 0 not in result
        assert 1 not in result

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_returns_union_for_multiple_targets(self, mock_g2p):
        mock_g2p.side_effect = _mock_g2p_batch
        vocab = {"cat": 0, "sat": 1, "ship": 2}
        tok = _mock_tokenizer(vocab)

        index = AttributeWordIndex()
        index.build(tok)

        # Target k and ʃ — "cat" has k, "ship" has ʃ
        result = index.get_attribute_tokens(["k", "ʃ"])
        assert 0 in result  # cat
        assert 2 in result  # ship

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_returns_empty_for_unknown_unit(self, mock_g2p):
        mock_g2p.side_effect = _mock_g2p_batch
        vocab = {"cat": 0}
        tok = _mock_tokenizer(vocab)

        index = AttributeWordIndex()
        index.build(tok)

        result = index.get_attribute_tokens(["ʒ"])  # not in any word
        assert result == set()

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_empty_targets_returns_empty(self, mock_g2p):
        mock_g2p.side_effect = _mock_g2p_batch
        vocab = {"cat": 0}
        tok = _mock_tokenizer(vocab)

        index = AttributeWordIndex()
        index.build(tok)

        result = index.get_attribute_tokens([])
        assert result == set()

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_matches_diphone_targets(self, mock_g2p):
        mock_g2p.side_effect = _mock_g2p_batch
        vocab = {"cat": 0, "sat": 1}
        tok = _mock_tokenizer(vocab)

        index = AttributeWordIndex()
        index.build(tok)

        # "cat" has diphone k-æ; "sat" does not
        result = index.get_attribute_tokens(["k-æ"])
        assert 0 in result
        assert 1 not in result

    def test_raises_if_not_built(self):
        index = AttributeWordIndex()
        with pytest.raises(RuntimeError, match="not.*built"):
            index.get_attribute_tokens(["k"])


# ---------------------------------------------------------------------------
# get_anti_attribute_tokens (option A: covered set)
# ---------------------------------------------------------------------------


class TestGetAntiAttributeTokens:
    """Tokens whose units are ALL already covered."""

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_returns_tokens_fully_covered(self, mock_g2p):
        mock_g2p.side_effect = _mock_g2p_batch
        vocab = {"cat": 0, "sat": 1, "ship": 2}
        tok = _mock_tokenizer(vocab)

        index = AttributeWordIndex()
        index.build(tok)

        # Mark k, æ, t as covered — "cat" is fully covered
        # "sat" has s which is NOT covered, "ship" has ʃ, ɪ, p not covered
        covered = {"k", "æ", "t", "k-æ", "æ-t", "k-æ-t"}
        result = index.get_anti_attribute_tokens(covered)
        assert 0 in result  # cat — all units covered

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_excludes_partially_covered(self, mock_g2p):
        mock_g2p.side_effect = _mock_g2p_batch
        vocab = {"cat": 0, "sat": 1}
        tok = _mock_tokenizer(vocab)

        index = AttributeWordIndex()
        index.build(tok)

        # Only æ covered — neither cat nor sat are fully covered
        covered = {"æ"}
        result = index.get_anti_attribute_tokens(covered)
        assert 0 not in result
        assert 1 not in result

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_empty_covered_returns_empty(self, mock_g2p):
        mock_g2p.side_effect = _mock_g2p_batch
        vocab = {"cat": 0}
        tok = _mock_tokenizer(vocab)

        index = AttributeWordIndex()
        index.build(tok)

        result = index.get_anti_attribute_tokens(set())
        assert result == set()

    def test_raises_if_not_built(self):
        index = AttributeWordIndex()
        with pytest.raises(RuntimeError, match="not.*built"):
            index.get_anti_attribute_tokens({"k"})


# ---------------------------------------------------------------------------
# get_anti_attribute_tokens_by_frequency (option B: frequency threshold)
# ---------------------------------------------------------------------------


class TestGetAntiAttributeTokensByFrequency:
    """Tokens whose units all exceed a frequency threshold."""

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_returns_tokens_above_threshold(self, mock_g2p):
        mock_g2p.side_effect = _mock_g2p_batch
        vocab = {"cat": 0, "sat": 1, "ship": 2}
        tok = _mock_tokenizer(vocab)

        index = AttributeWordIndex()
        index.build(tok)

        # Suppose k, æ, t all have high counts; s, ʃ, ɪ, p are low
        unit_counts = {
            "k": 100, "æ": 100, "t": 100,
            "k-æ": 80, "æ-t": 80, "k-æ-t": 60,
            "s": 5, "ʃ": 2, "ɪ": 3, "p": 4,
        }
        result = index.get_anti_attribute_tokens_by_frequency(
            unit_counts, threshold=50
        )
        assert 0 in result  # cat — all units ≥ 50

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_excludes_tokens_below_threshold(self, mock_g2p):
        mock_g2p.side_effect = _mock_g2p_batch
        vocab = {"cat": 0, "ship": 1}
        tok = _mock_tokenizer(vocab)

        index = AttributeWordIndex()
        index.build(tok)

        unit_counts = {
            "k": 100, "æ": 100, "t": 100,
            "k-æ": 80, "æ-t": 80, "k-æ-t": 60,
            "ʃ": 2, "ɪ": 3, "p": 4,
            "ʃ-ɪ": 1, "ɪ-p": 1, "ʃ-ɪ-p": 1,
        }
        result = index.get_anti_attribute_tokens_by_frequency(
            unit_counts, threshold=50
        )
        assert 0 in result   # cat — all above
        assert 1 not in result  # ship — ʃ, ɪ, p below

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_missing_unit_in_counts_treated_as_zero(self, mock_g2p):
        mock_g2p.side_effect = _mock_g2p_batch
        vocab = {"cat": 0}
        tok = _mock_tokenizer(vocab)

        index = AttributeWordIndex()
        index.build(tok)

        # k not in counts — treated as 0, below threshold
        unit_counts = {"æ": 100, "t": 100}
        result = index.get_anti_attribute_tokens_by_frequency(
            unit_counts, threshold=50
        )
        assert 0 not in result

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_threshold_zero_returns_all(self, mock_g2p):
        mock_g2p.side_effect = _mock_g2p_batch
        vocab = {"cat": 0, "ship": 1}
        tok = _mock_tokenizer(vocab)

        index = AttributeWordIndex()
        index.build(tok)

        unit_counts = {
            "k": 1, "æ": 1, "t": 1,
            "k-æ": 1, "æ-t": 1, "k-æ-t": 1,
            "ʃ": 1, "ɪ": 1, "p": 1,
            "ʃ-ɪ": 1, "ɪ-p": 1, "ʃ-ɪ-p": 1,
        }
        result = index.get_anti_attribute_tokens_by_frequency(
            unit_counts, threshold=0
        )
        # All tokens have counts > 0
        assert 0 in result
        assert 1 in result

    def test_raises_if_not_built(self):
        index = AttributeWordIndex()
        with pytest.raises(RuntimeError, match="not.*built"):
            index.get_anti_attribute_tokens_by_frequency({"k": 10}, threshold=5)


# ---------------------------------------------------------------------------
# unit_to_tokens property
# ---------------------------------------------------------------------------


class TestUnitToTokens:
    """Read-only access to the full mapping."""

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_returns_copy(self, mock_g2p):
        mock_g2p.side_effect = _mock_g2p_batch
        vocab = {"cat": 0}
        tok = _mock_tokenizer(vocab)

        index = AttributeWordIndex()
        index.build(tok)

        mapping1 = index.unit_to_tokens
        mapping2 = index.unit_to_tokens
        assert mapping1 is not mapping2  # different dict objects

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_mutation_does_not_affect_index(self, mock_g2p):
        mock_g2p.side_effect = _mock_g2p_batch
        vocab = {"cat": 0}
        tok = _mock_tokenizer(vocab)

        index = AttributeWordIndex()
        index.build(tok)

        mapping = index.unit_to_tokens
        mapping["k"] = set()  # mutate the copy
        assert index.unit_to_tokens["k"] == {0}  # original unchanged

    def test_raises_if_not_built(self):
        index = AttributeWordIndex()
        with pytest.raises(RuntimeError, match="not.*built"):
            _ = index.unit_to_tokens


# ---------------------------------------------------------------------------
# token_units property
# ---------------------------------------------------------------------------


class TestTokenUnits:
    """Reverse mapping: token ID → set of units it contains."""

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_returns_units_for_token(self, mock_g2p):
        mock_g2p.side_effect = _mock_g2p_batch
        vocab = {"cat": 0}
        tok = _mock_tokenizer(vocab)

        index = AttributeWordIndex()
        index.build(tok)

        token_map = index.token_units
        # cat → phonemes k, æ, t; diphones k-æ, æ-t; triphone k-æ-t
        assert "k" in token_map[0]
        assert "æ" in token_map[0]
        assert "k-æ" in token_map[0]
        assert "k-æ-t" in token_map[0]

    def test_raises_if_not_built(self):
        index = AttributeWordIndex()
        with pytest.raises(RuntimeError, match="not.*built"):
            _ = index.token_units
