"""Tests for DATGStrategy — Phon-DATG GuidanceStrategy implementation."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from corpusgen.generate.guidance import GuidanceStrategy
from corpusgen.generate.phon_ctg.targets import PhoneticTargetInventory
from corpusgen.generate.phon_datg.attribute_words import AttributeWordIndex
from corpusgen.generate.phon_datg.graph import DATGStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_targets(phonemes: list[str], unit: str = "phoneme") -> PhoneticTargetInventory:
    """Create a PhoneticTargetInventory from a phoneme list."""
    return PhoneticTargetInventory(target_phonemes=phonemes, unit=unit)


def _mock_tokenizer(vocab: dict[str, int]) -> MagicMock:
    """Create a mock tokenizer with a known vocabulary."""
    tok = MagicMock()
    tok.get_vocab.return_value = dict(vocab)
    id_to_str = {v: k for k, v in vocab.items()}
    tok.batch_decode.side_effect = lambda ids, **kw: [
        id_to_str.get(i, "") for i in ids
    ]
    tok.decode.side_effect = lambda i, **kw: id_to_str.get(i, "")
    return tok


def _mock_g2p_batch(texts: list[str], language: str = "en-us") -> list:
    """Simulated G2P for test vocabulary."""
    word_phonemes = {
        "cat": ["k", "æ", "t"],
        "kit": ["k", "ɪ", "t"],
        "sat": ["s", "æ", "t"],
        "ship": ["ʃ", "ɪ", "p"],
        "thin": ["θ", "ɪ", "n"],
    }
    results = []
    for text in texts:
        cleaned = text.strip().lower().lstrip("▁")
        phonemes = word_phonemes.get(cleaned, [])
        result = MagicMock()
        result.phonemes = phonemes
        results.append(result)
    return results


class _FakeTensor:
    """Minimal tensor for testing without torch."""

    def __init__(self, data: list[list[float]]) -> None:
        self.data = [row[:] for row in data]
        self.shape = (len(data), len(data[0]) if data else 0)

    def clone(self) -> "_FakeTensor":
        return _FakeTensor([row[:] for row in self.data])

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            row, col = key
            return self.data[row][col]
        return self.data[key]

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key) == 2:
            row, col = key
            self.data[row][col] = value
        else:
            self.data[key] = value


def _make_logits(vocab_size: int = 5, fill: float = 0.0) -> _FakeTensor:
    return _FakeTensor([[fill] * vocab_size])


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """DATGStrategy creation and configuration."""

    def test_basic_creation(self):
        targets = _make_targets(["k", "æ", "t"])
        strategy = DATGStrategy(targets=targets)
        assert isinstance(strategy, GuidanceStrategy)

    def test_name(self):
        targets = _make_targets(["k"])
        strategy = DATGStrategy(targets=targets)
        assert strategy.name == "phon_datg"

    def test_default_language(self):
        targets = _make_targets(["k"])
        strategy = DATGStrategy(targets=targets)
        assert strategy.language == "en-us"

    def test_custom_language(self):
        targets = _make_targets(["k"])
        strategy = DATGStrategy(targets=targets, language="fr-fr")
        assert strategy.language == "fr-fr"

    def test_default_boost_strength(self):
        targets = _make_targets(["k"])
        strategy = DATGStrategy(targets=targets)
        assert strategy.boost_strength == 5.0

    def test_default_penalty_strength(self):
        targets = _make_targets(["k"])
        strategy = DATGStrategy(targets=targets)
        assert strategy.penalty_strength == -5.0

    def test_custom_strengths(self):
        targets = _make_targets(["k"])
        strategy = DATGStrategy(
            targets=targets,
            boost_strength=10.0,
            penalty_strength=-3.0,
        )
        assert strategy.boost_strength == 10.0
        assert strategy.penalty_strength == -3.0

    def test_default_anti_attribute_mode(self):
        targets = _make_targets(["k"])
        strategy = DATGStrategy(targets=targets)
        assert strategy.anti_attribute_mode == "covered"

    def test_frequency_mode(self):
        targets = _make_targets(["k"])
        strategy = DATGStrategy(targets=targets, anti_attribute_mode="frequency")
        assert strategy.anti_attribute_mode == "frequency"

    def test_invalid_mode_raises(self):
        targets = _make_targets(["k"])
        with pytest.raises(ValueError, match="anti_attribute_mode"):
            DATGStrategy(targets=targets, anti_attribute_mode="invalid")

    def test_custom_frequency_threshold(self):
        targets = _make_targets(["k"])
        strategy = DATGStrategy(
            targets=targets,
            anti_attribute_mode="frequency",
            frequency_threshold=20,
        )
        assert strategy.frequency_threshold == 20

    def test_injected_attribute_word_index(self):
        targets = _make_targets(["k"])
        index = AttributeWordIndex(language="en-us")
        strategy = DATGStrategy(targets=targets, attribute_word_index=index)
        assert strategy.attribute_word_index is index


# ---------------------------------------------------------------------------
# Prepare: lazy index build
# ---------------------------------------------------------------------------


class TestPrepare:
    """prepare() builds index lazily and computes attribute sets."""

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_builds_index_on_first_prepare(self, mock_g2p):
        mock_g2p.side_effect = _mock_g2p_batch
        targets = _make_targets(["k", "æ", "t", "s", "ʃ", "ɪ", "p"])
        vocab = {"cat": 0, "sat": 1, "ship": 2}
        tok = _mock_tokenizer(vocab)
        model = MagicMock()

        strategy = DATGStrategy(targets=targets)
        strategy.prepare(target_units=["ʃ"], model=model, tokenizer=tok)

        assert strategy.attribute_word_index.is_built

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_index_built_once(self, mock_g2p):
        mock_g2p.side_effect = _mock_g2p_batch
        targets = _make_targets(["k", "æ", "t", "s", "ʃ", "ɪ", "p"])
        vocab = {"cat": 0, "ship": 1}
        tok = _mock_tokenizer(vocab)
        model = MagicMock()

        strategy = DATGStrategy(targets=targets)
        strategy.prepare(target_units=["ʃ"], model=model, tokenizer=tok)
        strategy.prepare(target_units=["k"], model=model, tokenizer=tok)

        # G2P only called once (idempotent build)
        assert mock_g2p.call_count == 1

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_attribute_tokens_populated(self, mock_g2p):
        mock_g2p.side_effect = _mock_g2p_batch
        targets = _make_targets(["k", "æ", "t", "s", "ʃ", "ɪ", "p"])
        vocab = {"cat": 0, "sat": 1, "ship": 2}
        tok = _mock_tokenizer(vocab)
        model = MagicMock()

        strategy = DATGStrategy(targets=targets)
        strategy.prepare(target_units=["ʃ"], model=model, tokenizer=tok)

        # "ship" (id=2) contains ʃ
        assert 2 in strategy.current_attribute_tokens

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_anti_attribute_tokens_covered_mode(self, mock_g2p):
        mock_g2p.side_effect = _mock_g2p_batch
        targets = _make_targets(["k", "æ", "t", "s", "ʃ", "ɪ", "p"])
        vocab = {"cat": 0, "sat": 1, "ship": 2}
        tok = _mock_tokenizer(vocab)
        model = MagicMock()

        # Simulate "k", "æ", "t" and all their n-grams already covered
        strategy = DATGStrategy(targets=targets, anti_attribute_mode="covered")
        # Cover k, æ, t and their diphones/triphones by updating tracker
        targets.tracker.update(["k", "æ", "t"], sentence_index=0)

        strategy.prepare(target_units=["ʃ"], model=model, tokenizer=tok)

        # "cat" (id=0) has only k, æ, t (+ k-æ, æ-t, k-æ-t) — all covered
        assert 0 in strategy.current_anti_attribute_tokens

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_prepare_updates_on_second_call(self, mock_g2p):
        """prepare() recomputes attribute sets with new targets."""
        mock_g2p.side_effect = _mock_g2p_batch
        targets = _make_targets(["k", "æ", "t", "s", "ʃ", "ɪ", "p"])
        vocab = {"cat": 0, "sat": 1, "ship": 2}
        tok = _mock_tokenizer(vocab)
        model = MagicMock()

        strategy = DATGStrategy(targets=targets)

        # First: target ʃ → ship is attribute
        strategy.prepare(target_units=["ʃ"], model=model, tokenizer=tok)
        assert 2 in strategy.current_attribute_tokens

        # Second: target s → sat is attribute
        strategy.prepare(target_units=["s"], model=model, tokenizer=tok)
        assert 1 in strategy.current_attribute_tokens


# ---------------------------------------------------------------------------
# modify_logits
# ---------------------------------------------------------------------------


class TestModifyLogits:
    """modify_logits() delegates to LogitModulator."""

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_boosts_attribute_tokens(self, mock_g2p):
        mock_g2p.side_effect = _mock_g2p_batch
        targets = _make_targets(["k", "æ", "t", "s", "ʃ", "ɪ", "p"])
        vocab = {"cat": 0, "sat": 1, "ship": 2}
        tok = _mock_tokenizer(vocab)
        model = MagicMock()

        strategy = DATGStrategy(
            targets=targets,
            boost_strength=5.0,
            penalty_strength=0.0,
        )
        strategy.prepare(target_units=["ʃ"], model=model, tokenizer=tok)

        logits = _make_logits(vocab_size=3, fill=0.0)
        result = strategy.modify_logits(input_ids=MagicMock(), logits=logits)

        # ship (id=2) should be boosted
        assert result[0, 2] > 0.0

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_penalizes_anti_attribute_tokens(self, mock_g2p):
        mock_g2p.side_effect = _mock_g2p_batch
        targets = _make_targets(["k", "æ", "t", "s", "ʃ", "ɪ", "p"])
        vocab = {"cat": 0, "sat": 1, "ship": 2}
        tok = _mock_tokenizer(vocab)
        model = MagicMock()

        strategy = DATGStrategy(
            targets=targets,
            boost_strength=0.0,
            penalty_strength=-5.0,
            anti_attribute_mode="covered",
        )
        # Cover cat's units
        targets.tracker.update(["k", "æ", "t"], sentence_index=0)

        strategy.prepare(target_units=["ʃ"], model=model, tokenizer=tok)

        logits = _make_logits(vocab_size=3, fill=0.0)
        result = strategy.modify_logits(input_ids=MagicMock(), logits=logits)

        # cat (id=0) should be penalized
        assert result[0, 0] < 0.0

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_raises_if_prepare_not_called(self, mock_g2p):
        targets = _make_targets(["k"])
        strategy = DATGStrategy(targets=targets)

        logits = _make_logits(vocab_size=3, fill=0.0)
        with pytest.raises(RuntimeError, match="prepare"):
            strategy.modify_logits(input_ids=MagicMock(), logits=logits)


# ---------------------------------------------------------------------------
# Anti-attribute frequency mode
# ---------------------------------------------------------------------------


class TestFrequencyMode:
    """anti_attribute_mode='frequency' uses tracker counts."""

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_frequency_mode_uses_counts(self, mock_g2p):
        mock_g2p.side_effect = _mock_g2p_batch
        targets = _make_targets(["k", "æ", "t", "s", "ʃ", "ɪ", "p"])
        vocab = {"cat": 0, "sat": 1, "ship": 2}
        tok = _mock_tokenizer(vocab)
        model = MagicMock()

        strategy = DATGStrategy(
            targets=targets,
            anti_attribute_mode="frequency",
            frequency_threshold=2,
        )

        # Update tracker multiple times so k, æ, t have counts > 2
        targets.tracker.update(["k", "æ", "t"], sentence_index=0)
        targets.tracker.update(["k", "æ", "t"], sentence_index=1)
        targets.tracker.update(["k", "æ", "t"], sentence_index=2)

        strategy.prepare(target_units=["ʃ"], model=model, tokenizer=tok)

        # "cat" units (k, æ, t, k-æ, æ-t, k-æ-t) should all be > 2
        assert 0 in strategy.current_anti_attribute_tokens


# ---------------------------------------------------------------------------
# Pre-built index
# ---------------------------------------------------------------------------


class TestPreBuiltIndex:
    """User can inject a pre-built AttributeWordIndex."""

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_uses_injected_index(self, mock_g2p):
        mock_g2p.side_effect = _mock_g2p_batch
        targets = _make_targets(["k", "æ", "t", "s", "ʃ", "ɪ", "p"])
        vocab = {"cat": 0, "ship": 1}
        tok = _mock_tokenizer(vocab)
        model = MagicMock()

        # Pre-build the index
        index = AttributeWordIndex(language="en-us")
        index.build(tok)

        strategy = DATGStrategy(targets=targets, attribute_word_index=index)
        strategy.prepare(target_units=["ʃ"], model=model, tokenizer=tok)

        # Should use the pre-built index, not build a new one
        assert strategy.attribute_word_index is index
        assert 1 in strategy.current_attribute_tokens  # ship


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary conditions."""

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_empty_target_units(self, mock_g2p):
        mock_g2p.side_effect = _mock_g2p_batch
        targets = _make_targets(["k"])
        vocab = {"cat": 0}
        tok = _mock_tokenizer(vocab)
        model = MagicMock()

        strategy = DATGStrategy(targets=targets)
        strategy.prepare(target_units=[], model=model, tokenizer=tok)

        assert strategy.current_attribute_tokens == set()

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_no_matching_attribute_words(self, mock_g2p):
        mock_g2p.side_effect = _mock_g2p_batch
        targets = _make_targets(["k", "æ", "t"])
        vocab = {"cat": 0}
        tok = _mock_tokenizer(vocab)
        model = MagicMock()

        strategy = DATGStrategy(targets=targets)
        # Target ʒ — not in any vocab word
        strategy.prepare(target_units=["ʒ"], model=model, tokenizer=tok)

        assert strategy.current_attribute_tokens == set()
