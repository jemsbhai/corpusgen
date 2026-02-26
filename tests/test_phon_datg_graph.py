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


class _FakeTensorView:
    """View into a _FakeTensor for vectorized [:, list] operations."""

    def __init__(self, tensor: "_FakeTensor", rows: list[int], cols: list[int]) -> None:
        self._tensor = tensor
        self._rows = rows
        self._cols = cols

    def __iadd__(self, scalar: float) -> "_FakeTensorView":
        for r in self._rows:
            for c in self._cols:
                self._tensor.data[r][c] += scalar
        return self


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
            if isinstance(row, slice) and isinstance(col, list):
                rows = list(range(*row.indices(self.shape[0])))
                return _FakeTensorView(self, rows, col)
            return self.data[row][col]
        return self.data[key]

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key) == 2:
            row, col = key
            if isinstance(row, slice) and isinstance(col, list) and isinstance(value, _FakeTensorView):
                pass  # already applied in-place via __iadd__
            else:
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


# ---------------------------------------------------------------------------
# Diphone unit_level — anti-attribute filtering at diphone level
# ---------------------------------------------------------------------------


class TestDiphoneUnitLevel:
    """DATGStrategy with unit='diphone' filters anti-attribute tokens by diphones only.

    When targets use diphone coverage, a token is anti-attribute only if ALL
    its diphone-level units are already covered. Phoneme and triphone units
    are irrelevant to this determination.

    This matters for papers: if we report diphone coverage and the steering
    was filtering on the wrong unit level, results would be misleading.
    """

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_anti_attribute_when_diphones_covered(self, mock_g2p):
        """Token whose diphones are all covered → anti-attribute.

        cat → [k, æ, t] → diphones: {k-æ, æ-t}
        Cover k-æ and æ-t → cat is anti-attribute at diphone level.
        """
        mock_g2p.side_effect = _mock_g2p_batch
        targets = _make_targets(
            ["k", "æ", "t", "s", "ʃ", "ɪ", "p"], unit="diphone"
        )
        vocab = {"cat": 0, "ship": 1}
        tok = _mock_tokenizer(vocab)
        model = MagicMock()

        # Cover diphones k-æ and æ-t (sequential from [k, æ, t])
        targets.tracker.update(["k", "æ", "t"], sentence_index=0)

        strategy = DATGStrategy(targets=targets, anti_attribute_mode="covered")
        strategy.prepare(target_units=["ʃ-ɪ"], model=model, tokenizer=tok)

        # cat's diphones (k-æ, æ-t) are both covered → anti-attribute
        assert 0 in strategy.current_anti_attribute_tokens

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_not_anti_attribute_when_diphones_uncovered(self, mock_g2p):
        """Token with uncovered diphones → NOT anti-attribute.

        ship → [ʃ, ɪ, p] → diphones: {ʃ-ɪ, ɪ-p}
        Only k-æ and æ-t covered → ship is NOT anti-attribute.
        """
        mock_g2p.side_effect = _mock_g2p_batch
        targets = _make_targets(
            ["k", "æ", "t", "s", "ʃ", "ɪ", "p"], unit="diphone"
        )
        vocab = {"cat": 0, "ship": 1}
        tok = _mock_tokenizer(vocab)
        model = MagicMock()

        # Only cover cat's diphones
        targets.tracker.update(["k", "æ", "t"], sentence_index=0)

        strategy = DATGStrategy(targets=targets, anti_attribute_mode="covered")
        strategy.prepare(target_units=["ʃ-ɪ"], model=model, tokenizer=tok)

        # ship's diphones (ʃ-ɪ, ɪ-p) are uncovered → not anti-attribute
        assert 1 not in strategy.current_anti_attribute_tokens

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_diphone_ignores_phoneme_coverage(self, mock_g2p):
        """Covering individual phonemes does NOT make diphone-level anti-attribute.

        Even if k, æ, t are all seen as phonemes, the diphones k-æ and æ-t
        must be explicitly covered for a token to be anti-attribute in diphone
        mode. This test ensures the unit_level filter is actually applied.
        """
        mock_g2p.side_effect = _mock_g2p_batch
        targets = _make_targets(
            ["k", "æ", "t"], unit="diphone"
        )
        vocab = {"cat": 0}
        tok = _mock_tokenizer(vocab)
        model = MagicMock()

        # Do NOT update tracker — no diphones covered
        # (even though the _phonemes_ exist in the inventory)
        strategy = DATGStrategy(targets=targets, anti_attribute_mode="covered")
        strategy.prepare(target_units=["k-t"], model=model, tokenizer=tok)

        # cat should NOT be anti-attribute — no diphones covered yet
        assert 0 not in strategy.current_anti_attribute_tokens


# ---------------------------------------------------------------------------
# Triphone unit_level — anti-attribute filtering at triphone level
# ---------------------------------------------------------------------------


class TestTriphoneUnitLevel:
    """DATGStrategy with unit='triphone' filters anti-attribute by triphones only."""

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_anti_attribute_when_triphones_covered(self, mock_g2p):
        """Token whose triphones are all covered → anti-attribute.

        cat → [k, æ, t] → triphones: {k-æ-t}
        Cover k-æ-t → cat is anti-attribute at triphone level.
        """
        mock_g2p.side_effect = _mock_g2p_batch
        targets = _make_targets(
            ["k", "æ", "t", "ʃ", "ɪ", "p"], unit="triphone"
        )
        vocab = {"cat": 0, "ship": 1}
        tok = _mock_tokenizer(vocab)
        model = MagicMock()

        # Cover triphone k-æ-t
        targets.tracker.update(["k", "æ", "t"], sentence_index=0)

        strategy = DATGStrategy(targets=targets, anti_attribute_mode="covered")
        strategy.prepare(target_units=["ʃ-ɪ-p"], model=model, tokenizer=tok)

        # cat's triphone (k-æ-t) is covered → anti-attribute
        assert 0 in strategy.current_anti_attribute_tokens

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_not_anti_attribute_when_triphones_uncovered(self, mock_g2p):
        """Token with uncovered triphones → NOT anti-attribute."""
        mock_g2p.side_effect = _mock_g2p_batch
        targets = _make_targets(
            ["k", "æ", "t", "ʃ", "ɪ", "p"], unit="triphone"
        )
        vocab = {"cat": 0, "ship": 1}
        tok = _mock_tokenizer(vocab)
        model = MagicMock()

        # Cover only cat's triphone, not ship's
        targets.tracker.update(["k", "æ", "t"], sentence_index=0)

        strategy = DATGStrategy(targets=targets, anti_attribute_mode="covered")
        strategy.prepare(target_units=["ʃ-ɪ-p"], model=model, tokenizer=tok)

        # ship's triphone (ʃ-ɪ-p) is uncovered → not anti-attribute
        assert 1 not in strategy.current_anti_attribute_tokens

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_triphone_ignores_diphone_coverage(self, mock_g2p):
        """Covering diphones does NOT satisfy triphone-level filtering.

        This ensures unit_level='triphone' truly filters to triphones only.
        """
        mock_g2p.side_effect = _mock_g2p_batch
        targets = _make_targets(
            ["k", "æ", "t"], unit="triphone"
        )
        vocab = {"cat": 0}
        tok = _mock_tokenizer(vocab)
        model = MagicMock()

        # No triphones covered (tracker is fresh)
        strategy = DATGStrategy(targets=targets, anti_attribute_mode="covered")
        strategy.prepare(target_units=["k-æ-t"], model=model, tokenizer=tok)

        # cat should NOT be anti-attribute — no triphones covered
        assert 0 not in strategy.current_anti_attribute_tokens

    @patch("corpusgen.generate.phon_datg.attribute_words._phonemize_batch")
    def test_short_token_no_triphones_not_anti_attribute(self, mock_g2p):
        """Token with <3 phonemes has no triphones → never anti-attribute in triphone mode.

        'the' → [ð, ə] → triphones: {} (empty)
        _filter_by_level returns empty set, and empty filtered set means
        the token is excluded (not anti-attribute).
        """
        mock_g2p.side_effect = _mock_g2p_batch
        targets = _make_targets(
            ["ð", "ə", "k", "æ", "t"], unit="triphone"
        )
        vocab = {"the": 0, "cat": 1}
        tok = _mock_tokenizer(vocab)
        model = MagicMock()

        # Cover everything cat has
        targets.tracker.update(["k", "æ", "t"], sentence_index=0)

        strategy = DATGStrategy(targets=targets, anti_attribute_mode="covered")
        strategy.prepare(target_units=["ð-ə-k"], model=model, tokenizer=tok)

        # 'the' has no triphones → filtered set is empty → not anti-attribute
        assert 0 not in strategy.current_anti_attribute_tokens
        # 'cat' has triphone k-æ-t covered → anti-attribute
        assert 1 in strategy.current_anti_attribute_tokens
