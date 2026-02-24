"""Tests for the G2P (grapheme-to-phoneme) module."""

import pytest

from corpusgen.g2p.result import G2PResult
from corpusgen.g2p.manager import G2PManager


# --- G2PResult dataclass tests ---


class TestG2PResult:
    """Tests for the G2PResult data container."""

    def test_create_result(self):
        """G2PResult stores IPA string and parsed phoneme list."""
        result = G2PResult(
            text="hello",
            ipa="həˈloʊ",
            phonemes=["h", "ə", "l", "oʊ"],
            language="en-us",
        )
        assert result.text == "hello"
        assert result.ipa == "həˈloʊ"
        assert result.phonemes == ["h", "ə", "l", "oʊ"]
        assert result.language == "en-us"

    def test_diphones(self):
        """G2PResult computes diphones from phoneme list."""
        result = G2PResult(
            text="hello",
            ipa="həˈloʊ",
            phonemes=["h", "ə", "l", "oʊ"],
            language="en-us",
        )
        assert result.diphones == ["h-ə", "ə-l", "l-oʊ"]

    def test_diphones_single_phoneme(self):
        """Diphones list is empty for a single phoneme."""
        result = G2PResult(
            text="a",
            ipa="ə",
            phonemes=["ə"],
            language="en-us",
        )
        assert result.diphones == []

    def test_diphones_empty(self):
        """Diphones list is empty for empty phonemes."""
        result = G2PResult(
            text="",
            ipa="",
            phonemes=[],
            language="en-us",
        )
        assert result.diphones == []

    def test_triphones(self):
        """G2PResult computes triphones from phoneme list."""
        result = G2PResult(
            text="hello",
            ipa="həˈloʊ",
            phonemes=["h", "ə", "l", "oʊ"],
            language="en-us",
        )
        assert result.triphones == ["h-ə-l", "ə-l-oʊ"]

    def test_phoneme_count(self):
        """G2PResult reports phoneme count."""
        result = G2PResult(
            text="hello",
            ipa="həˈloʊ",
            phonemes=["h", "ə", "l", "oʊ"],
            language="en-us",
        )
        assert result.phoneme_count == 4

    def test_unique_phonemes(self):
        """G2PResult reports unique phonemes as a set."""
        result = G2PResult(
            text="papa",
            ipa="pɑːpə",
            phonemes=["p", "ɑː", "p", "ə"],
            language="en-us",
        )
        assert result.unique_phonemes == {"p", "ɑː", "ə"}


# --- G2PManager tests ---


class TestG2PManager:
    """Tests for the G2P manager using espeak-ng backend."""

    @pytest.fixture
    def manager(self):
        """Create a G2PManager with default espeak backend."""
        return G2PManager(backend="espeak")

    def test_create_manager(self, manager):
        """Manager initializes with specified backend."""
        assert manager.backend == "espeak"

    def test_phonemize_single_word(self, manager):
        """Phonemize a single English word."""
        result = manager.phonemize("hello", language="en-us")
        assert isinstance(result, G2PResult)
        assert result.text == "hello"
        assert len(result.ipa) > 0
        assert len(result.phonemes) > 0

    def test_phonemize_sentence(self, manager):
        """Phonemize a full sentence."""
        result = manager.phonemize(
            "The quick brown fox.", language="en-us"
        )
        assert isinstance(result, G2PResult)
        assert len(result.phonemes) > 5  # sentence has many phonemes

    def test_phonemize_returns_ipa(self, manager):
        """Output should be valid IPA (contains IPA characters)."""
        result = manager.phonemize("think", language="en-us")
        # 'think' should contain θ (theta) in IPA
        assert "θ" in result.ipa or "ɪ" in result.ipa

    def test_phonemize_different_languages(self, manager):
        """Manager handles different languages via espeak-ng."""
        result_en = manager.phonemize("hello", language="en-us")
        result_fr = manager.phonemize("bonjour", language="fr-fr")
        # Different languages should produce different IPA
        assert result_en.ipa != result_fr.ipa

    def test_phonemize_empty_string(self, manager):
        """Empty input returns empty result."""
        result = manager.phonemize("", language="en-us")
        assert result.phonemes == []
        assert result.ipa == ""

    def test_phonemize_batch(self, manager):
        """Batch phonemize multiple texts."""
        texts = ["hello", "world", "test"]
        results = manager.phonemize_batch(texts, language="en-us")
        assert len(results) == 3
        assert all(isinstance(r, G2PResult) for r in results)

    def test_phonemize_variants(self, manager):
        """Get multiple pronunciation variants for a word."""
        variants = manager.phonemize_variants("the", language="en-us")
        assert isinstance(variants, list)
        assert len(variants) >= 1
        assert all(isinstance(v, G2PResult) for v in variants)

    def test_supported_languages(self, manager):
        """Manager reports which languages are supported."""
        languages = manager.supported_languages()
        assert isinstance(languages, list)
        assert len(languages) > 0
        assert "en-us" in languages or "en" in languages
