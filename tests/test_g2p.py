"""Tests for the G2P (grapheme-to-phoneme) module."""

import os
from unittest.mock import patch

import pytest
from phonemizer.backend import EspeakBackend

from corpusgen.g2p.manager import (
    G2PManager,
    _check_espeak_available,
    _windows_espeak_library_candidates,
)
from corpusgen.g2p.result import G2PResult

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

    def test_phonemize_batch_with_empty_strings(self, manager):
        """Batch with empty strings must preserve alignment.

        Regression: phonemizer drops empty utterances, so naive zip
        misaligns outputs. Results must match input order exactly.
        """
        texts = ["hello", "", "world", "  ", "test"]
        results = manager.phonemize_batch(texts, language="en-us")
        assert len(results) == 5
        # Index 0: "hello" → has phonemes
        assert results[0].text == "hello"
        assert len(results[0].phonemes) > 0
        # Index 1: "" → empty result
        assert results[1].text == ""
        assert results[1].phonemes == []
        assert results[1].ipa == ""
        # Index 2: "world" → has phonemes
        assert results[2].text == "world"
        assert len(results[2].phonemes) > 0
        # Index 3: "  " → whitespace-only, empty result
        assert results[3].phonemes == []
        # Index 4: "test" → has phonemes
        assert results[4].text == "test"
        assert len(results[4].phonemes) > 0

    def test_phonemize_batch_all_empty(self, manager):
        """Batch of all empty strings returns all empty results."""
        texts = ["", "", ""]
        results = manager.phonemize_batch(texts, language="en-us")
        assert len(results) == 3
        assert all(r.phonemes == [] for r in results)

    def test_phonemize_batch_empty_list(self, manager):
        """Empty input list returns empty output."""
        assert manager.phonemize_batch([], language="en-us") == []

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
        # Must contain actual language codes, not file paths
        for lang in languages:
            assert "\\" not in lang, f"File path leaked into language code: {lang}"
            assert "/" not in lang, f"File path leaked into language code: {lang}"

    def test_supported_languages_contains_english(self, manager):
        """English must be in the supported languages list."""
        languages = manager.supported_languages()
        has_english = any(lang.startswith("en") for lang in languages)
        assert has_english, f"No English variant found in: {languages[:10]}..."


class TestEspeakAvailability:
    """Tests for actionable, platform-aware espeak-ng detection."""

    def test_standard_windows_dll_is_configured_automatically(self):
        program_files = os.path.join("D:\\", "Programs")
        env = {"PROGRAMFILES": program_files}
        with (
            patch.dict(os.environ, env, clear=True),
            patch("corpusgen.g2p.manager.sys.platform", "win32"),
            patch("corpusgen.g2p.manager.sys.maxsize", 2**63 - 1),
            patch("corpusgen.g2p.manager.os.path.exists", return_value=True),
            patch.object(EspeakBackend, "set_library") as set_library,
            patch.object(
                EspeakBackend,
                "version",
                side_effect=[RuntimeError("not discovered"), (1, 52, 0)],
            ),
        ):
            _check_espeak_available()

        set_library.assert_called_once_with(
            os.path.join(program_files, "eSpeak NG", "libespeak-ng.dll")
        )

    @pytest.mark.parametrize(
        ("maxsize", "expected_roots"),
        [
            (2**63 - 1, [r"C:\Program Files", r"C:\Program Files (x86)"]),
            (2**31 - 1, [r"C:\Program Files (x86)", r"C:\Program Files"]),
        ],
    )
    def test_windows_dll_candidates_follow_interpreter_architecture(
        self, maxsize, expected_roots
    ):
        env = {
            "PROGRAMFILES": r"C:\Program Files",
            "PROGRAMFILES(X86)": r"C:\Program Files (x86)",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch("corpusgen.g2p.manager.sys.maxsize", maxsize),
        ):
            candidates = _windows_espeak_library_candidates()

        assert candidates == [
            os.path.join(root, "eSpeak NG", "libespeak-ng.dll")
            for root in expected_roots
        ]

    def test_windows_fallback_tries_both_install_trees(self):
        env = {
            "PROGRAMFILES": r"C:\Program Files",
            "PROGRAMFILES(X86)": r"C:\Program Files (x86)",
        }
        expected = [
            os.path.join(root, "eSpeak NG", "libespeak-ng.dll")
            for root in (env["PROGRAMFILES"], env["PROGRAMFILES(X86)"])
        ]
        with (
            patch.dict(os.environ, env, clear=True),
            patch("corpusgen.g2p.manager.sys.platform", "win32"),
            patch("corpusgen.g2p.manager.sys.maxsize", 2**63 - 1),
            patch("corpusgen.g2p.manager.os.path.exists", return_value=True),
            patch.object(EspeakBackend, "set_library") as set_library,
            patch.object(
                EspeakBackend,
                "version",
                side_effect=[
                    RuntimeError("not discovered"),
                    RuntimeError("wrong DLL architecture"),
                    (1, 52, 0),
                ],
            ),
        ):
            _check_espeak_available()

        assert [call.args[0] for call in set_library.call_args_list] == expected

    def test_programmatically_configured_library_is_preserved(self):
        env = {"PROGRAMFILES": r"C:\Program Files"}
        with (
            patch.dict(os.environ, env, clear=True),
            patch("corpusgen.g2p.manager.sys.platform", "win32"),
            patch("corpusgen.g2p.manager.os.path.exists", return_value=True),
            patch.object(EspeakBackend, "set_library") as set_library,
            patch.object(EspeakBackend, "version", return_value=(1, 52, 0)),
        ):
            _check_espeak_available()

        set_library.assert_not_called()

    def test_library_load_error_is_preserved(self):
        with patch.object(
            EspeakBackend,
            "version",
            side_effect=RuntimeError("wrong DLL architecture"),
        ), pytest.raises(RuntimeError, match="wrong DLL architecture"):
            _check_espeak_available()

    def test_legacy_espeak_is_rejected(self):
        with patch.object(
            EspeakBackend, "version", return_value=(1, 48, 0)
        ), pytest.raises(RuntimeError, match="Legacy eSpeak"):
            _check_espeak_available()


class TestG2PImports:
    """Test that g2p subpackage exports are accessible."""

    def test_import_from_g2p(self):
        from corpusgen.g2p import G2PManager, G2PResult
        assert G2PManager is not None
        assert G2PResult is not None
