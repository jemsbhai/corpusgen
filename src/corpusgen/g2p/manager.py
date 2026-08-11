"""G2PManager: multi-backend grapheme-to-phoneme conversion."""

from __future__ import annotations

import os
import re
import sys

from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

from corpusgen.g2p.result import G2PResult

_ESPEAK_INSTALL_HELP = """
===================================================================
 espeak-ng unavailable — required by corpusgen for phoneme conversion
===================================================================

 Install espeak-ng for your platform:

   Windows:  Download the matching-architecture .msi from
             https://github.com/espeak-ng/espeak-ng/releases
             The default install path is detected automatically. For a
             custom install, set the environment variable:
             [Environment]::SetEnvironmentVariable("PHONEMIZER_ESPEAK_LIBRARY",
                 "C:\\Program Files\\eSpeak NG\\libespeak-ng.dll", "User")

   macOS:    brew install espeak-ng

   Linux:    sudo apt-get install espeak-ng

 See https://github.com/jemsbhai/corpusgen#prerequisites for full instructions.
===================================================================
"""

def _windows_espeak_library_candidates() -> list[str]:
    """Return likely MSI-installed DLLs in interpreter-compatible order."""
    program_files = os.environ.get("PROGRAMFILES")
    program_files_x86 = os.environ.get("PROGRAMFILES(X86)")
    roots = (
        (program_files, program_files_x86)
        if sys.maxsize > 2**32
        else (program_files_x86, program_files)
    )

    candidates: list[str] = []
    seen: set[str] = set()
    for root in roots:
        if not root:
            continue
        candidate = os.path.join(root, "eSpeak NG", "libespeak-ng.dll")
        normalized = os.path.normcase(candidate)
        if normalized not in seen:
            seen.add(normalized)
            candidates.append(candidate)
    return candidates


def _check_espeak_available() -> None:
    """Verify that a loadable espeak-ng shared library is available."""
    try:
        version = EspeakBackend.version()
    except RuntimeError as exc:
        can_try_windows_defaults = sys.platform == "win32" and not os.environ.get(
            "PHONEMIZER_ESPEAK_LIBRARY"
        )
        if not can_try_windows_defaults:
            raise RuntimeError(
                f"{_ESPEAK_INSTALL_HELP}\nUnderlying library error: {exc}"
            ) from exc

        # Respect a backend configured programmatically. Only fall back to the
        # official MSI paths after the current phonemizer configuration fails.
        fallback_error: RuntimeError = exc
        for candidate in _windows_espeak_library_candidates():
            if not os.path.exists(candidate):
                continue
            EspeakBackend.set_library(candidate)
            try:
                version = EspeakBackend.version()
                break
            except RuntimeError as candidate_exc:
                fallback_error = candidate_exc
        else:
            raise RuntimeError(
                f"{_ESPEAK_INSTALL_HELP}\nUnderlying library error: {fallback_error}"
            ) from fallback_error

    if version < (1, 49):
        version_text = ".".join(str(part) for part in version)
        raise RuntimeError(
            "Legacy eSpeak was found, but corpusgen requires espeak-ng "
            f"(detected version {version_text}).\n{_ESPEAK_INSTALL_HELP}"
        )


# Separator config: space between phonemes, | between words, newline between utterances
_PHONEME_SEP = Separator(phone=" ", word="|", syllable="")


def _parse_phonemes(ipa_raw: str) -> list[str]:
    """Parse an IPA string with space-separated phones into a phoneme list.

    Strips stress marks and word boundary markers, filters empty tokens.
    """
    # Remove word boundary markers
    cleaned = ipa_raw.replace("|", " ")
    # Remove primary/secondary stress marks — keep the phoneme segments clean
    cleaned = cleaned.replace("ˈ", "").replace("ˌ", "")
    # Split on whitespace and filter empties
    tokens = [t.strip() for t in cleaned.split() if t.strip()]
    return tokens


def _map_espeak_language(language: str) -> str:
    """Map user-friendly language codes to espeak-ng voice names.

    Accepts codes like 'en-us', 'en', 'fr-fr', 'fr', 'ar', 'bn', etc.
    """
    # espeak-ng uses codes like 'en-us', 'fr-fr', 'ar', 'bn'
    # Normalize: lowercase, replace _ with -
    lang = language.lower().replace("_", "-")
    return lang


class G2PManager:
    """Manages grapheme-to-phoneme conversion with configurable backends.

    Currently supports:
        - "espeak": espeak-ng via the phonemizer library (100+ languages)

    Future backends (Phase 5+):
        - "neural": ByT5/CharsiuG2P transformer model
        - "epitran": Rule-based Epitran backend

    Args:
        backend: Which G2P backend to use. Default: "espeak".
    """

    def __init__(self, backend: str = "espeak") -> None:
        if backend not in ("espeak",):
            raise ValueError(f"Unsupported backend: {backend!r}. Supported: 'espeak'")
        self._backend_name = backend
        if backend == "espeak":
            _check_espeak_available()
        # Cache EspeakBackend instances per language
        self._backends: dict[str, EspeakBackend] = {}

    @property
    def backend(self) -> str:
        """Currently active backend name."""
        return self._backend_name

    def _get_espeak_backend(self, language: str) -> EspeakBackend:
        """Get or create a cached EspeakBackend for the given language."""
        lang = _map_espeak_language(language)
        if lang not in self._backends:
            self._backends[lang] = EspeakBackend(lang, with_stress=False)
        return self._backends[lang]

    def phonemize(self, text: str, language: str = "en-us") -> G2PResult:
        """Convert text to phonemes.

        Args:
            text: Input text (word or sentence).
            language: Language code (e.g., 'en-us', 'fr-fr', 'ar').

        Returns:
            G2PResult with IPA transcription and parsed phonemes.
        """
        if not text or not text.strip():
            return G2PResult(text=text, ipa="", phonemes=[], language=language)

        backend = self._get_espeak_backend(language)
        # phonemizer expects a list and returns a list
        ipa_list = backend.phonemize([text], separator=_PHONEME_SEP, strip=True)
        ipa_raw = ipa_list[0] if ipa_list else ""

        phonemes = _parse_phonemes(ipa_raw)

        # Build a clean IPA string (without separator artifacts)
        ipa_clean = ipa_raw.replace("|", " ").strip()
        # Collapse multiple spaces
        ipa_clean = re.sub(r"\s+", " ", ipa_clean)

        return G2PResult(
            text=text,
            ipa=ipa_clean,
            phonemes=phonemes,
            language=language,
        )

    def phonemize_batch(
        self, texts: list[str], language: str = "en-us"
    ) -> list[G2PResult]:
        """Phonemize multiple texts efficiently.

        Pre-filters empty/whitespace-only strings to avoid misalignment,
        since phonemizer may drop empty utterances from its output.

        Args:
            texts: List of input texts.
            language: Language code.

        Returns:
            List of G2PResult, one per input text (order preserved).
        """
        if not texts:
            return []

        # Separate non-empty texts from empty ones to avoid misalignment.
        # phonemizer can drop empty utterances, breaking naive zip.
        non_empty_indices: list[int] = []
        non_empty_texts: list[str] = []
        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_indices.append(i)
                non_empty_texts.append(text)

        # Phonemize only non-empty texts
        ipa_map: dict[int, str] = {}
        if non_empty_texts:
            backend = self._get_espeak_backend(language)
            ipa_list = backend.phonemize(
                non_empty_texts, separator=_PHONEME_SEP, strip=True
            )
            for idx, ipa_raw in zip(non_empty_indices, ipa_list):
                ipa_map[idx] = ipa_raw

        # Reassemble results in original order
        results: list[G2PResult] = []
        for i, text in enumerate(texts):
            if i in ipa_map:
                ipa_raw = ipa_map[i]
                phonemes = _parse_phonemes(ipa_raw)
                ipa_clean = re.sub(r"\s+", " ", ipa_raw.replace("|", " ").strip())
                results.append(
                    G2PResult(
                        text=text, ipa=ipa_clean, phonemes=phonemes, language=language
                    )
                )
            else:
                results.append(
                    G2PResult(text=text, ipa="", phonemes=[], language=language)
                )

        return results

    def phonemize_variants(
        self, text: str, language: str = "en-us"
    ) -> list[G2PResult]:
        """Get pronunciation variants for a word/text.

        Uses espeak-ng's built-in variant support. For espeak, this typically
        returns one canonical pronunciation. Future backends (neural, dictionary)
        will provide richer variant support.

        Args:
            text: Input text.
            language: Language code.

        Returns:
            List of G2PResult, one per pronunciation variant.
        """
        # espeak-ng produces one canonical pronunciation per language
        # For multi-dialect, we can query multiple language variants
        primary = self.phonemize(text, language=language)

        variants = [primary]

        # If language has known dialect variants, query those too
        dialect_map: dict[str, list[str]] = {
            "en": ["en-us", "en-gb"],
            "en-us": ["en-gb"],
            "en-gb": ["en-us"],
            "fr": ["fr-fr", "fr-be"],
            "fr-fr": ["fr-be"],
            "pt": ["pt-br"],
            "pt-br": ["pt"],
            "es": ["es-la"],
            "es-la": ["es"],
        }

        base_lang = language.lower().replace("_", "-")
        if base_lang in dialect_map:
            for dialect in dialect_map[base_lang]:
                try:
                    variant = self.phonemize(text, language=dialect)
                    # Only add if pronunciation actually differs
                    if variant.ipa and variant.ipa != primary.ipa:
                        variants.append(variant)
                except Exception:
                    # Dialect not available in espeak-ng, skip
                    continue

        return variants

    def supported_languages(self) -> list[str]:
        """List languages supported by the current backend.

        Returns:
            Sorted list of language codes (e.g., 'en-us', 'fr-fr').
        """
        try:
            # Use phonemizer's API — reliable across platforms
            lang_dict = EspeakBackend.supported_languages()
            return sorted(lang_dict.keys())
        except Exception:
            # Fallback: return languages we know espeak-ng supports
            return ["en-us", "en-gb", "fr-fr", "de", "es", "ar", "bn", "zh"]
