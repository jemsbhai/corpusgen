"""Text-level corpus quality metrics.

Computes linguistic statistics about the generated text itself —
sentence length, vocabulary diversity, and readability — independent
of phoneme coverage.  These metrics are essential for ensuring that
a corpus optimised for phonetic coverage has not sacrificed text
naturalness.

Metrics provided:

* **Sentence length** (words and phonemes): mean, median, std, min, max.
* **Vocabulary**: total words, unique words, type-token ratio (TTR),
  hapax ratio (fraction of words appearing exactly once).
* **Readability** (Flesch Reading Ease, Flesch-Kincaid Grade Level):
  computed when syllable counting is possible (Latin-script text).
  Returns ``None`` for non-Latin scripts where syllable heuristics
  do not apply.

Word tokenization is Unicode-aware, strips punctuation, and preserves
hyphenated words and contractions.  It lowercases for consistency.

References:
    Flesch, R. (1948). A new readability yardstick. Journal of Applied
        Psychology, 32(3), 221–233.
    Kincaid, J. P., et al. (1975). Derivation of new readability
        formulas for Navy enlisted personnel. (FK Grade Level)
"""

from __future__ import annotations

import math
import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any


# ---------------------------------------------------------------------------
# Word tokenizer
# ---------------------------------------------------------------------------

# Matches sequences of: word characters (including Unicode letters/digits),
# hyphens between word chars, and apostrophes within words.
_WORD_RE = re.compile(r"[\w]+(?:[-'][\w]+)*", re.UNICODE)


def tokenize_words(text: str) -> list[str]:
    """Tokenize text into lowercased words, stripping punctuation.

    Unicode-aware.  Preserves hyphenated words (e.g., "well-known")
    and contractions (e.g., "don't").  Numbers are kept as tokens.

    Args:
        text: Input text string.

    Returns:
        List of lowercased word tokens.
    """
    return [m.group().lower() for m in _WORD_RE.finditer(text)]


# ---------------------------------------------------------------------------
# Syllable counting (English / Latin-script heuristic)
# ---------------------------------------------------------------------------

_VOWELS = set("aeiouyAEIOUY")


def _count_syllables_word(word: str) -> int:
    """Estimate syllable count for a single word.

    Uses a simple English heuristic: count vowel groups, adjust for
    silent-e and common patterns.  Not accurate for non-English words
    but sufficient for readability formulas.

    Args:
        word: A single word (letters only expected).

    Returns:
        Estimated syllable count (minimum 1).
    """
    word = word.lower().strip()
    if not word:
        return 0

    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in _VOWELS
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel

    # Silent-e adjustment
    if word.endswith("e") and count > 1:
        count -= 1

    # -le ending is a syllable (e.g., "table")
    if word.endswith("le") and len(word) > 2 and word[-3] not in _VOWELS:
        count += 1

    return max(count, 1)


def _is_latin_script(text: str) -> bool:
    """Check whether the text is predominantly Latin script.

    Looks at letter characters only; returns True if >50% are Latin.

    Args:
        text: Input text.

    Returns:
        True if predominantly Latin script.
    """
    latin_count = 0
    letter_count = 0
    for ch in text:
        if unicodedata.category(ch).startswith("L"):
            letter_count += 1
            # Latin script characters are in Unicode blocks starting with
            # "LATIN" — we check the character name.
            try:
                name = unicodedata.name(ch, "")
                if name.startswith("LATIN") or ch.isascii():
                    latin_count += 1
            except ValueError:
                pass

    if letter_count == 0:
        return False
    return latin_count / letter_count > 0.5


# ---------------------------------------------------------------------------
# TextQualityMetrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextQualityMetrics:
    """Text-level quality metrics for a corpus.

    Attributes:
        sentence_length_words_mean: Mean sentence length in words.
        sentence_length_words_median: Median sentence length in words.
        sentence_length_words_std: Population std of sentence lengths (words).
        sentence_length_words_min: Shortest sentence (words).
        sentence_length_words_max: Longest sentence (words).
        sentence_length_phonemes_mean: Mean sentence length in phonemes.
        sentence_length_phonemes_median: Median in phonemes.
        sentence_length_phonemes_std: Population std (phonemes).
        sentence_length_phonemes_min: Shortest sentence (phonemes).
        sentence_length_phonemes_max: Longest sentence (phonemes).
        total_words: Total word tokens across all sentences.
        unique_words: Number of distinct word types.
        type_token_ratio: unique_words / total_words (0.0 if no words).
        hapax_ratio: Words appearing exactly once / unique_words.
        flesch_reading_ease: Flesch Reading Ease score, or None if not
            computable (non-Latin script or empty corpus).
        flesch_kincaid_grade: Flesch-Kincaid Grade Level, or None.
    """

    sentence_length_words_mean: float
    sentence_length_words_median: float
    sentence_length_words_std: float
    sentence_length_words_min: int
    sentence_length_words_max: int
    sentence_length_phonemes_mean: float
    sentence_length_phonemes_median: float
    sentence_length_phonemes_std: float
    sentence_length_phonemes_min: int
    sentence_length_phonemes_max: int
    total_words: int
    unique_words: int
    type_token_ratio: float
    hapax_ratio: float
    flesch_reading_ease: float | None
    flesch_kincaid_grade: float | None

    def to_dict(self) -> dict[str, Any]:
        """Export as a plain Python dict (JSON-safe)."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_text_quality_metrics(
    sentences: list[str],
    phoneme_sequences: list[list[str]],
) -> TextQualityMetrics:
    """Compute text-level quality metrics for a corpus.

    Args:
        sentences: List of text sentences.
        phoneme_sequences: Phoneme lists for each sentence (same order
            and length as ``sentences``).

    Returns:
        TextQualityMetrics with all computed fields.
    """
    n = len(sentences)

    # --- Tokenize all sentences ---
    tokenized = [tokenize_words(s) for s in sentences]
    word_lengths = [len(toks) for toks in tokenized]
    phoneme_lengths = [len(seq) for seq in phoneme_sequences]

    # --- Sentence length stats (words) ---
    if n == 0:
        wl_mean = wl_median = wl_std = 0.0
        wl_min = wl_max = 0
    else:
        wl_mean = statistics.mean(word_lengths)
        wl_median = statistics.median(word_lengths)
        wl_std = statistics.pstdev(word_lengths)
        wl_min = min(word_lengths)
        wl_max = max(word_lengths)

    # --- Sentence length stats (phonemes) ---
    if n == 0:
        pl_mean = pl_median = pl_std = 0.0
        pl_min = pl_max = 0
    else:
        pl_mean = statistics.mean(phoneme_lengths)
        pl_median = statistics.median(phoneme_lengths)
        pl_std = statistics.pstdev(phoneme_lengths)
        pl_min = min(phoneme_lengths)
        pl_max = max(phoneme_lengths)

    # --- Vocabulary stats ---
    all_words: list[str] = []
    for toks in tokenized:
        all_words.extend(toks)

    total_words = len(all_words)
    word_counts = Counter(all_words)
    unique_words = len(word_counts)

    ttr = unique_words / total_words if total_words > 0 else 0.0

    hapax_count = sum(1 for c in word_counts.values() if c == 1)
    hapax_ratio = hapax_count / unique_words if unique_words > 0 else 0.0

    # --- Readability (Flesch) ---
    flesch_re: float | None = None
    flesch_fk: float | None = None

    full_text = " ".join(sentences)
    if n > 0 and total_words > 0 and _is_latin_script(full_text):
        # Count total syllables
        total_syllables = sum(
            _count_syllables_word(w) for w in all_words
        )

        num_sentences = n
        avg_sentence_length = total_words / num_sentences
        avg_syllables_per_word = total_syllables / total_words

        # Flesch Reading Ease
        # RE = 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
        flesch_re = (
            206.835
            - 1.015 * avg_sentence_length
            - 84.6 * avg_syllables_per_word
        )

        # Flesch-Kincaid Grade Level
        # GL = 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
        flesch_fk = (
            0.39 * avg_sentence_length
            + 11.8 * avg_syllables_per_word
            - 15.59
        )

    return TextQualityMetrics(
        sentence_length_words_mean=wl_mean,
        sentence_length_words_median=wl_median,
        sentence_length_words_std=wl_std,
        sentence_length_words_min=wl_min,
        sentence_length_words_max=wl_max,
        sentence_length_phonemes_mean=pl_mean,
        sentence_length_phonemes_median=pl_median,
        sentence_length_phonemes_std=pl_std,
        sentence_length_phonemes_min=pl_min,
        sentence_length_phonemes_max=pl_max,
        total_words=total_words,
        unique_words=unique_words,
        type_token_ratio=ttr,
        hapax_ratio=hapax_ratio,
        flesch_reading_ease=flesch_re,
        flesch_kincaid_grade=flesch_fk,
    )
