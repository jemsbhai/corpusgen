"""Error rate metrics for speech corpus evaluation.

Computes standard error rates used in ASR and TTS evaluation by
comparing reference and hypothesis sequences at word, character,
and phoneme levels.  All metrics are built on Levenshtein edit
distance (minimum substitutions + deletions + insertions).

Metrics provided:

* **WER** (Word Error Rate) — ``(S + D + I) / N`` at word level.
  The standard ASR evaluation metric.
* **CER** (Character Error Rate) — same formula at character level.
  More granular than WER, useful for morphologically rich languages.
* **PER** (Phoneme Error Rate) — same formula at phoneme level.
  Requires pre-phonemized reference and hypothesis sequences.
* **SER** (Sentence Error Rate) — fraction of sentences with any
  word-level error.  Measures how often entire utterances are correct.

The corpus-level ``compute_error_rates()`` function aggregates across
all sentences and provides per-sentence breakdowns.

References:
    Morris, A. C., Maier, V., & Green, P. D. (2004). From WER and
        RIL to MER and WIL: improved evaluation measures for connected
        speech recognition. INTERSPEECH.
    McCowan, I., et al. (2005). On the use of information retrieval
        measures for speech recognition evaluation. IDIAP-RR 73.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# Edit distance
# ---------------------------------------------------------------------------


def edit_distance(ref: Sequence, hyp: Sequence) -> int:
    """Compute Levenshtein edit distance between two sequences.

    Works with any indexable sequences (lists, strings, tuples).
    Uses the standard dynamic-programming algorithm in O(n*m) time
    and O(min(n, m)) space.

    Args:
        ref: Reference (ground truth) sequence.
        hyp: Hypothesis (predicted) sequence.

    Returns:
        Minimum number of substitutions, deletions, and insertions.
    """
    n = len(ref)
    m = len(hyp)

    # Optimise space: iterate over the shorter dimension
    if n < m:
        return edit_distance(hyp, ref)

    # prev and curr are 1-D arrays of length (m + 1)
    prev = list(range(m + 1))
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(
                    prev[j],      # deletion
                    curr[j - 1],  # insertion
                    prev[j - 1],  # substitution
                )
        prev, curr = curr, prev

    return prev[m]


# ---------------------------------------------------------------------------
# Individual error rates
# ---------------------------------------------------------------------------


def word_error_rate(
    reference: str,
    hypothesis: str,
    case_sensitive: bool = False,
) -> float:
    """Compute Word Error Rate between two text strings.

    Tokenises on whitespace after optional lowercasing.

    Args:
        reference: Reference text.
        hypothesis: Hypothesis text.
        case_sensitive: If False (default), comparison is case-insensitive.

    Returns:
        WER as a float.  0.0 for perfect match.  ``float('inf')`` if
        the reference is empty but hypothesis is not.
    """
    if not case_sensitive:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else float("inf")

    dist = edit_distance(ref_words, hyp_words)
    return dist / len(ref_words)


def character_error_rate(
    reference: str,
    hypothesis: str,
) -> float:
    """Compute Character Error Rate between two text strings.

    Operates on raw characters (including spaces).  Strips nothing.

    Args:
        reference: Reference text.
        hypothesis: Hypothesis text.

    Returns:
        CER as a float.  ``float('inf')`` if reference is empty
        but hypothesis is not.
    """
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else float("inf")

    dist = edit_distance(reference, hypothesis)
    return dist / len(reference)


def phoneme_error_rate(
    reference: list[str],
    hypothesis: list[str],
) -> float:
    """Compute Phoneme Error Rate between two phoneme sequences.

    Args:
        reference: Reference phoneme list.
        hypothesis: Hypothesis phoneme list.

    Returns:
        PER as a float.  ``float('inf')`` if reference is empty
        but hypothesis is not.
    """
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else float("inf")

    dist = edit_distance(reference, hypothesis)
    return dist / len(reference)


def sentence_error_rate(
    references: list[str],
    hypotheses: list[str],
    case_sensitive: bool = False,
) -> float:
    """Compute Sentence Error Rate: fraction of sentences with any error.

    A sentence is counted as erroneous if its normalised word sequences
    differ at all.

    Args:
        references: List of reference texts.
        hypotheses: List of hypothesis texts (same length).
        case_sensitive: If False, comparison is case-insensitive.

    Returns:
        SER as a float in [0.0, 1.0].  0.0 if all match.
    """
    if len(references) == 0:
        return 0.0

    errors = 0
    for ref, hyp in zip(references, hypotheses):
        r = ref if case_sensitive else ref.lower()
        h = hyp if case_sensitive else hyp.lower()
        if r.split() != h.split():
            errors += 1

    return errors / len(references)


# ---------------------------------------------------------------------------
# Structured results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SentenceErrorDetail:
    """Per-sentence error breakdown.

    Attributes:
        index: Sentence position in the corpus.
        reference: Reference text.
        hypothesis: Hypothesis text.
        wer: Word Error Rate for this sentence.
        cer: Character Error Rate for this sentence.
        per: Phoneme Error Rate (None if phonemes not provided).
    """

    index: int
    reference: str
    hypothesis: str
    wer: float
    cer: float
    per: float | None


@dataclass(frozen=True)
class ErrorRateResult:
    """Corpus-level error rate metrics with per-sentence details.

    Corpus-level WER, CER, and PER are computed as total edit
    distance divided by total reference length (micro-average),
    not as the mean of per-sentence rates.  This is the standard
    convention in ASR evaluation.

    Attributes:
        wer: Corpus-level Word Error Rate (micro-averaged).
        cer: Corpus-level Character Error Rate (micro-averaged).
        per: Corpus-level Phoneme Error Rate (None if no phonemes).
        ser: Sentence Error Rate.
        details: Per-sentence breakdowns.
    """

    wer: float
    cer: float
    per: float | None
    ser: float
    details: list[SentenceErrorDetail]

    def to_dict(self) -> dict[str, Any]:
        """Export as a plain Python dict (JSON-safe)."""
        return {
            "wer": self.wer,
            "cer": self.cer,
            "per": self.per,
            "ser": self.ser,
            "details": [asdict(d) for d in self.details],
        }


# ---------------------------------------------------------------------------
# Corpus-level aggregation
# ---------------------------------------------------------------------------


def compute_error_rates(
    references: list[str],
    hypotheses: list[str],
    reference_phonemes: list[list[str]] | None = None,
    hypothesis_phonemes: list[list[str]] | None = None,
    case_sensitive: bool = False,
) -> ErrorRateResult:
    """Compute corpus-level error rates with per-sentence details.

    Corpus-level WER and CER are micro-averaged: total edit distance
    across all sentences divided by total reference tokens.  This is
    the standard convention in ASR evaluation (not macro-average of
    per-sentence rates).

    Args:
        references: List of reference text strings.
        hypotheses: List of hypothesis text strings (same length).
        reference_phonemes: Optional phoneme lists for each reference
            sentence.  Required (along with hypothesis_phonemes) to
            compute PER.
        hypothesis_phonemes: Optional phoneme lists for each hypothesis
            sentence.
        case_sensitive: If False (default), word/sentence comparisons
            are case-insensitive.

    Returns:
        ErrorRateResult with corpus-level and per-sentence metrics.

    Raises:
        ValueError: If references and hypotheses have different lengths,
            or if phoneme lists are provided with mismatched lengths.
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"references and hypotheses must have same length, "
            f"got {len(references)} and {len(hypotheses)}"
        )

    has_phonemes = (
        reference_phonemes is not None and hypothesis_phonemes is not None
    )
    if has_phonemes:
        if len(reference_phonemes) != len(references):
            raise ValueError(
                f"reference_phonemes must have same length as references, "
                f"got {len(reference_phonemes)} and {len(references)}"
            )
        if len(hypothesis_phonemes) != len(hypotheses):
            raise ValueError(
                f"hypothesis_phonemes must have same length as hypotheses, "
                f"got {len(hypothesis_phonemes)} and {len(hypotheses)}"
            )

    n = len(references)

    if n == 0:
        return ErrorRateResult(
            wer=0.0,
            cer=0.0,
            per=None,
            ser=0.0,
            details=[],
        )

    # Accumulators for micro-average
    total_word_edits = 0
    total_word_ref_len = 0
    total_char_edits = 0
    total_char_ref_len = 0
    total_phoneme_edits = 0
    total_phoneme_ref_len = 0

    details: list[SentenceErrorDetail] = []

    for i in range(n):
        ref_text = references[i]
        hyp_text = hypotheses[i]

        # Word level
        ref_words = (ref_text if case_sensitive else ref_text.lower()).split()
        hyp_words = (hyp_text if case_sensitive else hyp_text.lower()).split()
        w_edits = edit_distance(ref_words, hyp_words)
        w_ref_len = len(ref_words)
        total_word_edits += w_edits
        total_word_ref_len += w_ref_len
        sent_wer = w_edits / w_ref_len if w_ref_len > 0 else (
            0.0 if len(hyp_words) == 0 else float("inf")
        )

        # Character level
        c_edits = edit_distance(ref_text, hyp_text)
        c_ref_len = len(ref_text)
        total_char_edits += c_edits
        total_char_ref_len += c_ref_len
        sent_cer = c_edits / c_ref_len if c_ref_len > 0 else (
            0.0 if len(hyp_text) == 0 else float("inf")
        )

        # Phoneme level
        sent_per: float | None = None
        if has_phonemes:
            ref_ph = reference_phonemes[i]
            hyp_ph = hypothesis_phonemes[i]
            p_edits = edit_distance(ref_ph, hyp_ph)
            p_ref_len = len(ref_ph)
            total_phoneme_edits += p_edits
            total_phoneme_ref_len += p_ref_len
            sent_per = p_edits / p_ref_len if p_ref_len > 0 else (
                0.0 if len(hyp_ph) == 0 else float("inf")
            )

        details.append(SentenceErrorDetail(
            index=i,
            reference=ref_text,
            hypothesis=hyp_text,
            wer=sent_wer,
            cer=sent_cer,
            per=sent_per,
        ))

    # Corpus-level micro-averages
    corpus_wer = (
        total_word_edits / total_word_ref_len
        if total_word_ref_len > 0 else 0.0
    )
    corpus_cer = (
        total_char_edits / total_char_ref_len
        if total_char_ref_len > 0 else 0.0
    )
    corpus_per: float | None = None
    if has_phonemes:
        corpus_per = (
            total_phoneme_edits / total_phoneme_ref_len
            if total_phoneme_ref_len > 0 else 0.0
        )

    corpus_ser = sentence_error_rate(
        references, hypotheses, case_sensitive=case_sensitive
    )

    return ErrorRateResult(
        wer=corpus_wer,
        cer=corpus_cer,
        per=corpus_per,
        ser=corpus_ser,
        details=details,
    )
