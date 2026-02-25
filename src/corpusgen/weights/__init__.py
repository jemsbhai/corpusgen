"""Phoneme weighting strategies for coverage-based selection.

Provides functions that produce ``dict[str, float]`` mappings from
phonetic units to weights. These weights modulate marginal gain
calculations in selection algorithms, allowing users to prioritise
rare, linguistically important, or otherwise high-value units.
"""

from __future__ import annotations

import math
from collections import Counter


def uniform_weights(target_units: set[str]) -> dict[str, float]:
    """Assign equal weight of 1.0 to every target unit.

    This is the default when no weighting is specified and is equivalent
    to standard unweighted Set Cover.

    Args:
        target_units: Set of phonetic units.

    Returns:
        Mapping from each unit to 1.0.
    """
    return {u: 1.0 for u in target_units}


def frequency_inverse_weights(
    target_units: set[str],
    corpus_phonemes: list[list[str]],
) -> dict[str, float]:
    """Weight units inversely proportional to their corpus frequency.

    Rare phonemes receive higher weights, ensuring they are prioritised
    during selection. Uses smoothed inverse frequency: weight(u) =
    log(N / (count(u) + 1)) where N is total token count.

    Weights are normalised so they average to 1.0 (sum = len(target)).

    Args:
        target_units: Set of phonetic units to weight.
        corpus_phonemes: List of phoneme lists from the candidate corpus.

    Returns:
        Mapping from each unit to its inverse-frequency weight.
    """
    if not target_units:
        return {}

    # Count occurrences across the corpus
    counts: Counter[str] = Counter()
    for phonemes in corpus_phonemes:
        counts.update(phonemes)

    total = sum(counts.values()) if counts else 1

    # Smoothed inverse frequency
    raw: dict[str, float] = {}
    for u in target_units:
        freq = counts.get(u, 0)
        raw[u] = math.log(total / (freq + 1)) + 1.0  # +1 ensures min weight > 0

    # Normalise so weights average to 1.0
    avg = sum(raw.values()) / len(raw) if raw else 1.0
    if avg == 0:
        return {u: 1.0 for u in target_units}
    return {u: (v / avg) for u, v in raw.items()}


def linguistic_class_weights(
    target_units: set[str],
    class_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Weight units by their phonological class (vowel, consonant, etc.).

    Uses panphon to classify IPA segments. Multi-segment units (diphones,
    triphones) are not classified and receive the default weight.

    Args:
        target_units: Set of phonetic units to weight.
        class_weights: Mapping from class name to weight. Recognised classes:
            ``"vowel"``, ``"consonant"``. Defaults to
            ``{"vowel": 1.5, "consonant": 1.0}`` — slightly boosting
            vowels since they are fewer and phonologically critical.

    Returns:
        Mapping from each unit to its class-based weight.
    """
    if not target_units:
        return {}

    if class_weights is None:
        class_weights = {"vowel": 1.5, "consonant": 1.0}

    default_weight = 1.0

    weights: dict[str, float] = {}
    for u in target_units:
        # Skip multi-segment units (contain '-' separator)
        if "-" in u:
            weights[u] = default_weight
            continue

        cls = _classify_ipa_segment(u)
        weights[u] = class_weights.get(cls, default_weight)

    return weights


# IPA vowel characters (monophthongs and common diacriticised forms)
_IPA_VOWELS = set(
    "iyɨʉɯuɪʏʊeøɘɵɤoəɛœɜɞʌɔæɐaɶɑɒ"
)


def _classify_ipa_segment(segment: str) -> str:
    """Classify an IPA segment as 'vowel' or 'consonant'.

    Uses the base character (stripping combining diacritics and
    modifiers) to determine class. Falls back to 'unknown' if
    the segment cannot be classified.
    """
    if not segment:
        return "unknown"

    # Use the first base character (skip combining diacritics)
    import unicodedata

    for ch in segment:
        cat = unicodedata.category(ch)
        # Skip combining marks (Mn, Mc, Me) and modifier letters (Lm)
        if cat.startswith("M") or cat == "Lm":
            continue
        if ch in _IPA_VOWELS:
            return "vowel"
        # If it's a letter-like character but not a vowel → consonant
        if cat.startswith("L"):
            return "consonant"

    return "unknown"
