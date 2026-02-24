"""Shared test fixtures for corpusgen."""

import pytest


@pytest.fixture
def sample_english_sentences() -> list[str]:
    """A small set of English sentences for testing."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
        "How much wood would a woodchuck chuck?",
        "Peter Piper picked a peck of pickled peppers.",
        "The treasure was buried beneath the old church.",
    ]


@pytest.fixture
def sample_english_phonemes() -> list[str]:
    """A subset of English phonemes (IPA) for testing."""
    return [
        "p", "b", "t", "d", "k", "ɡ",          # stops
        "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ",  # fricatives
        "tʃ", "dʒ",                               # affricates
        "m", "n", "ŋ",                            # nasals
        "l", "ɹ", "j", "w",                       # approximants
        "h",                                       # glottal
        "iː", "ɪ", "eɪ", "ɛ", "æ",              # vowels (subset)
        "ɑː", "ɒ", "ɔː", "oʊ", "ʊ", "uː",
        "ʌ", "ə", "ɜː",
        "aɪ", "aʊ", "ɔɪ",                        # diphthongs
    ]
