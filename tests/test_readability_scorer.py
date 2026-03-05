"""Tests for corpusgen.generate.scorers.readability — readability scorer.

Test tiers:
    - **Fast tests** (default): Pure Python, no model loading.

Scientific contract:
    - Simple mode: score = clamp(FRE / 100, 0, 1)
    - Target-range mode: trapezoidal — 1.0 inside [lo, hi],
      linear decay to 0 at FRE=0 (below) and FRE=100 (above)
    - Returns 0.0 for None, empty, whitespace, or non-Latin text
    - compute_fre() returns raw Flesch Reading Ease
    - as_filter() returns a candidate filter callable for GenerationLoop

Mathematical reference (Flesch Reading Ease, single sentence):
    FRE = 206.835 - 1.015 * total_words - 84.6 * (total_syllables / total_words)

Hand-computed test case:
    "She sells seashells by the seashore on a sunny day"
    Words: 10, Syllables: 1+1+2+1+1+2+1+1+2+1 = 13
    FRE = 206.835 - 1.015*10 - 84.6*(13/10)
        = 206.835 - 10.15 - 109.98 = 86.705
"""

from __future__ import annotations

import pytest

from corpusgen.generate.scorers.readability import ReadabilityScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _approx(value: float, abs_tol: float = 1e-3) -> pytest.approx:
    """Shorthand for pytest.approx with tolerance suitable for FRE-derived scores."""
    return pytest.approx(value, abs=abs_tol)


# Known test sentence:
#   "She sells seashells by the seashore on a sunny day"
#   10 words, 13 syllables → FRE = 86.705
_KNOWN_TEXT = "She sells seashells by the seashore on a sunny day"
_KNOWN_FRE = 86.705

# Short, simple sentence:
#   "The cat sat" → 3 words, 3 syllables
#   FRE = 206.835 - 1.015*3 - 84.6*(3/3) = 206.835 - 3.045 - 84.6 = 119.19
_SIMPLE_TEXT = "The cat sat"
_SIMPLE_FRE = 119.19


# ---------------------------------------------------------------------------
# 1. Construction and validation
# ---------------------------------------------------------------------------


class TestReadabilityScorerConstruction:
    """Test ReadabilityScorer initialization."""

    def test_simple_mode_default(self):
        """Default construction should be simple mode (no target range)."""
        scorer = ReadabilityScorer()
        assert scorer.target_range is None

    def test_target_range_mode(self):
        """Construction with target_range stores the range."""
        scorer = ReadabilityScorer(target_range=(60, 80))
        assert scorer.target_range == (60, 80)

    def test_invalid_range_lo_gt_hi_raises(self):
        """target_range lo > hi should raise ValueError."""
        with pytest.raises(ValueError, match="[Rr]ange"):
            ReadabilityScorer(target_range=(80, 60))

    def test_invalid_range_negative_raises(self):
        """Negative range values should raise ValueError."""
        with pytest.raises(ValueError, match="[Rr]ange"):
            ReadabilityScorer(target_range=(-10, 80))


# ---------------------------------------------------------------------------
# 2. compute_fre — raw Flesch Reading Ease
# ---------------------------------------------------------------------------


class TestComputeFre:
    """Test raw FRE computation."""

    def test_known_sentence(self):
        """FRE for known test sentence should match hand computation."""
        scorer = ReadabilityScorer()
        fre = scorer.compute_fre(_KNOWN_TEXT)
        assert fre == _approx(_KNOWN_FRE)

    def test_simple_sentence(self):
        """Very simple sentence should have high FRE."""
        scorer = ReadabilityScorer()
        fre = scorer.compute_fre(_SIMPLE_TEXT)
        assert fre == _approx(_SIMPLE_FRE)

    def test_none_returns_none(self):
        """None input should return None."""
        scorer = ReadabilityScorer()
        assert scorer.compute_fre(None) is None

    def test_empty_returns_none(self):
        """Empty string should return None."""
        scorer = ReadabilityScorer()
        assert scorer.compute_fre("") is None

    def test_whitespace_returns_none(self):
        """Whitespace-only should return None."""
        scorer = ReadabilityScorer()
        assert scorer.compute_fre("   \n\t  ") is None

    def test_non_latin_returns_none(self):
        """Non-Latin text should return None (syllable heuristic doesn't apply)."""
        scorer = ReadabilityScorer()
        assert scorer.compute_fre("\u3053\u3093\u306b\u3061\u306f\u4e16\u754c") is None


# ---------------------------------------------------------------------------
# 3. Simple mode scoring: score = clamp(FRE / 100, 0, 1)
# ---------------------------------------------------------------------------


class TestSimpleModeScoring:
    """Test simple mode (no target range) scoring."""

    def test_known_sentence_score(self):
        """Score for known sentence: clamp(86.705 / 100, 0, 1) = 0.86705."""
        scorer = ReadabilityScorer()
        score = scorer(_KNOWN_TEXT)
        assert score == _approx(0.86705)

    def test_very_simple_text_clamped_to_one(self):
        """FRE > 100 should clamp score to 1.0.

        "The cat sat" → FRE = 119.19 → clamp(1.1919, 0, 1) = 1.0.
        """
        scorer = ReadabilityScorer()
        score = scorer(_SIMPLE_TEXT)
        assert score == _approx(1.0)

    def test_none_returns_zero(self):
        """None should return 0.0."""
        scorer = ReadabilityScorer()
        assert scorer(None) == 0.0

    def test_empty_returns_zero(self):
        """Empty string should return 0.0."""
        scorer = ReadabilityScorer()
        assert scorer("") == 0.0

    def test_whitespace_returns_zero(self):
        """Whitespace-only should return 0.0."""
        scorer = ReadabilityScorer()
        assert scorer("   ") == 0.0

    def test_non_latin_returns_zero(self):
        """Non-Latin text should return 0.0."""
        scorer = ReadabilityScorer()
        assert scorer("\u3053\u3093\u306b\u3061\u306f") == 0.0

    def test_score_in_zero_one(self):
        """Score must always be in [0, 1]."""
        scorer = ReadabilityScorer()
        score = scorer(_KNOWN_TEXT)
        assert 0.0 <= score <= 1.0

    def test_callable_interface(self):
        """Scorer should work as callable for PhoneticScorer hook."""
        scorer = ReadabilityScorer()
        result = scorer("Hello world.")
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# 4. Target-range mode scoring
#    Trapezoidal: 1.0 in [lo, hi], linear decay outside.
#    Below lo: score = FRE / lo  (0 at FRE=0, 1.0 at FRE=lo)
#    Above hi (hi < 100): score = (100 - FRE) / (100 - hi)  (1.0 at hi, 0 at FRE=100)
# ---------------------------------------------------------------------------


class TestTargetRangeModeScoring:
    """Test target-range mode scoring with trapezoidal mapping."""

    def test_inside_range_scores_one(self):
        """FRE inside [lo, hi] should score 1.0.

        target_range=(80, 90), FRE=86.705 is inside -> 1.0.
        """
        scorer = ReadabilityScorer(target_range=(80, 90))
        score = scorer(_KNOWN_TEXT)
        assert score == _approx(1.0)

    def test_above_range_decays(self):
        """FRE above hi should decay linearly toward 0 at FRE=100.

        target_range=(60, 80), FRE=86.705.
        score = (100 - 86.705) / (100 - 80) = 13.295 / 20 = 0.66475.
        """
        scorer = ReadabilityScorer(target_range=(60, 80))
        score = scorer(_KNOWN_TEXT)
        expected = (100 - _KNOWN_FRE) / (100 - 80)
        assert score == _approx(expected)

    def test_below_range_decays(self):
        """FRE below lo should decay linearly toward 0 at FRE=0.

        To test this, we need text with FRE < lo.
        target_range=(90, 100), FRE=86.705 (below lo=90).
        score = 86.705 / 90 = 0.96339.
        """
        scorer = ReadabilityScorer(target_range=(90, 100))
        score = scorer(_KNOWN_TEXT)
        expected = _KNOWN_FRE / 90
        assert score == _approx(expected)

    def test_at_lo_boundary_scores_one(self):
        """FRE exactly at lo should score 1.0.

        Use target_range where lo = FRE.
        target_range=(86.705, 95). FRE=86.705 -> 1.0.
        """
        scorer = ReadabilityScorer(target_range=(_KNOWN_FRE, 95))
        score = scorer(_KNOWN_TEXT)
        assert score == _approx(1.0)

    def test_at_hi_boundary_scores_one(self):
        """FRE exactly at hi should score 1.0.

        target_range=(60, 86.705). FRE=86.705 -> 1.0.
        """
        scorer = ReadabilityScorer(target_range=(60, _KNOWN_FRE))
        score = scorer(_KNOWN_TEXT)
        assert score == _approx(1.0)

    def test_fre_above_100_clamps_to_zero(self):
        """FRE >= 100 with hi < 100 should score 0.0.

        "The cat sat" → FRE = 119.19.
        target_range=(60, 80): score = (100 - 119.19) / (100 - 80) < 0 → clamp to 0.
        """
        scorer = ReadabilityScorer(target_range=(60, 80))
        score = scorer(_SIMPLE_TEXT)
        assert score == _approx(0.0)

    def test_none_returns_zero_in_range_mode(self):
        """None input should return 0.0 even in target-range mode."""
        scorer = ReadabilityScorer(target_range=(60, 80))
        assert scorer(None) == 0.0


# ---------------------------------------------------------------------------
# 5. as_filter — hard filter for GenerationLoop
# ---------------------------------------------------------------------------


class TestAsFilter:
    """Test as_filter() for hard readability filtering."""

    def test_returns_callable(self):
        """as_filter should return a callable."""
        scorer = ReadabilityScorer()
        filt = scorer.as_filter(min_fre=60, max_fre=90)
        assert callable(filt)

    def test_accepts_candidate_in_range(self):
        """Candidate with FRE in range should pass the filter.

        FRE=86.705, range [60, 90] -> True.
        """
        scorer = ReadabilityScorer()
        filt = scorer.as_filter(min_fre=60, max_fre=90)
        candidate = {"text": _KNOWN_TEXT, "phonemes": ["x"]}
        assert filt(candidate) is True

    def test_rejects_candidate_above_range(self):
        """Candidate with FRE above max should fail the filter.

        "The cat sat" → FRE=119.19, range [60, 90] -> False.
        """
        scorer = ReadabilityScorer()
        filt = scorer.as_filter(min_fre=60, max_fre=90)
        candidate = {"text": _SIMPLE_TEXT, "phonemes": ["x"]}
        assert filt(candidate) is False

    def test_rejects_candidate_below_range(self):
        """Candidate with FRE below min should fail the filter.

        FRE=86.705, range [90, 100] -> False.
        """
        scorer = ReadabilityScorer()
        filt = scorer.as_filter(min_fre=90, max_fre=100)
        candidate = {"text": _KNOWN_TEXT, "phonemes": ["x"]}
        assert filt(candidate) is False

    def test_rejects_candidate_without_text(self):
        """Candidate with no text should fail the filter."""
        scorer = ReadabilityScorer()
        filt = scorer.as_filter(min_fre=60, max_fre=90)
        candidate = {"phonemes": ["x"]}
        assert filt(candidate) is False

    def test_rejects_none_text(self):
        """Candidate with text=None should fail the filter."""
        scorer = ReadabilityScorer()
        filt = scorer.as_filter(min_fre=60, max_fre=90)
        candidate = {"text": None, "phonemes": ["x"]}
        assert filt(candidate) is False

    def test_boundary_inclusive(self):
        """FRE exactly at min or max should pass.

        FRE=86.705, filter range [86.705, 90] -> True.
        """
        scorer = ReadabilityScorer()
        filt = scorer.as_filter(min_fre=_KNOWN_FRE, max_fre=90)
        candidate = {"text": _KNOWN_TEXT, "phonemes": ["x"]}
        # Due to float precision, use approximate comparison
        # The filter should accept FRE very close to the boundary
        assert filt(candidate) is True
