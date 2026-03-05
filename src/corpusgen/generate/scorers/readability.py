"""Readability scorer for Phon-CTG candidate evaluation.

Scores text by its Flesch Reading Ease (FRE), mapping to [0, 1] for use
as a scorer hook in :class:`PhoneticScorer`.

Two modes:

* **Simple** (default): ``score = clamp(FRE / 100, 0, 1)``.
  Higher FRE (easier text) yields higher score.

* **Target-range**: trapezoidal mapping around a desired readability
  band ``[lo, hi]``.  Score is 1.0 inside the band and decays
  linearly to 0.0 at FRE=0 (below) and FRE=100 (above).  This
  penalises both overly complex and overly trivial text.

Also provides :meth:`as_filter` to create a hard accept/reject filter
for the :class:`GenerationLoop` ``candidate_filter`` hook.

Reuses the syllable-counting and Latin-script detection helpers from
:mod:`corpusgen.evaluate.text_quality`.

Callable interface: ``scorer(text) -> float`` in [0, 1].

Usage::

    from corpusgen.generate.scorers.readability import ReadabilityScorer

    # Simple mode — higher FRE = higher score
    scorer = ReadabilityScorer()
    score = scorer("The cat sat on the mat.")  # ~1.0

    # Target-range mode — peak score inside [60, 80]
    scorer = ReadabilityScorer(target_range=(60, 80))
    score = scorer("The cat sat on the mat.")  # may be < 1.0 if too easy

    # Hard filter for GenerationLoop
    filt = scorer.as_filter(min_fre=60, max_fre=90)
    ok = filt({"text": "Some sentence.", "phonemes": [...]})
"""

from __future__ import annotations

from collections.abc import Callable

from corpusgen.evaluate.text_quality import (
    _count_syllables_word,
    _is_latin_script,
    tokenize_words,
)


class ReadabilityScorer:
    """Readability scorer based on Flesch Reading Ease.

    Callable interface: ``scorer(text) -> float`` in [0, 1].

    Args:
        target_range: Optional ``(lo, hi)`` FRE band for trapezoidal
            scoring.  If *None*, uses simple clamped mode.

    Raises:
        ValueError: If *target_range* has lo > hi or negative values.
    """

    def __init__(
        self,
        target_range: tuple[float, float] | None = None,
    ) -> None:
        if target_range is not None:
            lo, hi = target_range
            if lo < 0 or hi < 0:
                raise ValueError(
                    f"Range values must be non-negative, got ({lo}, {hi})."
                )
            if lo > hi:
                raise ValueError(
                    f"Range lo must be <= hi, got ({lo}, {hi})."
                )
        self._target_range = target_range

    # -------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------

    @property
    def target_range(self) -> tuple[float, float] | None:
        """The target FRE range, or None for simple mode."""
        return self._target_range

    # -------------------------------------------------------------------
    # Raw FRE computation
    # -------------------------------------------------------------------

    def compute_fre(self, text: str | None) -> float | None:
        """Compute raw Flesch Reading Ease for a single sentence.

        Args:
            text: Input text.  Returns *None* for None, empty,
                whitespace-only, or non-Latin-script text.

        Returns:
            Flesch Reading Ease score, or *None* if not computable.
        """
        if text is None or not text.strip():
            return None

        if not _is_latin_script(text):
            return None

        words = tokenize_words(text)
        if not words:
            return None

        total_words = len(words)
        total_syllables = sum(_count_syllables_word(w) for w in words)

        # Single sentence: num_sentences = 1
        fre = (
            206.835
            - 1.015 * total_words
            - 84.6 * (total_syllables / total_words)
        )
        return fre

    # -------------------------------------------------------------------
    # Scoring
    # -------------------------------------------------------------------

    def __call__(self, text: str | None) -> float:
        """Score text for readability.

        Args:
            text: Text to score.  Returns 0.0 for None, empty,
                whitespace, or non-Latin text.

        Returns:
            Float in [0, 1].
        """
        fre = self.compute_fre(text)
        if fre is None:
            return 0.0

        if self._target_range is None:
            # Simple mode: clamp(FRE / 100, 0, 1)
            return max(0.0, min(fre / 100.0, 1.0))

        # Target-range mode: trapezoidal
        lo, hi = self._target_range
        if lo <= fre <= hi:
            return 1.0
        elif fre < lo:
            # Linear decay from 1.0 at lo to 0.0 at FRE=0
            if lo == 0:
                return 1.0
            return max(0.0, fre / lo)
        else:
            # fre > hi: linear decay from 1.0 at hi to 0.0 at FRE=100
            if hi >= 100:
                return 1.0
            return max(0.0, (100.0 - fre) / (100.0 - hi))

    # -------------------------------------------------------------------
    # Hard filter for GenerationLoop
    # -------------------------------------------------------------------

    def as_filter(
        self,
        min_fre: float,
        max_fre: float,
    ) -> Callable[[dict], bool]:
        """Create a hard accept/reject filter for candidate dicts.

        Returns a callable suitable for the ``candidate_filter``
        parameter of :class:`GenerationLoop`.

        Args:
            min_fre: Minimum Flesch Reading Ease to accept (inclusive).
            max_fre: Maximum Flesch Reading Ease to accept (inclusive).

        Returns:
            A callable ``(candidate_dict) -> bool``.
        """

        def _filter(candidate: dict) -> bool:
            text = candidate.get("text")
            fre = self.compute_fre(text)
            if fre is None:
                return False
            return min_fre <= fre <= max_fre

        return _filter
