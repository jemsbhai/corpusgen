"""Phonotactic Constraint Module: validates phoneme sequence legality.

Provides an ABC for phonotactic constraint implementations and a concrete
n-gram statistical model. The constraint's ``score`` method is designed to
plug directly into ``PhoneticScorer(phonotactic_scorer=model.score)``.

Implementations:
    - **NgramPhonotacticModel**: Trained on observed phoneme sequences,
      scores new sequences by average log-probability of their n-grams
      with add-k smoothing. Supports bigram, trigram, or higher orders.

Future implementations:
    - Rule-based (per-language phonotactic rules)
    - Neural (CNN-Transformer phonotactic model)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any


class PhonotacticConstraint(ABC):
    """Abstract base class for phonotactic constraint models.

    All implementations must provide:
        - ``fit(phoneme_sequences)``: train from observed data
        - ``score(phonemes)``: evaluate a phoneme sequence -> float
    """

    @abstractmethod
    def fit(self, phoneme_sequences: list[list[str]]) -> None:
        """Train the constraint model from phoneme sequences.

        Args:
            phoneme_sequences: List of phoneme lists, each representing
                one utterance or sentence.
        """

    @abstractmethod
    def score(self, phonemes: list[str]) -> float:
        """Score a phoneme sequence for phonotactic legality.

        Args:
            phonemes: List of phonemes to evaluate.

        Returns:
            Float in [0.0, 1.0] where 1.0 = fully legal.
        """


# ---------------------------------------------------------------------------
# Boundary symbols for n-gram context
# ---------------------------------------------------------------------------

_BOS = "<s>"   # beginning of sequence
_EOS = "</s>"  # end of sequence


class NgramPhonotacticModel(PhonotacticConstraint):
    """N-gram statistical model for phonotactic constraint scoring.

    Trains on observed phoneme sequences and scores new sequences by
    the average log-probability of their n-grams, mapped to [0, 1]
    via a sigmoid-like transform. Uses add-k smoothing to handle
    unseen n-grams.

    Args:
        order: N-gram order (2 = bigram, 3 = trigram, etc.). Default: 2.
        smoothing: Add-k smoothing constant. Default: 0.01.
    """

    def __init__(self, order: int = 2, smoothing: float = 0.01) -> None:
        if order < 1:
            raise ValueError(
                f"Order must be >= 1, got {order}."
            )
        self._order = order
        self._smoothing = smoothing

        # N-gram counts: context tuple -> {next_phoneme: count}
        self._ngram_counts: dict[tuple[str, ...], dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # Context totals for efficient probability computation
        self._context_totals: dict[tuple[str, ...], int] = defaultdict(int)
        # Vocabulary (set of all seen phonemes + boundary symbols)
        self._vocabulary: set[str] = set()
        self._is_fitted = False

    # -------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------

    @property
    def order(self) -> int:
        """N-gram order."""
        return self._order

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been trained."""
        return self._is_fitted

    @property
    def vocabulary(self) -> set[str]:
        """Set of phonemes seen during training (excluding boundary symbols)."""
        return self._vocabulary - {_BOS, _EOS}

    # -------------------------------------------------------------------
    # Fit: from pre-phonemized data
    # -------------------------------------------------------------------

    def fit(self, phoneme_sequences: list[list[str]]) -> None:
        """Train from pre-phonemized sequences.

        Args:
            phoneme_sequences: List of phoneme lists.

        Raises:
            ValueError: If input is empty or contains no valid sequences.
        """
        if not phoneme_sequences:
            raise ValueError("Empty phoneme_sequences list.")

        # Filter out empty sequences
        valid = [seq for seq in phoneme_sequences if seq]
        if not valid:
            raise ValueError("No valid (non-empty) phoneme sequences provided.")

        # Reset state
        self._ngram_counts = defaultdict(lambda: defaultdict(int))
        self._context_totals = defaultdict(int)
        self._vocabulary = {_BOS, _EOS}

        for seq in valid:
            self._vocabulary.update(seq)
            # Pad with boundary symbols
            padded = [_BOS] * (self._order - 1) + list(seq) + [_EOS]

            for i in range(len(padded) - self._order + 1):
                ngram = padded[i : i + self._order]
                context = tuple(ngram[:-1])
                target = ngram[-1]
                self._ngram_counts[context][target] += 1
                self._context_totals[context] += 1

        self._is_fitted = True

    # -------------------------------------------------------------------
    # Fit: from raw text via G2P
    # -------------------------------------------------------------------

    def fit_from_text(
        self,
        texts: list[str],
        language: str = "en-us",
    ) -> None:
        """Train from raw text by running G2P conversion first.

        Args:
            texts: List of text strings (sentences or words).
            language: Language code for G2P conversion.

        Raises:
            ValueError: If texts is empty.
        """
        if not texts:
            raise ValueError("Empty texts list.")

        from corpusgen.g2p.manager import G2PManager

        g2p = G2PManager()
        results = g2p.phonemize_batch(texts, language=language)
        sequences = [r.phonemes for r in results]
        self.fit(sequences)

    # -------------------------------------------------------------------
    # Score
    # -------------------------------------------------------------------

    def _log_prob(self, context: tuple[str, ...], target: str) -> float:
        """Compute smoothed log-probability of target given context."""
        vocab_size = len(self._vocabulary)
        count = self._ngram_counts[context][target]
        total = self._context_totals[context]

        # Add-k smoothing
        prob = (count + self._smoothing) / (total + self._smoothing * vocab_size)
        return math.log(prob)

    def score(self, phonemes: list[str]) -> float:
        """Score a phoneme sequence for phonotactic legality.

        Computes the average log-probability of the sequence's n-grams,
        then maps to [0, 1] using a sigmoid transform.

        Args:
            phonemes: List of phonemes to evaluate.

        Returns:
            Float in [0.0, 1.0] where higher = more phonotactically legal.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Model not fitted. Call fit() or fit_from_text() first."
            )

        if not phonemes:
            # Empty sequence — return neutral score
            return 0.5

        # Pad with boundary symbols
        padded = [_BOS] * (self._order - 1) + list(phonemes) + [_EOS]

        # Compute average log-probability
        log_probs = []
        for i in range(len(padded) - self._order + 1):
            ngram = padded[i : i + self._order]
            context = tuple(ngram[:-1])
            target = ngram[-1]
            log_probs.append(self._log_prob(context, target))

        if not log_probs:
            return 0.5

        avg_log_prob = sum(log_probs) / len(log_probs)

        # Map to [0, 1] via sigmoid: 1 / (1 + exp(-x))
        # Shift so that avg_log_prob near 0 maps to ~0.5
        return 1.0 / (1.0 + math.exp(-avg_log_prob))

    # -------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Export model state as a plain dict for serialization.

        Returns:
            Dict suitable for JSON serialization.
        """
        # Convert defaultdicts to plain dicts
        ngram_counts: dict[str, dict[str, int]] = {}
        for context, targets in self._ngram_counts.items():
            key = "|".join(context)
            ngram_counts[key] = dict(targets)

        context_totals: dict[str, int] = {}
        for context, total in self._context_totals.items():
            key = "|".join(context)
            context_totals[key] = total

        return {
            "order": self._order,
            "smoothing": self._smoothing,
            "is_fitted": self._is_fitted,
            "vocabulary": sorted(self._vocabulary),
            "ngram_counts": ngram_counts,
            "context_totals": context_totals,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NgramPhonotacticModel:
        """Restore a model from a dict produced by ``to_dict()``.

        Args:
            data: Dict with model state.

        Returns:
            A restored NgramPhonotacticModel instance.
        """
        model = cls(order=data["order"], smoothing=data["smoothing"])

        if not data.get("is_fitted", False):
            return model

        model._vocabulary = set(data["vocabulary"])

        model._ngram_counts = defaultdict(lambda: defaultdict(int))
        for key, targets in data["ngram_counts"].items():
            context = tuple(key.split("|"))
            for target, count in targets.items():
                model._ngram_counts[context][target] = count

        model._context_totals = defaultdict(int)
        for key, total in data["context_totals"].items():
            context = tuple(key.split("|"))
            model._context_totals[context] = total

        model._is_fitted = True
        return model
