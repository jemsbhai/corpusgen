"""N-gram phonotactic scorer for Phon-CTG candidate evaluation.

Scores phoneme sequences by their phonotactic naturalness using an
n-gram transition model with Laplace smoothing. Higher scores indicate
more natural phoneme transitions for the target language.

Two modes:
    - **Inventory-derived** (default): Builds a uniform distribution over
      all in-inventory n-gram transitions. Out-of-inventory transitions
      receive lower probability via Laplace smoothing. This is a weak
      baseline that distinguishes "possible" from "impossible" transitions
      but does not capture language-specific phonotactic preferences.
    - **Corpus-trained**: Builds an n-gram model from observed phoneme
      sequences. Captures actual transition frequencies from real data.
      Scientifically stronger — use when a reference corpus is available.

Smoothing:
    Laplace (add-1) smoothing ensures no transition has zero probability.
    For inventory-derived mode, in-inventory transitions get probability
    ``(1 + 1) / (V + V^2)`` while out-of-inventory transitions get
    ``1 / (V + V^2)`` for bigrams (where V = inventory size).

References:
    - Vitevitch, M. S., & Luce, P. A. (2004). A Web-based interface to
      calculate phonotactic probability for words and nonwords in English.
      Behavior Research Methods, 36(3), 481–487.
    - Jurafsky, D., & Martin, J. H. (2024). Speech and Language Processing
      (3rd ed.). Chapter 3: N-gram Language Models.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path


class NgramPhonotacticScorer:
    """N-gram phonotactic scorer with Laplace smoothing.

    Scores phoneme sequences based on transition probabilities.
    Callable interface: ``scorer(phonemes) -> float`` in [0, 1].

    Args:
        phonemes: Target phoneme inventory.
        n: N-gram order (2 = bigram, 3 = trigram). Must be >= 2.

    Raises:
        ValueError: If phonemes is empty, has < 2 elements, or n < 2.
    """

    def __init__(
        self,
        phonemes: list[str],
        n: int = 2,
    ) -> None:
        if n < 2:
            raise ValueError(f"n must be >= 2, got {n}")
        if not phonemes:
            raise ValueError("phonemes must not be empty.")
        if len(phonemes) < 2:
            raise ValueError(
                f"phonemes must contain at least 2 elements for n-gram "
                f"construction, got {len(phonemes)}."
            )

        self._phonemes = list(phonemes)
        self._n = n
        self._vocabulary = set(phonemes)
        self._vocab_size = len(self._vocabulary)

        # Build uniform n-gram counts: every in-inventory n-gram gets count 1
        # This is equivalent to observing each valid transition once.
        self._counts: dict[tuple[str, ...], dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._context_totals: dict[tuple[str, ...], int] = defaultdict(int)

        self._build_uniform_model()

    def _build_uniform_model(self) -> None:
        """Build uniform counts: every in-inventory n-gram gets count 1."""
        vocab = sorted(self._vocabulary)

        if self._n == 2:
            # Bigrams: context is single phoneme, next is single phoneme
            for p1 in vocab:
                ctx = (p1,)
                for p2 in vocab:
                    self._counts[ctx][p2] = 1
                    self._context_totals[ctx] += 1
        else:
            # Higher-order: context is (n-1) phonemes
            # Generate all possible (n-1)-grams as contexts
            self._build_uniform_recursive(vocab, (), self._n - 1)

    def _build_uniform_recursive(
        self,
        vocab: list[str],
        prefix: tuple[str, ...],
        remaining: int,
    ) -> None:
        """Recursively build uniform counts for n-grams of any order."""
        if remaining == 0:
            # prefix is now a full (n-1)-gram context
            ctx = prefix
            for p in vocab:
                self._counts[ctx][p] = 1
                self._context_totals[ctx] += 1
        else:
            for p in vocab:
                self._build_uniform_recursive(vocab, prefix + (p,), remaining - 1)

    # -------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------

    @property
    def n(self) -> int:
        """N-gram order."""
        return self._n

    @property
    def phonemes(self) -> list[str]:
        """Target phoneme inventory."""
        return list(self._phonemes)

    # -------------------------------------------------------------------
    # Alternative constructor: corpus-trained
    # -------------------------------------------------------------------

    @classmethod
    def from_corpus(
        cls,
        sequences: list[list[str]],
        n: int = 2,
    ) -> NgramPhonotacticScorer:
        """Build an n-gram model from observed phoneme sequences.

        Trains on actual transition frequencies from a corpus. This
        produces a scientifically stronger model than the inventory-derived
        default.

        Args:
            sequences: List of phoneme sequences (each a list of str).
            n: N-gram order. Must be >= 2.

        Returns:
            A trained NgramPhonotacticScorer.

        Raises:
            ValueError: If sequences is empty, n < 2, or no sequences
                are long enough to form n-grams.
        """
        if n < 2:
            raise ValueError(f"n must be >= 2, got {n}")
        if not sequences:
            raise ValueError("corpus must not be empty.")

        # Collect vocabulary and counts from corpus
        vocabulary: set[str] = set()
        counts: dict[tuple[str, ...], dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        context_totals: dict[tuple[str, ...], int] = defaultdict(int)
        valid_sequences = 0

        for seq in sequences:
            if len(seq) < n:
                continue  # skip sequences too short for this n-gram order
            valid_sequences += 1
            vocabulary.update(seq)

            for i in range(len(seq) - n + 1):
                ngram = tuple(seq[i : i + n])
                context = ngram[:-1]
                target = ngram[-1]
                counts[context][target] += 1
                context_totals[context] += 1

        if valid_sequences == 0:
            raise ValueError(
                f"No sequences long enough to form n-gram transitions "
                f"(n={n}). Need sequences of length >= {n}."
            )

        # Create instance with corpus-derived data
        phonemes = sorted(vocabulary)
        scorer = cls.__new__(cls)
        scorer._phonemes = phonemes
        scorer._n = n
        scorer._vocabulary = vocabulary
        scorer._vocab_size = len(vocabulary)
        scorer._counts = counts
        scorer._context_totals = context_totals

        return scorer

    # -------------------------------------------------------------------
    # Persistence (save/load)
    # -------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the n-gram model to a JSON file for reproducibility.

        Args:
            path: File path to write. Created or overwritten.
        """
        # Convert defaultdict → plain dict with tuple keys as strings
        counts_serializable: dict[str, dict[str, int]] = {}
        for ctx, targets in self._counts.items():
            key = "|".join(ctx)
            counts_serializable[key] = dict(targets)

        context_totals_serializable: dict[str, int] = {
            "|".join(ctx): total
            for ctx, total in self._context_totals.items()
        }

        data = {
            "phonemes": self._phonemes,
            "n": self._n,
            "counts": counts_serializable,
            "context_totals": context_totals_serializable,
        }
        Path(path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> NgramPhonotacticScorer:
        """Load an n-gram model from a JSON file.

        Args:
            path: Path to a file previously created by :meth:`save`.

        Returns:
            A restored NgramPhonotacticScorer.

        Raises:
            FileNotFoundError: If path does not exist.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Phonotactic model file not found: {path}")

        data = json.loads(p.read_text(encoding="utf-8"))

        scorer = cls.__new__(cls)
        scorer._phonemes = data["phonemes"]
        scorer._n = data["n"]
        scorer._vocabulary = set(scorer._phonemes)
        scorer._vocab_size = len(scorer._vocabulary)

        # Restore counts from serialized format
        scorer._counts = defaultdict(lambda: defaultdict(int))
        for key_str, targets in data["counts"].items():
            ctx = tuple(key_str.split("|"))
            for target, count in targets.items():
                scorer._counts[ctx][target] = count

        scorer._context_totals = defaultdict(int)
        for key_str, total in data["context_totals"].items():
            ctx = tuple(key_str.split("|"))
            scorer._context_totals[ctx] = total

        return scorer

    # -------------------------------------------------------------------
    # Scoring
    # -------------------------------------------------------------------

    def __call__(self, phonemes: list[str]) -> float:
        """Score a phoneme sequence for phonotactic naturalness.

        Computes the mean log-probability of n-gram transitions under the
        model with Laplace smoothing, then normalizes to [0, 1].

        Args:
            phonemes: Phoneme sequence to score.

        Returns:
            Float in [0, 1]. Higher = more phonotactically natural.
            Returns 0.0 for sequences shorter than n.
        """
        if len(phonemes) < self._n:
            return 0.0

        total_log_prob = 0.0
        num_transitions = 0

        for i in range(len(phonemes) - self._n + 1):
            ngram = tuple(phonemes[i : i + self._n])
            context = ngram[:-1]
            target = ngram[-1]

            # Laplace-smoothed probability
            count = self._counts.get(context, {}).get(target, 0)
            context_total = self._context_totals.get(context, 0)

            # P(target | context) = (count + 1) / (context_total + V_smooth)
            # where V_smooth = vocabulary size + 1 (the +1 accounts for
            # unseen/OOV phonemes, ensuring that in-inventory transitions
            # score strictly higher than out-of-inventory transitions)
            v_smooth = self._vocab_size + 1
            prob = (count + 1) / (context_total + v_smooth)
            total_log_prob += math.log(prob)
            num_transitions += 1

        if num_transitions == 0:
            return 0.0

        # Mean log-probability per transition
        mean_log_prob = total_log_prob / num_transitions

        # Normalize to [0, 1] using sigmoid-like transform:
        # score = 1 / (1 + exp(-mean_log_prob - offset))
        # The offset centers the sigmoid so that "average" inventory
        # transitions map to ~0.5.
        #
        # For a uniform model with V phonemes, the expected log-prob
        # of an in-inventory transition is log(2 / (V + V^2)) for bigrams.
        # We use log(1/V) as a reasonable centering point.
        offset = -math.log(max(self._vocab_size + 1, 2))
        score = 1.0 / (1.0 + math.exp(-(mean_log_prob - offset)))

        return score
