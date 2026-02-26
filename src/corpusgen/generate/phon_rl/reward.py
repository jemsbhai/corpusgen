"""PhoneticReward: composite reward function for Phon-RL.

Provides sentence-level and token-level reward signals for reinforcement
learning with phonetic objectives. The composite reward combines:

    R = w_cov · R_coverage + w_phono · R_phonotactic + w_fluency · R_fluency

Operates in three modes:
    - **sentence_reward** (peek): scores a complete sentence without
      modifying the target inventory.
    - **commit_sentence_reward**: scores then updates the inventory.
    - **token_rewards**: sparse per-token rewards at word boundaries,
      providing denser learning signal for PPO.
    - **hierarchical_reward**: combines sentence-level terminal bonus
      with token-level dense signal.

The coverage component is normalized by the target inventory size to
produce values in [0, 1]. Phonotactic and fluency components are
passed through from external scorers.

For KL-based fluency regularization, accepts a ``ref_log_probs_fn``
callable that returns the reference model's log-probability for a
given text. This is used as the fluency signal when no explicit
``fluency_scorer`` is provided.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from corpusgen.generate.phon_ctg.targets import PhoneticTargetInventory

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Result dataclasses
# -----------------------------------------------------------------------


@dataclass
class RewardBreakdown:
    """Structured result from sentence-level reward computation.

    Attributes:
        coverage_reward: Normalized coverage gain (new_units / target_size).
        phonotactic_reward: Score from phonotactic constraint hook.
        fluency_reward: Score from fluency hook or ref_log_probs_fn.
        composite_reward: Weighted combination of all components.
        new_units: Set of target units newly covered by this sentence.
        coverage_gain: Number of new target units covered.
    """

    coverage_reward: float
    phonotactic_reward: float
    fluency_reward: float
    composite_reward: float
    new_units: set[str]
    coverage_gain: int


@dataclass
class TokenRewardResult:
    """Structured result from token-level reward computation.

    Attributes:
        per_token_rewards: Reward value for each token. Non-boundary
            tokens receive 0.0; boundary tokens receive the coverage
            reward for the completed word.
        word_boundaries: Indices of tokens that complete a word.
        words_phonemized: The words that were phonemized at each
            boundary (for debugging / logging).
    """

    per_token_rewards: list[float]
    word_boundaries: list[int]
    words_phonemized: list[str] = field(default_factory=list)


# -----------------------------------------------------------------------
# PhoneticReward
# -----------------------------------------------------------------------


class PhoneticReward:
    """Composite reward function for Phon-RL training.

    Evaluates generated text against a phonetic target inventory,
    combining coverage gain with optional phonotactic and fluency
    signals into a single scalar reward.

    Args:
        targets: PhoneticTargetInventory tracking coverage state.
        phonotactic_scorer: Optional callable (phonemes -> float)
            for phonotactic legality scoring.
        fluency_scorer: Optional callable (text -> float) for
            fluency scoring. Takes precedence over ref_log_probs_fn.
        ref_log_probs_fn: Optional callable (text -> float) returning
            the reference model's log-probability. Used as the fluency
            signal when fluency_scorer is None.
        coverage_weight: Weight for the coverage component (must be >= 0).
        phonotactic_weight: Weight for the phonotactic component (must be >= 0).
        fluency_weight: Weight for the fluency component (must be >= 0).
    """

    def __init__(
        self,
        targets: PhoneticTargetInventory,
        phonotactic_scorer: Callable[[list[str]], float] | None = None,
        fluency_scorer: Callable[[str | None], float] | None = None,
        ref_log_probs_fn: Callable[[str], float] | None = None,
        coverage_weight: float = 1.0,
        phonotactic_weight: float = 0.0,
        fluency_weight: float = 0.0,
    ) -> None:
        if coverage_weight < 0:
            raise ValueError(
                f"coverage_weight must be >= 0, got {coverage_weight}"
            )
        if phonotactic_weight < 0:
            raise ValueError(
                f"phonotactic_weight must be >= 0, got {phonotactic_weight}"
            )
        if fluency_weight < 0:
            raise ValueError(
                f"fluency_weight must be >= 0, got {fluency_weight}"
            )

        self._targets = targets
        self._phonotactic_scorer = phonotactic_scorer
        self._fluency_scorer = fluency_scorer
        self._ref_log_probs_fn = ref_log_probs_fn
        self._coverage_weight = coverage_weight
        self._phonotactic_weight = phonotactic_weight
        self._fluency_weight = fluency_weight

    # -------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------

    @property
    def targets(self) -> PhoneticTargetInventory:
        """The target inventory being scored against."""
        return self._targets

    @property
    def coverage_weight(self) -> float:
        """Weight for coverage component."""
        return self._coverage_weight

    @property
    def phonotactic_weight(self) -> float:
        """Weight for phonotactic component."""
        return self._phonotactic_weight

    @property
    def fluency_weight(self) -> float:
        """Weight for fluency component."""
        return self._fluency_weight

    @property
    def phonotactic_scorer(self) -> Callable[[list[str]], float] | None:
        """Optional phonotactic scoring callable."""
        return self._phonotactic_scorer

    @property
    def fluency_scorer(self) -> Callable[[str | None], float] | None:
        """Optional fluency scoring callable."""
        return self._fluency_scorer

    @property
    def ref_log_probs_fn(self) -> Callable[[str], float] | None:
        """Optional reference model log-prob callable for KL fluency."""
        return self._ref_log_probs_fn

    # -------------------------------------------------------------------
    # Internal: unit extraction (mirrors PhoneticScorer logic)
    # -------------------------------------------------------------------

    def _extract_units(self, phonemes: list[str]) -> list[str]:
        """Extract coverage units from a phoneme sequence."""
        unit = self._targets.unit
        if unit == "phoneme":
            return phonemes
        elif unit == "diphone":
            return [
                f"{phonemes[i]}-{phonemes[i + 1]}"
                for i in range(len(phonemes) - 1)
            ]
        elif unit == "triphone":
            return [
                f"{phonemes[i]}-{phonemes[i + 1]}-{phonemes[i + 2]}"
                for i in range(len(phonemes) - 2)
            ]
        return []

    def _compute_new_units(self, phonemes: list[str]) -> set[str]:
        """Find which target units this phoneme sequence would newly cover."""
        units = self._extract_units(phonemes)
        target_set = self._targets.target_units
        already_covered = self._targets.covered_units
        return {u for u in units if u in target_set and u not in already_covered}

    # -------------------------------------------------------------------
    # Internal: compute fluency score
    # -------------------------------------------------------------------

    def _compute_fluency(self, text: str | None) -> float:
        """Compute fluency score using the best available signal.

        Priority: fluency_scorer > ref_log_probs_fn > 0.0
        """
        if self._fluency_scorer is not None:
            return self._fluency_scorer(text)
        if self._ref_log_probs_fn is not None and text is not None:
            return self._ref_log_probs_fn(text)
        return 0.0

    # -------------------------------------------------------------------
    # Internal: compute phonotactic score
    # -------------------------------------------------------------------

    def _compute_phonotactic(self, phonemes: list[str]) -> float:
        """Compute phonotactic score if a scorer is available."""
        if self._phonotactic_scorer is not None:
            return self._phonotactic_scorer(phonemes)
        return 0.0

    # -------------------------------------------------------------------
    # Sentence-level reward (peek mode)
    # -------------------------------------------------------------------

    def sentence_reward(
        self,
        phonemes: list[str],
        text: str | None = None,
    ) -> RewardBreakdown:
        """Compute sentence-level composite reward without modifying inventory.

        Coverage is normalized by target inventory size to yield values
        in [0, 1].

        Args:
            phonemes: Phoneme list for the generated sentence.
            text: Raw text of the generated sentence (for fluency scoring).

        Returns:
            RewardBreakdown with all score components.
        """
        new_units = self._compute_new_units(phonemes)
        coverage_gain = len(new_units)

        # Normalize coverage by target size
        target_size = self._targets.target_size
        if target_size > 0:
            coverage_reward = coverage_gain / target_size
        else:
            coverage_reward = 0.0

        phonotactic_reward = self._compute_phonotactic(phonemes)
        fluency_reward = self._compute_fluency(text)

        composite = (
            self._coverage_weight * coverage_reward
            + self._phonotactic_weight * phonotactic_reward
            + self._fluency_weight * fluency_reward
        )

        return RewardBreakdown(
            coverage_reward=coverage_reward,
            phonotactic_reward=phonotactic_reward,
            fluency_reward=fluency_reward,
            composite_reward=composite,
            new_units=new_units,
            coverage_gain=coverage_gain,
        )

    # -------------------------------------------------------------------
    # Sentence-level reward (commit mode)
    # -------------------------------------------------------------------

    def commit_sentence_reward(
        self,
        phonemes: list[str],
        text: str | None = None,
        sentence_index: int = 0,
    ) -> RewardBreakdown:
        """Compute sentence-level reward then update the target inventory.

        Scores first (peek), then commits the coverage update.

        Args:
            phonemes: Phoneme list for the generated sentence.
            text: Raw text of the generated sentence.
            sentence_index: Index for provenance tracking in the inventory.

        Returns:
            RewardBreakdown computed before the inventory update.
        """
        result = self.sentence_reward(phonemes=phonemes, text=text)
        self._targets.update(phonemes, sentence_index)
        return result

    # -------------------------------------------------------------------
    # Token-level rewards (sparse, at word boundaries)
    # -------------------------------------------------------------------

    def token_rewards(
        self,
        token_ids: list[int],
        tokenizer: Any,
    ) -> TokenRewardResult:
        """Compute sparse per-token rewards at word boundaries.

        Decodes tokens incrementally, detects word boundaries (tokens
        ending in whitespace or being the final token), phonemizes
        completed words, and assigns the coverage reward for each word
        to the boundary token that completed it. Non-boundary tokens
        receive 0.0.

        This provides denser learning signal than pure sentence-level
        reward without fabricating sub-word phonetic information.

        Args:
            token_ids: List of generated token IDs.
            tokenizer: HuggingFace-compatible tokenizer with
                ``decode()`` method.

        Returns:
            TokenRewardResult with per-token rewards and boundary info.
        """
        if not token_ids:
            return TokenRewardResult(
                per_token_rewards=[],
                word_boundaries=[],
                words_phonemized=[],
            )

        # Decode each token to its string representation
        token_strings = [
            tokenizer.decode(tid, skip_special_tokens=True)
            for tid in token_ids
        ]

        # Accumulate tokens into words, detecting boundaries
        per_token_rewards: list[float] = [0.0] * len(token_ids)
        word_boundaries: list[int] = []
        words_phonemized: list[str] = []
        current_word_tokens: list[str] = []

        target_size = self._targets.target_size

        for i, tok_str in enumerate(token_strings):
            current_word_tokens.append(tok_str)

            # A word boundary occurs when:
            # 1. The token ends with whitespace, OR
            # 2. This is the last token in the sequence
            is_last = i == len(token_strings) - 1
            ends_with_space = bool(tok_str and tok_str[-1] in (" ", "\t", "\n"))

            if ends_with_space or is_last:
                # Assemble the completed word
                word = "".join(current_word_tokens).strip()
                if word:
                    word_boundaries.append(i)
                    words_phonemized.append(word)

                    # Compute coverage reward for this word
                    word_phonemes = self._simple_char_phonemes(word)
                    new_units = self._compute_new_units(word_phonemes)
                    if target_size > 0 and new_units:
                        per_token_rewards[i] = len(new_units) / target_size
                    else:
                        per_token_rewards[i] = 0.0

                current_word_tokens = []

        return TokenRewardResult(
            per_token_rewards=per_token_rewards,
            word_boundaries=word_boundaries,
            words_phonemized=words_phonemized,
        )

    # -------------------------------------------------------------------
    # Hierarchical reward (sentence + token combined)
    # -------------------------------------------------------------------

    def hierarchical_reward(
        self,
        text: str,
        phonemes: list[str],
        token_ids: list[int],
        tokenizer: Any,
    ) -> tuple[RewardBreakdown, TokenRewardResult]:
        """Compute both sentence-level and token-level rewards.

        Neither component mutates the target inventory (both are peek).

        Args:
            text: Full generated text.
            phonemes: Full phoneme sequence for the text.
            token_ids: Token IDs of the generated sequence.
            tokenizer: HuggingFace-compatible tokenizer.

        Returns:
            Tuple of (RewardBreakdown, TokenRewardResult).
        """
        sentence_result = self.sentence_reward(phonemes=phonemes, text=text)
        token_result = self.token_rewards(
            token_ids=token_ids,
            tokenizer=tokenizer,
        )
        return sentence_result, token_result

    # -------------------------------------------------------------------
    # Internal: simple character-based phoneme approximation
    # -------------------------------------------------------------------

    @staticmethod
    def _simple_char_phonemes(word: str) -> list[str]:
        """Extract approximate phonemes from a word for token-level scoring.

        Uses a simple character-level decomposition as a fast approximation.
        For token-level rewards during RL training, exact G2P accuracy is
        less critical than providing a directionally correct signal at
        word boundaries. Full G2P is applied at the sentence level.

        Args:
            word: A single word string.

        Returns:
            List of lowercase alphabetic characters as pseudo-phonemes.
        """
        # Strip non-alphabetic characters and lowercase
        cleaned = re.sub(r"[^a-zA-Z]", "", word.lower())
        return list(cleaned)
