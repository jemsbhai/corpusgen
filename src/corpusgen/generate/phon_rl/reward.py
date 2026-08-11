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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from corpusgen.generate.phon_ctg.targets import PhoneticTargetInventory
from corpusgen.weights import validate_component_weights

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
        language: str = "en-us",
    ) -> None:
        validate_component_weights(
            {
                "coverage_weight": coverage_weight,
                "phonotactic_weight": phonotactic_weight,
                "fluency_weight": fluency_weight,
            }
        )

        self._targets = targets
        self._phonotactic_scorer = phonotactic_scorer
        self._fluency_scorer = fluency_scorer
        self._ref_log_probs_fn = ref_log_probs_fn
        self._coverage_weight = coverage_weight
        self._phonotactic_weight = phonotactic_weight
        self._fluency_weight = fluency_weight
        self._language = language

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
    def language(self) -> str:
        """Language code for G2P phonemization."""
        return self._language

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

    def _compute_new_units(
        self,
        phonemes: list[str],
        covered_units: set[str] | None = None,
    ) -> set[str]:
        """Find which target units this phoneme sequence would newly cover."""
        units = self._extract_units(phonemes)
        target_set = self._targets.target_units
        already_covered = (
            self._targets.covered_units
            if covered_units is None
            else covered_units
        )
        return {u for u in units if u in target_set and u not in already_covered}

    def _score_sentence_against(
        self,
        phonemes: list[str],
        text: str | None,
        covered_units: set[str],
    ) -> RewardBreakdown:
        """Score a sentence against an explicit coverage snapshot."""
        new_units = self._compute_new_units(
            phonemes,
            covered_units=covered_units,
        )
        coverage_gain = len(new_units)

        target_size = self._targets.target_size
        coverage_reward = (
            coverage_gain / target_size if target_size > 0 else 0.0
        )
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
        return self._score_sentence_against(
            phonemes=phonemes,
            text=text,
            covered_units=self._targets.covered_units,
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

        Decodes tokens incrementally, detects word boundaries from trailing
        whitespace, leading whitespace/word markers, or the final token,
        phonemizes completed words, and assigns the coverage reward for each
        word to the boundary token that completed it. Non-boundary tokens
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

        # Keep decoded text for word assembly/G2P, but retain raw token
        # strings for boundary markers that decode(single_id) may strip.
        token_strings = [
            tokenizer.decode(tid, skip_special_tokens=True)
            for tid in token_ids
        ]
        convert_id = getattr(tokenizer, "convert_ids_to_tokens", None)
        raw_token_strings: list[str] = []
        for token_id, decoded_token in zip(token_ids, token_strings):
            raw_token = (
                convert_id(token_id)
                if callable(convert_id)
                else decoded_token
            )
            raw_token_strings.append(
                raw_token if isinstance(raw_token, str) else decoded_token
            )

        # Accumulate tokens into words, detecting boundaries
        per_token_rewards: list[float] = [0.0] * len(token_ids)
        word_boundaries: list[int] = []
        words_phonemized: list[str] = []
        current_word_tokens: list[str] = []
        rewarded_units: set[str] = set()
        g2p: Any | None = None

        target_size = self._targets.target_size

        def score_current_word(boundary_index: int) -> None:
            nonlocal g2p

            word = "".join(current_word_tokens).strip()
            current_word_tokens.clear()
            if not word:
                return

            word_boundaries.append(boundary_index)
            words_phonemized.append(word)

            from corpusgen.g2p.manager import G2PManager

            try:
                if g2p is None:
                    g2p = G2PManager()
                g2p_result = g2p.phonemize(
                    word, language=self._language
                )
                word_phonemes = g2p_result.phonemes
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to phonemize token-level word {word!r} "
                    f"for language {self._language!r}."
                ) from exc

            new_units = self._compute_new_units(word_phonemes) - rewarded_units
            rewarded_units.update(new_units)
            if target_size > 0 and new_units:
                per_token_rewards[boundary_index] = len(new_units) / target_size

        leading_boundary_markers = ("▁", "Ġ")

        for i, (tok_str, raw_tok_str) in enumerate(
            zip(token_strings, raw_token_strings)
        ):
            starts_with_space = bool(
                (tok_str and tok_str[0].isspace())
                or (raw_tok_str and raw_tok_str[0].isspace())
            )
            starts_with_marker = (
                tok_str.startswith(leading_boundary_markers)
                or raw_tok_str.startswith(leading_boundary_markers)
            )

            # A leading boundary means the preceding token completed the
            # word accumulated so far.
            if (starts_with_space or starts_with_marker) and current_word_tokens:
                score_current_word(i - 1)

            if tok_str.startswith(leading_boundary_markers):
                tok_str = tok_str[1:]
            if starts_with_space:
                tok_str = tok_str.lstrip()
            current_word_tokens.append(tok_str)

            is_last = i == len(token_strings) - 1
            ends_with_space = bool(
                (tok_str and tok_str[-1].isspace())
                or (raw_tok_str and raw_tok_str[-1].isspace())
            )
            if ends_with_space or is_last:
                score_current_word(i)

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
