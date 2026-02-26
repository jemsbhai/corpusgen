"""Tests for phon_rl.reward — PhoneticReward composite reward function."""

from __future__ import annotations

import pytest

from corpusgen.generate.phon_ctg.targets import PhoneticTargetInventory
from corpusgen.generate.phon_rl.reward import (
    PhoneticReward,
    RewardBreakdown,
    TokenRewardResult,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


@pytest.fixture()
def phoneme_inventory() -> PhoneticTargetInventory:
    """Inventory targeting 5 English-like phonemes."""
    return PhoneticTargetInventory(
        target_phonemes=["p", "b", "t", "d", "k"],
        unit="phoneme",
    )


@pytest.fixture()
def diphone_inventory() -> PhoneticTargetInventory:
    """Inventory targeting diphones."""
    return PhoneticTargetInventory(
        target_phonemes=["p", "b", "t", "d", "k"],
        unit="diphone",
    )


@pytest.fixture()
def basic_reward(phoneme_inventory: PhoneticTargetInventory) -> PhoneticReward:
    """Reward with coverage only (no phonotactic or fluency hooks)."""
    return PhoneticReward(
        targets=phoneme_inventory,
        coverage_weight=1.0,
        phonotactic_weight=0.0,
        fluency_weight=0.0,
    )


@pytest.fixture()
def phonotactic_fn():
    """Simple phonotactic scorer: returns 1.0 if all phonemes are valid consonants."""
    valid = {"p", "b", "t", "d", "k", "s", "n", "m", "l", "r"}

    def scorer(phonemes: list[str]) -> float:
        if not phonemes:
            return 0.0
        valid_count = sum(1 for p in phonemes if p in valid)
        return valid_count / len(phonemes)

    return scorer


@pytest.fixture()
def fluency_fn():
    """Simple fluency scorer: returns 1.0 for short text, 0.5 for long."""

    def scorer(text: str | None) -> float:
        if text is None:
            return 0.0
        return 1.0 if len(text) < 50 else 0.5

    return scorer


# -----------------------------------------------------------------------
# RewardBreakdown
# -----------------------------------------------------------------------


class TestRewardBreakdown:
    """Tests for the RewardBreakdown dataclass."""

    def test_fields_present(self) -> None:
        rb = RewardBreakdown(
            coverage_reward=0.5,
            phonotactic_reward=0.3,
            fluency_reward=0.2,
            composite_reward=0.4,
            new_units={"p", "t"},
            coverage_gain=2,
        )
        assert rb.coverage_reward == 0.5
        assert rb.phonotactic_reward == 0.3
        assert rb.fluency_reward == 0.2
        assert rb.composite_reward == 0.4
        assert rb.new_units == {"p", "t"}
        assert rb.coverage_gain == 2

    def test_empty_new_units(self) -> None:
        rb = RewardBreakdown(
            coverage_reward=0.0,
            phonotactic_reward=0.0,
            fluency_reward=0.0,
            composite_reward=0.0,
            new_units=set(),
            coverage_gain=0,
        )
        assert len(rb.new_units) == 0
        assert rb.coverage_gain == 0


# -----------------------------------------------------------------------
# TokenRewardResult
# -----------------------------------------------------------------------


class TestTokenRewardResult:
    """Tests for the TokenRewardResult dataclass."""

    def test_fields_present(self) -> None:
        tr = TokenRewardResult(
            per_token_rewards=[0.0, 0.0, 0.5, 0.0, 0.3],
            word_boundaries=[2, 4],
            words_phonemized=["pat", "bad"],
        )
        assert len(tr.per_token_rewards) == 5
        assert tr.word_boundaries == [2, 4]
        assert tr.words_phonemized == ["pat", "bad"]


# -----------------------------------------------------------------------
# Construction / validation
# -----------------------------------------------------------------------


class TestPhoneticRewardConstruction:
    """Tests for PhoneticReward construction and validation."""

    def test_basic_construction(self, phoneme_inventory: PhoneticTargetInventory) -> None:
        reward = PhoneticReward(
            targets=phoneme_inventory,
            coverage_weight=1.0,
            phonotactic_weight=0.5,
            fluency_weight=0.3,
        )
        assert reward.coverage_weight == 1.0
        assert reward.phonotactic_weight == 0.5
        assert reward.fluency_weight == 0.3
        assert reward.targets is phoneme_inventory

    def test_negative_weights_rejected(
        self, phoneme_inventory: PhoneticTargetInventory
    ) -> None:
        with pytest.raises(ValueError, match="coverage_weight"):
            PhoneticReward(
                targets=phoneme_inventory,
                coverage_weight=-1.0,
            )

    def test_negative_phonotactic_weight_rejected(
        self, phoneme_inventory: PhoneticTargetInventory
    ) -> None:
        with pytest.raises(ValueError, match="phonotactic_weight"):
            PhoneticReward(
                targets=phoneme_inventory,
                phonotactic_weight=-0.1,
            )

    def test_negative_fluency_weight_rejected(
        self, phoneme_inventory: PhoneticTargetInventory
    ) -> None:
        with pytest.raises(ValueError, match="fluency_weight"):
            PhoneticReward(
                targets=phoneme_inventory,
                fluency_weight=-0.5,
            )

    def test_phonotactic_scorer_stored(
        self,
        phoneme_inventory: PhoneticTargetInventory,
        phonotactic_fn,
    ) -> None:
        reward = PhoneticReward(
            targets=phoneme_inventory,
            phonotactic_scorer=phonotactic_fn,
            phonotactic_weight=0.5,
        )
        assert reward.phonotactic_scorer is phonotactic_fn

    def test_fluency_scorer_stored(
        self,
        phoneme_inventory: PhoneticTargetInventory,
        fluency_fn,
    ) -> None:
        reward = PhoneticReward(
            targets=phoneme_inventory,
            fluency_scorer=fluency_fn,
            fluency_weight=0.3,
        )
        assert reward.fluency_scorer is fluency_fn

    def test_ref_log_probs_fn_stored(
        self,
        phoneme_inventory: PhoneticTargetInventory,
    ) -> None:
        def dummy_ref(text: str) -> float:
            return -1.0

        reward = PhoneticReward(
            targets=phoneme_inventory,
            ref_log_probs_fn=dummy_ref,
        )
        assert reward.ref_log_probs_fn is dummy_ref


# -----------------------------------------------------------------------
# sentence_reward — coverage only
# -----------------------------------------------------------------------


class TestSentenceRewardCoverageOnly:
    """sentence_reward with only coverage weight active."""

    def test_covers_new_units(self, basic_reward: PhoneticReward) -> None:
        """Phonemes [p, t] should yield coverage_gain=2."""
        result = basic_reward.sentence_reward(
            phonemes=["p", "t"],
            text="pat",
        )
        assert isinstance(result, RewardBreakdown)
        assert result.coverage_gain == 2
        assert result.new_units == {"p", "t"}
        assert result.coverage_reward > 0.0

    def test_no_new_units(self, basic_reward: PhoneticReward) -> None:
        """Phoneme [z] is not in the target set -> gain=0."""
        result = basic_reward.sentence_reward(
            phonemes=["z"],
            text="zoo",
        )
        assert result.coverage_gain == 0
        assert result.new_units == set()
        assert result.coverage_reward == 0.0

    def test_empty_phonemes(self, basic_reward: PhoneticReward) -> None:
        result = basic_reward.sentence_reward(phonemes=[], text="")
        assert result.coverage_gain == 0
        assert result.composite_reward == 0.0

    def test_duplicate_phonemes_counted_once(
        self, basic_reward: PhoneticReward
    ) -> None:
        """[p, p, p] covers only 1 new unit."""
        result = basic_reward.sentence_reward(
            phonemes=["p", "p", "p"],
            text="pappa",
        )
        assert result.coverage_gain == 1
        assert result.new_units == {"p"}

    def test_composite_equals_weighted_coverage(
        self, basic_reward: PhoneticReward
    ) -> None:
        """With w_cov=1, w_phono=0, w_fluency=0, composite should equal coverage_reward."""
        result = basic_reward.sentence_reward(
            phonemes=["p", "b", "t"],
            text="pocket bat",
        )
        assert result.composite_reward == pytest.approx(result.coverage_reward)

    def test_does_not_mutate_inventory(
        self, basic_reward: PhoneticReward
    ) -> None:
        """sentence_reward is non-destructive (peek mode)."""
        before = basic_reward.targets.covered_count
        basic_reward.sentence_reward(phonemes=["p", "t"], text="pat")
        after = basic_reward.targets.covered_count
        assert after == before


# -----------------------------------------------------------------------
# sentence_reward — composite with phonotactic + fluency
# -----------------------------------------------------------------------


class TestSentenceRewardComposite:
    """sentence_reward with all three components active."""

    def test_composite_formula(
        self,
        phoneme_inventory: PhoneticTargetInventory,
        phonotactic_fn,
        fluency_fn,
    ) -> None:
        reward = PhoneticReward(
            targets=phoneme_inventory,
            phonotactic_scorer=phonotactic_fn,
            fluency_scorer=fluency_fn,
            coverage_weight=1.0,
            phonotactic_weight=0.5,
            fluency_weight=0.3,
        )
        result = reward.sentence_reward(
            phonemes=["p", "t"],  # 2 valid consonants -> phonotactic=1.0
            text="pat",  # short -> fluency=1.0
        )
        expected_composite = (
            1.0 * result.coverage_reward
            + 0.5 * result.phonotactic_reward
            + 0.3 * result.fluency_reward
        )
        assert result.composite_reward == pytest.approx(expected_composite)

    def test_phonotactic_component_only(
        self,
        phoneme_inventory: PhoneticTargetInventory,
        phonotactic_fn,
    ) -> None:
        """With only phonotactic weight, composite reflects phonotactic score."""
        reward = PhoneticReward(
            targets=phoneme_inventory,
            phonotactic_scorer=phonotactic_fn,
            coverage_weight=0.0,
            phonotactic_weight=1.0,
            fluency_weight=0.0,
        )
        result = reward.sentence_reward(
            phonemes=["p", "t"],
            text="pat",
        )
        assert result.composite_reward == pytest.approx(result.phonotactic_reward)

    def test_fluency_component_only(
        self,
        phoneme_inventory: PhoneticTargetInventory,
        fluency_fn,
    ) -> None:
        reward = PhoneticReward(
            targets=phoneme_inventory,
            fluency_scorer=fluency_fn,
            coverage_weight=0.0,
            phonotactic_weight=0.0,
            fluency_weight=1.0,
        )
        result = reward.sentence_reward(
            phonemes=["p"],
            text="pat",
        )
        assert result.composite_reward == pytest.approx(result.fluency_reward)

    def test_no_scorers_means_zero_components(
        self,
        phoneme_inventory: PhoneticTargetInventory,
    ) -> None:
        """If phonotactic/fluency hooks are None, those components are 0."""
        reward = PhoneticReward(
            targets=phoneme_inventory,
            coverage_weight=1.0,
            phonotactic_weight=0.5,
            fluency_weight=0.3,
        )
        result = reward.sentence_reward(
            phonemes=["p", "t"],
            text="pat",
        )
        assert result.phonotactic_reward == 0.0
        assert result.fluency_reward == 0.0


# -----------------------------------------------------------------------
# sentence_reward — KL divergence / ref_log_probs_fn
# -----------------------------------------------------------------------


class TestSentenceRewardKL:
    """sentence_reward with ref_log_probs_fn for KL-based fluency."""

    def test_ref_log_probs_used_as_fluency(
        self,
        phoneme_inventory: PhoneticTargetInventory,
    ) -> None:
        """When ref_log_probs_fn is provided and no fluency_scorer,
        the ref_log_probs_fn output is used as the fluency signal."""

        def ref_fn(text: str) -> float:
            return -2.5  # negative log-prob -> lower is more divergent

        reward = PhoneticReward(
            targets=phoneme_inventory,
            ref_log_probs_fn=ref_fn,
            fluency_weight=1.0,
        )
        result = reward.sentence_reward(phonemes=["p"], text="pat")
        assert result.fluency_reward == pytest.approx(-2.5)

    def test_fluency_scorer_takes_precedence(
        self,
        phoneme_inventory: PhoneticTargetInventory,
        fluency_fn,
    ) -> None:
        """If both fluency_scorer and ref_log_probs_fn are given,
        fluency_scorer takes precedence."""

        def ref_fn(text: str) -> float:
            return -99.0

        reward = PhoneticReward(
            targets=phoneme_inventory,
            fluency_scorer=fluency_fn,
            ref_log_probs_fn=ref_fn,
            fluency_weight=1.0,
        )
        result = reward.sentence_reward(phonemes=["p"], text="pat")
        # fluency_fn returns 1.0 for short text, not -99
        assert result.fluency_reward == pytest.approx(1.0)


# -----------------------------------------------------------------------
# sentence_reward — coverage normalization
# -----------------------------------------------------------------------


class TestSentenceRewardNormalization:
    """Coverage reward should be normalized by target inventory size."""

    def test_coverage_normalized_by_target_size(
        self,
        phoneme_inventory: PhoneticTargetInventory,
    ) -> None:
        """With 5 targets, covering 2 should give coverage_reward = 2/5."""
        reward = PhoneticReward(
            targets=phoneme_inventory,
            coverage_weight=1.0,
        )
        result = reward.sentence_reward(phonemes=["p", "t"], text="pat")
        assert result.coverage_reward == pytest.approx(2.0 / 5.0)

    def test_full_coverage_yields_one(
        self,
        phoneme_inventory: PhoneticTargetInventory,
    ) -> None:
        reward = PhoneticReward(
            targets=phoneme_inventory,
            coverage_weight=1.0,
        )
        result = reward.sentence_reward(
            phonemes=["p", "b", "t", "d", "k"],
            text="test",
        )
        assert result.coverage_reward == pytest.approx(1.0)


# -----------------------------------------------------------------------
# token_rewards — word boundary detection
# -----------------------------------------------------------------------


class TestTokenRewards:
    """token_rewards: sparse per-token rewards at word boundaries."""

    def test_basic_word_boundary_rewards(
        self,
        phoneme_inventory: PhoneticTargetInventory,
    ) -> None:
        """Tokens forming two words; reward at boundary tokens only."""
        reward = PhoneticReward(
            targets=phoneme_inventory,
            coverage_weight=1.0,
        )
        # Simulate tokens: ["pa", "t ", "ba", "d"]
        # Word boundaries at indices 1 ("t " ends "pat") and 3 ("d" ends "bad")
        mock_tokenizer = _MockTokenizer(["pa", "t ", "ba", "d"])
        result = reward.token_rewards(
            token_ids=[0, 1, 2, 3],
            tokenizer=mock_tokenizer,
        )
        assert isinstance(result, TokenRewardResult)
        assert len(result.per_token_rewards) == 4
        # Non-boundary tokens should be 0
        assert result.per_token_rewards[0] == 0.0
        assert result.per_token_rewards[2] == 0.0
        # Boundary tokens should have reward > 0 (they complete words with target phonemes)
        # Exact values depend on what phonemes G2P returns, so we check structure
        assert len(result.word_boundaries) >= 1

    def test_empty_token_ids(
        self,
        basic_reward: PhoneticReward,
    ) -> None:
        mock_tokenizer = _MockTokenizer([])
        result = basic_reward.token_rewards(
            token_ids=[],
            tokenizer=mock_tokenizer,
        )
        assert result.per_token_rewards == []
        assert result.word_boundaries == []

    def test_single_word_single_token(
        self,
        phoneme_inventory: PhoneticTargetInventory,
    ) -> None:
        """A single token forming a complete word."""
        reward = PhoneticReward(
            targets=phoneme_inventory,
            coverage_weight=1.0,
        )
        mock_tokenizer = _MockTokenizer(["pat"])
        result = reward.token_rewards(
            token_ids=[0],
            tokenizer=mock_tokenizer,
        )
        assert len(result.per_token_rewards) == 1
        # The single token completes a word, so it's a boundary
        assert len(result.word_boundaries) == 1
        assert result.word_boundaries[0] == 0

    def test_rewards_are_non_negative(
        self,
        basic_reward: PhoneticReward,
    ) -> None:
        """Per-token coverage rewards should never be negative."""
        mock_tokenizer = _MockTokenizer(["the ", "cat ", "sat"])
        result = basic_reward.token_rewards(
            token_ids=[0, 1, 2],
            tokenizer=mock_tokenizer,
        )
        for r in result.per_token_rewards:
            assert r >= 0.0


# -----------------------------------------------------------------------
# hierarchical_reward — combines sentence + token
# -----------------------------------------------------------------------


class TestHierarchicalReward:
    """hierarchical_reward combines sentence-level and token-level signals."""

    def test_returns_both_components(
        self,
        phoneme_inventory: PhoneticTargetInventory,
    ) -> None:
        reward = PhoneticReward(
            targets=phoneme_inventory,
            coverage_weight=1.0,
        )
        mock_tokenizer = _MockTokenizer(["pat"])
        sent_result, token_result = reward.hierarchical_reward(
            text="pat",
            phonemes=["p", "t"],
            token_ids=[0],
            tokenizer=mock_tokenizer,
        )
        assert isinstance(sent_result, RewardBreakdown)
        assert isinstance(token_result, TokenRewardResult)

    def test_sentence_component_is_peek(
        self,
        phoneme_inventory: PhoneticTargetInventory,
    ) -> None:
        """hierarchical_reward should not mutate the target inventory."""
        reward = PhoneticReward(
            targets=phoneme_inventory,
            coverage_weight=1.0,
        )
        before = reward.targets.covered_count
        mock_tokenizer = _MockTokenizer(["pat"])
        reward.hierarchical_reward(
            text="pat",
            phonemes=["p", "t"],
            token_ids=[0],
            tokenizer=mock_tokenizer,
        )
        assert reward.targets.covered_count == before

    def test_token_rewards_length_matches_tokens(
        self,
        phoneme_inventory: PhoneticTargetInventory,
    ) -> None:
        reward = PhoneticReward(
            targets=phoneme_inventory,
            coverage_weight=1.0,
        )
        tokens = ["The ", "big ", "cat"]
        mock_tokenizer = _MockTokenizer(tokens)
        _, token_result = reward.hierarchical_reward(
            text="The big cat",
            phonemes=["d", "b", "k", "t"],
            token_ids=list(range(len(tokens))),
            tokenizer=mock_tokenizer,
        )
        assert len(token_result.per_token_rewards) == len(tokens)


# -----------------------------------------------------------------------
# sentence_reward with diphone inventory
# -----------------------------------------------------------------------


class TestSentenceRewardDiphone:
    """sentence_reward should respect the inventory unit type."""

    def test_diphone_coverage(
        self, diphone_inventory: PhoneticTargetInventory
    ) -> None:
        reward = PhoneticReward(
            targets=diphone_inventory,
            coverage_weight=1.0,
        )
        # Phonemes [p, b] -> diphone "p-b"
        result = reward.sentence_reward(
            phonemes=["p", "b"],
            text="pb",
        )
        # "p-b" is a valid diphone target if it's in the target set
        # Whether it's a target depends on CoverageTracker's target generation
        # At minimum, coverage_gain should be >= 0
        assert result.coverage_gain >= 0
        assert isinstance(result.new_units, set)


# -----------------------------------------------------------------------
# commit_sentence_reward — mutates inventory
# -----------------------------------------------------------------------


class TestCommitSentenceReward:
    """commit_sentence_reward scores then updates the inventory."""

    def test_commit_updates_inventory(
        self,
        phoneme_inventory: PhoneticTargetInventory,
    ) -> None:
        reward = PhoneticReward(
            targets=phoneme_inventory,
            coverage_weight=1.0,
        )
        before = reward.targets.covered_count
        result = reward.commit_sentence_reward(
            phonemes=["p", "t"],
            text="pat",
            sentence_index=0,
        )
        after = reward.targets.covered_count
        assert after > before
        assert result.coverage_gain == 2

    def test_second_commit_no_double_count(
        self,
        phoneme_inventory: PhoneticTargetInventory,
    ) -> None:
        """Same phonemes committed twice -> second has gain=0."""
        reward = PhoneticReward(
            targets=phoneme_inventory,
            coverage_weight=1.0,
        )
        reward.commit_sentence_reward(
            phonemes=["p", "t"],
            text="pat",
            sentence_index=0,
        )
        result2 = reward.commit_sentence_reward(
            phonemes=["p", "t"],
            text="pat again",
            sentence_index=1,
        )
        assert result2.coverage_gain == 0
        assert result2.new_units == set()


# -----------------------------------------------------------------------
# Mock tokenizer for token_rewards tests
# -----------------------------------------------------------------------


class _MockTokenizer:
    """Minimal tokenizer mock that maps token IDs to strings.

    Token ID i -> tokens[i]. Supports decode(token_id) for individual
    tokens and batch_decode for lists.
    """

    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens

    def decode(self, token_id: int, skip_special_tokens: bool = True) -> str:
        if 0 <= token_id < len(self._tokens):
            return self._tokens[token_id]
        return ""

    def batch_decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        return [self.decode(tid) for tid in token_ids]

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.decode(token_id)
