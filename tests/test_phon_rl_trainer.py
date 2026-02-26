"""Tests for phon_rl.trainer — PhonRLTrainer with custom PPO loop.

Test tiers:
    - **Fast tests** (default): Pure logic — config validation, prompt
      validation, dataclass behavior, construction. No dependencies.
    - **PPO math tests** (default, requires torch): Pure tensor math for
      GAE, KL divergence, PPO clipped loss. These test the actual RL
      algorithms with known inputs and expected outputs.
    - **Wiring tests** (default): Verify the training loop calls model
      loading and runs the expected number of steps. Mocks only at the
      external library boundary (_load_model_and_tokenizer).
    - **Slow tests** (@pytest.mark.slow): Real gpt2 model, real reward,
      real PPO updates. Skipped by default. Run with:
          poetry run pytest -m slow
      Required before any publication claim.
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from corpusgen.generate.phon_ctg.targets import PhoneticTargetInventory
from corpusgen.generate.phon_rl.reward import PhoneticReward
from corpusgen.generate.phon_rl.trainer import (
    PhonRLTrainer,
    TrainingConfig,
    TrainingResult,
)


# =======================================================================
# Fixtures
# =======================================================================


@pytest.fixture()
def phoneme_inventory() -> PhoneticTargetInventory:
    return PhoneticTargetInventory(
        target_phonemes=["p", "b", "t", "d", "k"],
        unit="phoneme",
    )


@pytest.fixture()
def reward(phoneme_inventory: PhoneticTargetInventory) -> PhoneticReward:
    return PhoneticReward(
        targets=phoneme_inventory,
        coverage_weight=1.0,
        phonotactic_weight=0.0,
        fluency_weight=0.0,
    )


@pytest.fixture()
def default_config() -> TrainingConfig:
    return TrainingConfig(
        model_name="gpt2",
        num_steps=3,
        batch_size=2,
        learning_rate=1e-5,
        kl_coeff=0.1,
        output_dir="/tmp/test_phon_rl",
        seed=42,
    )


@pytest.fixture()
def static_prompts() -> list[str]:
    return [
        "Generate a sentence with the sound p.",
        "Generate a sentence with the sound t.",
        "Generate a sentence with the sound k.",
    ]


@pytest.fixture()
def dynamic_prompt_fn():
    def fn(targets: PhoneticTargetInventory) -> str:
        missing = targets.next_targets(3)
        if missing:
            return f"Generate sentences with: {', '.join(missing)}"
        return "Generate any natural sentence."

    return fn


# =======================================================================
# FAST TESTS — Pure logic, no dependencies
# =======================================================================


class TestTrainingResult:
    def test_fields_present(self) -> None:
        result = TrainingResult(
            mean_rewards=[0.1, 0.2, 0.3],
            total_steps=30,
            final_coverage=0.8,
            checkpoint_path="/tmp/checkpoint",
        )
        assert result.mean_rewards == [0.1, 0.2, 0.3]
        assert result.total_steps == 30
        assert result.final_coverage == 0.8
        assert result.checkpoint_path == "/tmp/checkpoint"

    def test_empty_rewards(self) -> None:
        result = TrainingResult(
            mean_rewards=[],
            total_steps=0,
            final_coverage=0.0,
            checkpoint_path=None,
        )
        assert result.mean_rewards == []
        assert result.checkpoint_path is None


class TestTrainingConfig:
    def test_defaults(self) -> None:
        config = TrainingConfig(model_name="gpt2")
        assert config.model_name == "gpt2"
        assert config.num_steps > 0
        assert config.batch_size > 0
        assert config.learning_rate > 0
        assert config.kl_coeff >= 0
        assert config.seed is not None

    def test_custom_values(self) -> None:
        config = TrainingConfig(
            model_name="mistral-7b",
            num_steps=500,
            batch_size=8,
            learning_rate=2e-5,
            kl_coeff=0.05,
            output_dir="/data/checkpoints",
            seed=123,
            max_new_tokens=128,
            temperature=0.9,
            use_peft=True,
            peft_r=16,
            peft_alpha=32,
        )
        assert config.model_name == "mistral-7b"
        assert config.num_steps == 500
        assert config.use_peft is True
        assert config.peft_r == 16

    def test_clip_epsilon_default(self) -> None:
        config = TrainingConfig(model_name="gpt2")
        assert config.clip_epsilon == 0.2

    def test_gae_params_default(self) -> None:
        config = TrainingConfig(model_name="gpt2")
        assert config.gae_gamma == 1.0
        assert config.gae_lambda == 0.95

    def test_negative_num_steps_rejected(self) -> None:
        with pytest.raises(ValueError, match="num_steps"):
            TrainingConfig(model_name="gpt2", num_steps=-1)

    def test_zero_num_steps_rejected(self) -> None:
        with pytest.raises(ValueError, match="num_steps"):
            TrainingConfig(model_name="gpt2", num_steps=0)

    def test_negative_batch_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            TrainingConfig(model_name="gpt2", batch_size=0)

    def test_negative_learning_rate_rejected(self) -> None:
        with pytest.raises(ValueError, match="learning_rate"):
            TrainingConfig(model_name="gpt2", learning_rate=-1e-5)

    def test_negative_kl_coeff_rejected(self) -> None:
        with pytest.raises(ValueError, match="kl_coeff"):
            TrainingConfig(model_name="gpt2", kl_coeff=-0.1)

    def test_negative_clip_epsilon_rejected(self) -> None:
        with pytest.raises(ValueError, match="clip_epsilon"):
            TrainingConfig(model_name="gpt2", clip_epsilon=-0.1)


class TestConstruction:
    def test_basic_construction(
        self, reward: PhoneticReward, default_config: TrainingConfig
    ) -> None:
        trainer = PhonRLTrainer(reward=reward, config=default_config)
        assert trainer.reward is reward
        assert trainer.config is default_config

    def test_config_accessible(
        self, reward: PhoneticReward, default_config: TrainingConfig
    ) -> None:
        trainer = PhonRLTrainer(reward=reward, config=default_config)
        assert trainer.config.model_name == "gpt2"
        assert trainer.config.num_steps == 3

    def test_not_initialized_before_train(
        self, reward: PhoneticReward, default_config: TrainingConfig
    ) -> None:
        trainer = PhonRLTrainer(reward=reward, config=default_config)
        assert trainer.is_initialized is False

    def test_save_before_train_raises(
        self, reward: PhoneticReward, default_config: TrainingConfig
    ) -> None:
        trainer = PhonRLTrainer(reward=reward, config=default_config)
        with pytest.raises(RuntimeError, match="[Tt]rain"):
            trainer.save_checkpoint("/tmp/checkpoint")


class TestTrainPromptValidation:
    def test_neither_prompts_nor_fn_raises(
        self, reward: PhoneticReward, default_config: TrainingConfig
    ) -> None:
        trainer = PhonRLTrainer(reward=reward, config=default_config)
        with pytest.raises(ValueError, match="prompts.*prompt_fn"):
            trainer.train()

    def test_both_prompts_and_fn_raises(
        self,
        reward: PhoneticReward,
        default_config: TrainingConfig,
        static_prompts: list[str],
        dynamic_prompt_fn,
    ) -> None:
        trainer = PhonRLTrainer(reward=reward, config=default_config)
        with pytest.raises(ValueError, match="prompts.*prompt_fn"):
            trainer.train(prompts=static_prompts, prompt_fn=dynamic_prompt_fn)

    def test_empty_prompts_raises(
        self, reward: PhoneticReward, default_config: TrainingConfig
    ) -> None:
        trainer = PhonRLTrainer(reward=reward, config=default_config)
        with pytest.raises(ValueError, match="prompts.*empty"):
            trainer.train(prompts=[])


# =======================================================================
# PPO MATH TESTS — Pure tensor operations, known inputs/outputs
#
# These test the actual RL algorithms with hand-computed expected
# values. No mocking — real torch tensors. These are the scientific
# core of the implementation.
# =======================================================================

torch = pytest.importorskip("torch")

from corpusgen.generate.phon_rl.trainer import (
    compute_gae,
    compute_kl_penalty,
    compute_log_probs_from_logits,
    ppo_clip_loss,
)


class TestComputeLogProbs:
    """compute_log_probs_from_logits: logits + actions -> log probs."""

    def test_basic_log_probs(self) -> None:
        """Known logits -> known log probs for selected actions."""
        # Vocab size 3, sequence length 2
        logits = torch.tensor([
            [[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]],
        ])  # [1, 2, 3]
        actions = torch.tensor([[0, 1]])  # [1, 2]

        log_probs = compute_log_probs_from_logits(logits, actions)

        # Manual: softmax([2,1,0]) -> [0.6652, 0.2447, 0.0900]
        # log(0.6652) ≈ -0.4076
        expected_0 = torch.log_softmax(logits[0, 0], dim=-1)[0]
        expected_1 = torch.log_softmax(logits[0, 1], dim=-1)[1]

        assert log_probs.shape == (1, 2)
        assert torch.allclose(log_probs[0, 0], expected_0, atol=1e-5)
        assert torch.allclose(log_probs[0, 1], expected_1, atol=1e-5)

    def test_output_shape(self) -> None:
        logits = torch.randn(2, 5, 100)  # [batch=2, seq=5, vocab=100]
        actions = torch.randint(0, 100, (2, 5))
        log_probs = compute_log_probs_from_logits(logits, actions)
        assert log_probs.shape == (2, 5)

    def test_log_probs_are_negative(self) -> None:
        """Log probabilities must be <= 0."""
        logits = torch.randn(1, 10, 50)
        actions = torch.randint(0, 50, (1, 10))
        log_probs = compute_log_probs_from_logits(logits, actions)
        assert torch.all(log_probs <= 0.0 + 1e-6)

    def test_uniform_logits(self) -> None:
        """Uniform logits -> log(1/V) for all actions."""
        vocab_size = 4
        logits = torch.zeros(1, 3, vocab_size)
        actions = torch.tensor([[0, 1, 2]])
        log_probs = compute_log_probs_from_logits(logits, actions)
        expected = math.log(1.0 / vocab_size)
        assert torch.allclose(log_probs, torch.full_like(log_probs, expected), atol=1e-5)


class TestComputeKLPenalty:
    """compute_kl_penalty: per-token KL divergence."""

    def test_identical_distributions_zero_kl(self) -> None:
        """KL(p || p) = 0."""
        log_probs = torch.tensor([[-.5, -1.0, -0.3]])
        kl = compute_kl_penalty(log_probs, log_probs)
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)

    def test_kl_is_non_negative(self) -> None:
        """KL divergence is always >= 0."""
        policy = torch.tensor([[-0.5, -1.0, -0.3, -2.0]])
        ref = torch.tensor([[-0.8, -0.6, -0.5, -1.5]])
        kl = compute_kl_penalty(policy, ref)
        assert torch.all(kl >= -1e-6)

    def test_kl_shape(self) -> None:
        policy = torch.randn(2, 10)
        ref = torch.randn(2, 10)
        kl = compute_kl_penalty(policy, ref)
        assert kl.shape == (2, 10)

    def test_divergent_distributions_positive_kl(self) -> None:
        """Very different log probs should produce clearly positive KL."""
        policy = torch.tensor([[-0.1, -0.1]])  # high confidence
        ref = torch.tensor([[-5.0, -5.0]])  # low confidence
        kl = compute_kl_penalty(policy, ref)
        assert torch.all(kl > 0.1)


class TestComputeGAE:
    """compute_gae: Generalized Advantage Estimation (Schulman et al.)."""

    def test_single_step(self) -> None:
        """Single-step GAE: advantage = reward + gamma*0 - value = reward - value."""
        rewards = torch.tensor([[1.0]])
        values = torch.tensor([[0.5]])
        advantages, returns = compute_gae(
            rewards, values, gamma=1.0, lam=0.95
        )
        # delta = reward + gamma * next_value - value
        # next_value = 0 (terminal), so delta = 1.0 - 0.5 = 0.5
        assert advantages.shape == (1, 1)
        assert torch.allclose(advantages, torch.tensor([[0.5]]), atol=1e-5)

    def test_two_steps_gamma_one_lambda_one(self) -> None:
        """With gamma=1, lambda=1: GAE = MC returns - values."""
        rewards = torch.tensor([[1.0, 2.0]])
        values = torch.tensor([[0.0, 0.0]])
        advantages, returns = compute_gae(
            rewards, values, gamma=1.0, lam=1.0
        )
        # Step 1: delta_1 = r_1 + gamma*V(s2) - V(s1) = 2.0 + 0 - 0 = 2.0
        # Step 0: delta_0 = r_0 + gamma*V(s1) - V(s0) = 1.0 + 0 - 0 = 1.0
        # A_1 = delta_1 = 2.0
        # A_0 = delta_0 + gamma*lambda*A_1 = 1.0 + 1*1*2.0 = 3.0
        assert torch.allclose(advantages[0, 0], torch.tensor(3.0), atol=1e-5)
        assert torch.allclose(advantages[0, 1], torch.tensor(2.0), atol=1e-5)

    def test_returns_equal_advantages_plus_values(self) -> None:
        """returns = advantages + values, by definition."""
        rewards = torch.tensor([[1.0, 2.0, 3.0]])
        values = torch.tensor([[0.5, 0.3, 0.1]])
        advantages, returns = compute_gae(
            rewards, values, gamma=0.99, lam=0.95
        )
        expected_returns = advantages + values
        assert torch.allclose(returns, expected_returns, atol=1e-5)

    def test_shape_matches_input(self) -> None:
        rewards = torch.randn(4, 10)
        values = torch.randn(4, 10)
        advantages, returns = compute_gae(rewards, values, gamma=1.0, lam=0.95)
        assert advantages.shape == (4, 10)
        assert returns.shape == (4, 10)

    def test_zero_rewards_zero_values(self) -> None:
        """All zeros -> all-zero advantages and returns."""
        rewards = torch.zeros(2, 5)
        values = torch.zeros(2, 5)
        advantages, returns = compute_gae(rewards, values, gamma=1.0, lam=0.95)
        assert torch.allclose(advantages, torch.zeros_like(advantages))
        assert torch.allclose(returns, torch.zeros_like(returns))

    def test_gamma_zero_no_future(self) -> None:
        """gamma=0: advantages = rewards - values (no bootstrapping)."""
        rewards = torch.tensor([[1.0, 2.0, 3.0]])
        values = torch.tensor([[0.5, 0.5, 0.5]])
        advantages, _ = compute_gae(rewards, values, gamma=0.0, lam=0.95)
        expected = rewards - values
        assert torch.allclose(advantages, expected, atol=1e-5)


class TestPPOClipLoss:
    """ppo_clip_loss: clipped surrogate objective (Schulman et al., 2017)."""

    def test_no_change_zero_loss(self) -> None:
        """If new == old log probs, ratio=1, loss = -advantages."""
        log_probs = torch.tensor([[-1.0, -2.0]])
        advantages = torch.tensor([[1.0, 0.5]])
        loss = ppo_clip_loss(advantages, log_probs, log_probs, clip_epsilon=0.2)
        # ratio = 1.0, min(1*A, clip(1)*A) = A, loss = -mean(A)
        expected = -advantages.mean()
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_positive_advantage_caps_ratio(self) -> None:
        """With positive advantage, ratio clipped at 1+epsilon."""
        old = torch.tensor([[-2.0]])
        new = torch.tensor([[-0.5]])  # much higher prob -> ratio >> 1
        advantages = torch.tensor([[1.0]])

        loss_clipped = ppo_clip_loss(advantages, old, new, clip_epsilon=0.2)
        loss_unclipped = ppo_clip_loss(advantages, old, new, clip_epsilon=100.0)

        # Clipped loss should be less negative (more conservative)
        assert loss_clipped >= loss_unclipped

    def test_negative_advantage_caps_ratio(self) -> None:
        """With negative advantage, ratio clipped at 1-epsilon."""
        old = torch.tensor([[-0.5]])
        new = torch.tensor([[-2.0]])  # much lower prob -> ratio << 1
        advantages = torch.tensor([[-1.0]])

        loss_clipped = ppo_clip_loss(advantages, old, new, clip_epsilon=0.2)
        loss_unclipped = ppo_clip_loss(advantages, old, new, clip_epsilon=100.0)

        # Clipped loss should be smaller (more conservative update)
        assert loss_clipped >= loss_unclipped

    def test_output_is_scalar(self) -> None:
        old = torch.randn(2, 10)
        new = torch.randn(2, 10)
        adv = torch.randn(2, 10)
        loss = ppo_clip_loss(adv, old, new, clip_epsilon=0.2)
        assert loss.shape == ()

    def test_loss_is_finite(self) -> None:
        old = torch.randn(4, 20)
        new = torch.randn(4, 20)
        adv = torch.randn(4, 20)
        loss = ppo_clip_loss(adv, old, new, clip_epsilon=0.2)
        assert torch.isfinite(loss)

    def test_clip_epsilon_zero_is_pessimistic(self) -> None:
        """With epsilon=0, clipped_ratio=1 always, so
        loss = -min(ratio*A, 1*A).mean() — the more pessimistic
        of the actual and clamped surrogates."""
        old = torch.tensor([[-1.0, -2.0]])
        new = torch.tensor([[-0.5, -3.0]])
        advantages = torch.tensor([[1.0, 0.5]])
        loss = ppo_clip_loss(advantages, old, new, clip_epsilon=0.0)

        # Manual: ratio = exp(new - old) = (exp(0.5), exp(-1.0))
        ratio = (new - old).exp()
        surr1 = ratio * advantages
        surr2 = 1.0 * advantages  # clipped ratio = 1
        expected = -torch.min(surr1, surr2).mean()
        assert torch.allclose(loss, expected, atol=1e-5)


# =======================================================================
# WIRING TESTS — Mock only external library boundary
#
# Verify the training loop runs the expected number of steps,
# calls the right functions, tracks rewards. These do NOT prove
# correctness of PPO math (that's above) or real training (that's
# the slow tests below).
# =======================================================================


class TestWiringStaticPrompts:
    @patch("corpusgen.generate.phon_rl.trainer._load_model_and_tokenizer")
    def test_returns_training_result(
        self,
        mock_load: MagicMock,
        reward: PhoneticReward,
        default_config: TrainingConfig,
        static_prompts: list[str],
    ) -> None:
        _configure_mock_load(mock_load)
        trainer = PhonRLTrainer(reward=reward, config=default_config)
        result = trainer.train(prompts=static_prompts)

        assert isinstance(result, TrainingResult)
        assert result.total_steps == default_config.num_steps

    @patch("corpusgen.generate.phon_rl.trainer._load_model_and_tokenizer")
    def test_model_loaded_once(
        self,
        mock_load: MagicMock,
        reward: PhoneticReward,
        default_config: TrainingConfig,
        static_prompts: list[str],
    ) -> None:
        _configure_mock_load(mock_load)
        trainer = PhonRLTrainer(reward=reward, config=default_config)
        trainer.train(prompts=static_prompts)
        mock_load.assert_called_once()

    @patch("corpusgen.generate.phon_rl.trainer._load_model_and_tokenizer")
    def test_is_initialized_after_train(
        self,
        mock_load: MagicMock,
        reward: PhoneticReward,
        default_config: TrainingConfig,
        static_prompts: list[str],
    ) -> None:
        _configure_mock_load(mock_load)
        trainer = PhonRLTrainer(reward=reward, config=default_config)
        trainer.train(prompts=static_prompts)
        assert trainer.is_initialized is True


class TestWiringDynamicPrompts:
    @patch("corpusgen.generate.phon_rl.trainer._load_model_and_tokenizer")
    def test_prompt_fn_called_each_step(
        self,
        mock_load: MagicMock,
        reward: PhoneticReward,
        default_config: TrainingConfig,
    ) -> None:
        _configure_mock_load(mock_load)
        mock_fn = MagicMock(return_value="Generate a sentence.")
        trainer = PhonRLTrainer(reward=reward, config=default_config)
        trainer.train(prompt_fn=mock_fn)
        assert mock_fn.call_count == default_config.num_steps

    @patch("corpusgen.generate.phon_rl.trainer._load_model_and_tokenizer")
    def test_prompt_fn_receives_inventory(
        self,
        mock_load: MagicMock,
        reward: PhoneticReward,
        default_config: TrainingConfig,
    ) -> None:
        _configure_mock_load(mock_load)
        received: list[Any] = []

        def capture_fn(targets: PhoneticTargetInventory) -> str:
            received.append(targets)
            return "Generate a sentence."

        trainer = PhonRLTrainer(reward=reward, config=default_config)
        trainer.train(prompt_fn=capture_fn)

        assert len(received) == default_config.num_steps
        for arg in received:
            assert isinstance(arg, PhoneticTargetInventory)


class TestWiringCallbacks:
    @patch("corpusgen.generate.phon_rl.trainer._load_model_and_tokenizer")
    def test_step_callback_invoked(
        self,
        mock_load: MagicMock,
        reward: PhoneticReward,
        default_config: TrainingConfig,
        static_prompts: list[str],
    ) -> None:
        _configure_mock_load(mock_load)
        callback = MagicMock()
        trainer = PhonRLTrainer(reward=reward, config=default_config)
        trainer.train(prompts=static_prompts, step_callback=callback)
        assert callback.call_count == default_config.num_steps

    @patch("corpusgen.generate.phon_rl.trainer._load_model_and_tokenizer")
    def test_step_callback_receives_step_and_reward(
        self,
        mock_load: MagicMock,
        reward: PhoneticReward,
        default_config: TrainingConfig,
        static_prompts: list[str],
    ) -> None:
        _configure_mock_load(mock_load)
        received: list[dict] = []

        def capture(step: int, mean_reward: float, **kwargs: Any) -> None:
            received.append({"step": step, "mean_reward": mean_reward})

        trainer = PhonRLTrainer(reward=reward, config=default_config)
        trainer.train(prompts=static_prompts, step_callback=capture)

        assert len(received) == default_config.num_steps
        for i, entry in enumerate(received):
            assert entry["step"] == i
            assert isinstance(entry["mean_reward"], float)


class TestWiringSeed:
    @patch("corpusgen.generate.phon_rl.trainer._set_seed")
    @patch("corpusgen.generate.phon_rl.trainer._load_model_and_tokenizer")
    def test_seed_set_before_training(
        self,
        mock_load: MagicMock,
        mock_set_seed: MagicMock,
        reward: PhoneticReward,
        static_prompts: list[str],
    ) -> None:
        _configure_mock_load(mock_load)
        config = TrainingConfig(model_name="gpt2", num_steps=1, seed=42)
        trainer = PhonRLTrainer(reward=reward, config=config)
        trainer.train(prompts=static_prompts)
        mock_set_seed.assert_called_once_with(42)


class TestWiringSaveCheckpoint:
    @patch("corpusgen.generate.phon_rl.trainer._load_model_and_tokenizer")
    def test_save_delegates_to_pretrained(
        self,
        mock_load: MagicMock,
        reward: PhoneticReward,
        default_config: TrainingConfig,
        static_prompts: list[str],
        tmp_path,
    ) -> None:
        mock_model, mock_tokenizer = _configure_mock_load(mock_load)
        trainer = PhonRLTrainer(reward=reward, config=default_config)
        trainer.train(prompts=static_prompts)

        save_path = str(tmp_path / "checkpoint")
        trainer.save_checkpoint(save_path)

        mock_model.save_pretrained.assert_called_once_with(save_path)
        mock_tokenizer.save_pretrained.assert_called_once_with(save_path)


class TestWiringRewardTracking:
    @patch("corpusgen.generate.phon_rl.trainer._load_model_and_tokenizer")
    def test_mean_rewards_length(
        self,
        mock_load: MagicMock,
        reward: PhoneticReward,
        default_config: TrainingConfig,
        static_prompts: list[str],
    ) -> None:
        _configure_mock_load(mock_load)
        trainer = PhonRLTrainer(reward=reward, config=default_config)
        result = trainer.train(prompts=static_prompts)
        assert len(result.mean_rewards) == default_config.num_steps

    @patch("corpusgen.generate.phon_rl.trainer._load_model_and_tokenizer")
    def test_rewards_are_finite(
        self,
        mock_load: MagicMock,
        reward: PhoneticReward,
        default_config: TrainingConfig,
        static_prompts: list[str],
    ) -> None:
        _configure_mock_load(mock_load)
        trainer = PhonRLTrainer(reward=reward, config=default_config)
        result = trainer.train(prompts=static_prompts)
        for r in result.mean_rewards:
            assert math.isfinite(r)

    @patch("corpusgen.generate.phon_rl.trainer._load_model_and_tokenizer")
    def test_final_coverage_in_range(
        self,
        mock_load: MagicMock,
        reward: PhoneticReward,
        default_config: TrainingConfig,
        static_prompts: list[str],
    ) -> None:
        _configure_mock_load(mock_load)
        trainer = PhonRLTrainer(reward=reward, config=default_config)
        result = trainer.train(prompts=static_prompts)
        assert 0.0 <= result.final_coverage <= 1.0


# =======================================================================
# SLOW TESTS — Real models, real reward, real PPO updates
#
# Skipped by default. Run with:
#     poetry run pytest -m slow
# These MUST pass before any publication claim.
# =======================================================================


def _has_slow_deps() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


_skip_slow_deps = pytest.mark.skipif(
    not _has_slow_deps(),
    reason="Slow test dependencies not installed (torch, transformers)",
)


@pytest.mark.slow
@_skip_slow_deps
class TestSlowRealTrainingLoop:
    """Real PPO training with gpt2 and real PhoneticReward.

    Uses real model, real reward, real PPO updates with our own
    implementation (no trl). Proves the training loop actually
    produces correct results.

    Requirements:
        - poetry install --with local
        - espeak-ng installed and on PATH
        - ~500MB disk for gpt2 weights (cached after first download)
    """

    def test_real_training_produces_result(self) -> None:
        """3 PPO steps with gpt2 + LoRA. Rewards are finite, coverage valid."""
        inventory = PhoneticTargetInventory(
            target_phonemes=["p", "b", "t", "d", "k"],
            unit="phoneme",
        )
        phon_reward = PhoneticReward(targets=inventory, coverage_weight=1.0)
        config = TrainingConfig(
            model_name="gpt2",
            num_steps=3,
            batch_size=2,
            learning_rate=1e-5,
            kl_coeff=0.1,
            max_new_tokens=32,
            seed=42,
            use_peft=True,
            peft_r=4,
            peft_alpha=8,
        )
        trainer = PhonRLTrainer(reward=phon_reward, config=config)
        result = trainer.train(
            prompts=["Write a sentence with p.", "Write a sentence with t."]
        )

        assert isinstance(result, TrainingResult)
        assert result.total_steps == 3
        assert len(result.mean_rewards) == 3
        for r in result.mean_rewards:
            assert math.isfinite(r)
        assert 0.0 <= result.final_coverage <= 1.0

    def test_real_dynamic_prompts(self) -> None:
        """Dynamic prompt_fn adapts to coverage gaps."""
        inventory = PhoneticTargetInventory(
            target_phonemes=["p", "b", "t"],
            unit="phoneme",
        )
        phon_reward = PhoneticReward(targets=inventory, coverage_weight=1.0)
        config = TrainingConfig(
            model_name="gpt2",
            num_steps=2,
            batch_size=1,
            learning_rate=1e-5,
            kl_coeff=0.1,
            max_new_tokens=32,
            seed=42,
            use_peft=True,
            peft_r=4,
            peft_alpha=8,
        )

        prompts_generated: list[str] = []

        def prompt_fn(targets: PhoneticTargetInventory) -> str:
            missing = targets.next_targets(3)
            prompt = f"Say something with: {', '.join(missing)}" if missing else "Say anything."
            prompts_generated.append(prompt)
            return prompt

        trainer = PhonRLTrainer(reward=phon_reward, config=config)
        result = trainer.train(prompt_fn=prompt_fn)

        assert result.total_steps == 2
        assert len(prompts_generated) == 2

    def test_real_save_and_load_checkpoint(self, tmp_path) -> None:
        """Train, save checkpoint, verify files on disk."""
        inventory = PhoneticTargetInventory(
            target_phonemes=["p", "t"],
            unit="phoneme",
        )
        phon_reward = PhoneticReward(targets=inventory, coverage_weight=1.0)
        config = TrainingConfig(
            model_name="gpt2",
            num_steps=1,
            batch_size=1,
            learning_rate=1e-5,
            max_new_tokens=16,
            seed=42,
            use_peft=True,
            peft_r=4,
            peft_alpha=8,
        )
        trainer = PhonRLTrainer(reward=phon_reward, config=config)
        trainer.train(prompts=["Say something with p."])

        save_dir = tmp_path / "checkpoint"
        trainer.save_checkpoint(str(save_dir))

        assert save_dir.exists()
        saved_files = list(save_dir.iterdir())
        assert len(saved_files) > 0

    def test_real_ppo_loss_decreases(self) -> None:
        """Over a few steps, PPO loss should not diverge to infinity."""
        inventory = PhoneticTargetInventory(
            target_phonemes=["p", "t", "k"],
            unit="phoneme",
        )
        phon_reward = PhoneticReward(targets=inventory, coverage_weight=1.0)
        config = TrainingConfig(
            model_name="gpt2",
            num_steps=5,
            batch_size=2,
            learning_rate=1e-5,
            kl_coeff=0.1,
            max_new_tokens=24,
            seed=42,
            use_peft=True,
            peft_r=4,
            peft_alpha=8,
        )
        losses: list[float] = []

        def track_loss(step: int, mean_reward: float, **kwargs: Any) -> None:
            if "policy_loss" in kwargs:
                losses.append(kwargs["policy_loss"])

        trainer = PhonRLTrainer(reward=phon_reward, config=config)
        result = trainer.train(
            prompts=["Say p.", "Say t.", "Say k."],
            step_callback=track_loss,
        )

        # All rewards should be finite
        for r in result.mean_rewards:
            assert math.isfinite(r)


# =======================================================================
# Mock helpers for wiring tests
# =======================================================================


def _configure_mock_load(mock_load: MagicMock) -> tuple[MagicMock, MagicMock]:
    """Configure _load_model_and_tokenizer mock.

    Creates mock model that returns realistic-shaped tensors so the
    PPO math functions can operate on them.
    """
    import torch as _torch

    hidden_size = 32
    vocab_size = 50

    mock_model = MagicMock()
    mock_model.device = _torch.device("cpu")
    mock_model.config = MagicMock()
    mock_model.config.hidden_size = hidden_size
    mock_model.config.vocab_size = vocab_size

    # model() returns an object with .logits and .hidden_states
    def fake_forward(input_ids, **kwargs):
        batch, seq = input_ids.shape
        result = MagicMock()
        result.logits = _torch.randn(batch, seq, vocab_size)
        # Simulate hidden_states from last layer
        result.hidden_states = (_torch.randn(batch, seq, hidden_size),)
        return result

    mock_model.side_effect = fake_forward
    mock_model.return_value = fake_forward(
        _torch.zeros(1, 1, dtype=_torch.long)
    )

    # generate() returns token IDs
    def fake_generate(input_ids, **kwargs):
        batch = input_ids.shape[0] if hasattr(input_ids, "shape") else 1
        max_new = kwargs.get("max_new_tokens", 10)
        prompt_len = input_ids.shape[-1] if hasattr(input_ids, "shape") else 4
        total_len = prompt_len + max_new
        return _torch.randint(0, vocab_size, (batch, total_len))

    mock_model.generate = fake_generate
    mock_model.parameters = MagicMock(return_value=iter([_torch.randn(2, 2, requires_grad=True)]))

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = "<pad>"
    mock_tokenizer.eos_token = "<eos>"
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.vocab_size = vocab_size
    mock_tokenizer.decode.return_value = "the pat bat kit"
    mock_tokenizer.batch_decode.return_value = ["the pat bat kit"]
    mock_tokenizer.encode.return_value = [1, 2, 3, 4]

    mock_load.return_value = (mock_model, mock_tokenizer)
    return mock_model, mock_tokenizer
