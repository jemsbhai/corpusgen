"""PhonRLTrainer: PPO training loop for Phon-RL.

Implements Proximal Policy Optimization (Schulman et al., 2017) from
scratch using raw PyTorch. No dependency on trl — we own the full
training loop, making the contribution clear and avoiding coupling
to fast-moving external APIs.

Training loop per step:
    1. Generate text from current policy given a prompt
    2. Forward pass through policy → log probs, hidden states
    3. Forward pass through frozen reference model → ref log probs
    4. Value head estimates expected return from hidden states
    5. Compute sentence-level phonetic reward (PhoneticReward)
    6. Per-token KL penalty: reward_t -= kl_coeff * KL(policy || ref)
    7. GAE advantage estimation from rewards and values
    8. PPO clipped surrogate loss + value loss
    9. Backward pass and optimizer step

Supports two prompt modes:
    - **Static**: a list of prompt strings, cycled through
    - **Dynamic**: a callable receiving PhoneticTargetInventory,
      enabling adaptive targeting of coverage gaps

All heavy dependencies (torch, transformers, peft) are imported lazily
and model loading is isolated into a mockable helper function.

References:
    Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O.
    (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Callable

from corpusgen.generate.phon_ctg.targets import PhoneticTargetInventory
from corpusgen.generate.phon_rl.reward import PhoneticReward

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Result / config dataclasses
# -----------------------------------------------------------------------


@dataclass
class TrainingResult:
    """Result from a completed PPO training run.

    Attributes:
        mean_rewards: Mean composite reward per training step.
        total_steps: Total number of PPO steps completed.
        final_coverage: Coverage fraction at end of training (0.0–1.0).
        checkpoint_path: Path where the model was saved, or None.
    """

    mean_rewards: list[float]
    total_steps: int
    final_coverage: float
    checkpoint_path: str | None


@dataclass
class TrainingConfig:
    """Configuration for PhonRLTrainer.

    Args:
        model_name: HuggingFace model ID or local path.
        num_steps: Number of PPO training steps.
        batch_size: Number of sequences generated per step.
        learning_rate: Learning rate for the optimizer.
        kl_coeff: KL penalty coefficient (constrains policy drift).
        clip_epsilon: PPO clipping parameter (Schulman et al., 2017).
        gae_gamma: Discount factor for GAE.
        gae_lambda: GAE lambda for bias-variance tradeoff.
        value_loss_coeff: Coefficient for value function loss.
        output_dir: Directory for checkpoints and logs.
        seed: Random seed for reproducibility.
        max_new_tokens: Maximum new tokens per generated sequence.
        temperature: Sampling temperature.
        use_peft: Whether to use PEFT/LoRA for parameter-efficient training.
        peft_r: LoRA rank (when use_peft is True).
        peft_alpha: LoRA alpha scaling (when use_peft is True).
    """

    model_name: str
    num_steps: int = 100
    batch_size: int = 4
    learning_rate: float = 1.41e-5
    kl_coeff: float = 0.1
    clip_epsilon: float = 0.2
    gae_gamma: float = 1.0
    gae_lambda: float = 0.95
    value_loss_coeff: float = 0.5
    output_dir: str | None = None
    seed: int = 42
    max_new_tokens: int = 64
    temperature: float = 0.8
    device: str | None = None  # None = auto-detect (CUDA if available, else CPU)
    use_peft: bool = False
    peft_r: int = 8
    peft_alpha: int = 16

    def __post_init__(self) -> None:
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be > 0, got {self.num_steps}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be > 0, got {self.learning_rate}"
            )
        if self.kl_coeff < 0:
            raise ValueError(f"kl_coeff must be >= 0, got {self.kl_coeff}")
        if self.clip_epsilon < 0:
            raise ValueError(
                f"clip_epsilon must be >= 0, got {self.clip_epsilon}"
            )


# -----------------------------------------------------------------------
# PPO math functions (pure torch, no side effects)
#
# These implement the core RL algorithms from Schulman et al. (2017).
# They operate on tensors and are independently testable.
# -----------------------------------------------------------------------


def compute_log_probs_from_logits(
    logits: Any,
    actions: Any,
) -> Any:
    """Compute log probabilities for selected actions given logits.

    Args:
        logits: Model output logits, shape ``[batch, seq_len, vocab_size]``.
        actions: Selected token IDs, shape ``[batch, seq_len]``.

    Returns:
        Log probabilities for the selected actions,
        shape ``[batch, seq_len]``.
    """
    import torch

    log_probs = torch.log_softmax(logits, dim=-1)
    # Gather log probs for the selected actions
    return log_probs.gather(
        dim=-1,
        index=actions.unsqueeze(-1),
    ).squeeze(-1)


def compute_kl_penalty(
    policy_log_probs: Any,
    ref_log_probs: Any,
) -> Any:
    """Compute per-token KL divergence between policy and reference.

    Uses the approximation KL ≈ exp(log_policy - log_ref) - (log_policy - log_ref) - 1,
    which is the exact KL for the selected action and is numerically
    stable (always non-negative).

    Args:
        policy_log_probs: Log probs under current policy,
            shape ``[batch, seq_len]``.
        ref_log_probs: Log probs under reference model,
            shape ``[batch, seq_len]``.

    Returns:
        Per-token KL divergence, shape ``[batch, seq_len]``.
    """
    log_ratio = policy_log_probs - ref_log_probs
    # Schulman's KL estimator k3: exp(log_ratio) - log_ratio - 1
    # This is always >= 0 and equals 0 when policy == ref
    return (log_ratio.exp() - log_ratio - 1.0)


def compute_gae(
    rewards: Any,
    values: Any,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> tuple[Any, Any]:
    """Generalized Advantage Estimation (Schulman et al., 2016).

    Computes advantages and returns from per-token rewards and value
    estimates. Assumes the episode terminates after the last token
    (next_value = 0).

    Args:
        rewards: Per-token rewards, shape ``[batch, seq_len]``.
        values: Value estimates, shape ``[batch, seq_len]``.
        gamma: Discount factor.
        lam: GAE lambda for bias-variance tradeoff.

    Returns:
        Tuple of (advantages, returns), each shape ``[batch, seq_len]``.
    """
    import torch

    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(batch_size, device=rewards.device)

    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            next_value = torch.zeros(batch_size, device=rewards.device)
        else:
            next_value = values[:, t + 1]

        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_advantage = delta + gamma * lam * last_advantage
        advantages[:, t] = last_advantage

    returns = advantages + values
    return advantages, returns


def ppo_clip_loss(
    advantages: Any,
    old_log_probs: Any,
    new_log_probs: Any,
    clip_epsilon: float = 0.2,
) -> Any:
    """PPO clipped surrogate objective (Schulman et al., 2017).

    L = -E[min(r * A, clip(r, 1-ε, 1+ε) * A)]

    where r = exp(new_log_probs - old_log_probs) is the probability ratio.

    Args:
        advantages: GAE advantages, shape ``[batch, seq_len]``.
        old_log_probs: Log probs from the old policy (before update),
            shape ``[batch, seq_len]``.
        new_log_probs: Log probs from the current policy,
            shape ``[batch, seq_len]``.
        clip_epsilon: Clipping parameter.

    Returns:
        Scalar loss (mean over all tokens).
    """
    import torch

    ratio = (new_log_probs - old_log_probs).exp()
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)

    surrogate1 = ratio * advantages
    surrogate2 = clipped_ratio * advantages

    return -torch.min(surrogate1, surrogate2).mean()


# -----------------------------------------------------------------------
# Isolated helpers (mockable at library boundary)
# -----------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _detect_device(device: str | None) -> str:
    """Auto-detect the best device if none specified."""
    if device is not None:
        return device
    try:
        import torch
        if torch.cuda.is_available():
            detected = "cuda"
            logger.info(
                "CUDA detected: %s (%s)",
                torch.cuda.get_device_name(0),
                f"{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB",
            )
        else:
            detected = "cpu"
            logger.info("CUDA not available, using CPU.")
        return detected
    except ImportError:
        return "cpu"


def _load_model_and_tokenizer(
    model_name: str,
    device: str = "cpu",
    use_peft: bool = False,
    peft_r: int = 8,
    peft_alpha: int = 16,
) -> tuple[Any, Any]:
    """Load model and tokenizer for PPO training.

    Places the model on the specified device. When use_peft is True,
    wraps the model with a LoRA adapter.

    Args:
        model_name: HuggingFace model ID or local path.
        device: Target device ("cuda", "cpu", or specific like "cuda:0").
        use_peft: Whether to apply PEFT/LoRA.
        peft_r: LoRA rank.
        peft_alpha: LoRA alpha scaling.

    Returns:
        Tuple of (model, tokenizer).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model %s on device=%s", model_name, device)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)

    if use_peft:
        from peft import LoraConfig, get_peft_model  # type: ignore[import-untyped]

        lora_config = LoraConfig(
            r=peft_r,
            lora_alpha=peft_alpha,
            task_type="CAUSAL_LM",
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        logger.info(
            "PEFT/LoRA applied (r=%d, alpha=%d). Trainable params: %d",
            peft_r,
            peft_alpha,
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )

    return model, tokenizer


# -----------------------------------------------------------------------
# PhonRLTrainer
# -----------------------------------------------------------------------


class PhonRLTrainer:
    """Orchestrates PPO training with a phonetic composite reward.

    Implements the full PPO loop from Schulman et al. (2017):
    generate → reward → GAE → clipped policy update. No dependency
    on trl — the entire training loop is self-contained.

    Args:
        reward: PhoneticReward instance providing the reward signal.
        config: TrainingConfig with model, training, and PEFT settings.
    """

    def __init__(
        self,
        reward: PhoneticReward,
        config: TrainingConfig,
    ) -> None:
        self._reward = reward
        self._config = config
        self._model: Any = None
        self._ref_model: Any = None
        self._tokenizer: Any = None
        self._value_head: Any = None
        self._optimizer: Any = None
        self._initialized = False

    # -------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------

    @property
    def reward(self) -> PhoneticReward:
        """The phonetic reward function."""
        return self._reward

    @property
    def config(self) -> TrainingConfig:
        """Training configuration."""
        return self._config

    @property
    def is_initialized(self) -> bool:
        """Whether train() has been called and completed."""
        return self._initialized

    # -------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------

    def train(
        self,
        prompts: list[str] | None = None,
        prompt_fn: Callable[[PhoneticTargetInventory], str] | None = None,
        step_callback: Callable[..., None] | None = None,
    ) -> TrainingResult:
        """Run PPO training with phonetic reward.

        Exactly one of ``prompts`` or ``prompt_fn`` must be provided.

        Args:
            prompts: Static list of prompt strings, cycled through.
            prompt_fn: Dynamic callable receiving the current
                PhoneticTargetInventory, returning a prompt string.
            step_callback: Optional callback invoked after each step
                with keyword arguments: ``step`` (int),
                ``mean_reward`` (float), ``policy_loss`` (float).

        Returns:
            TrainingResult with per-step rewards and final coverage.
        """
        import torch

        # --- Validate prompt args ---
        self._validate_prompt_args(prompts, prompt_fn)

        # --- Set seed ---
        _set_seed(self._config.seed)

        # --- Detect device ---
        device = _detect_device(self._config.device)

        # --- Load model and tokenizer ---
        self._model, self._tokenizer = _load_model_and_tokenizer(
            model_name=self._config.model_name,
            device=device,
            use_peft=self._config.use_peft,
            peft_r=self._config.peft_r,
            peft_alpha=self._config.peft_alpha,
        )

        # --- Create frozen reference model ---
        self._ref_model = copy.deepcopy(self._model)
        self._ref_model.eval()
        for param in self._ref_model.parameters():
            param.requires_grad = False

        # --- Create value head ---
        from corpusgen.generate.phon_rl.value_head import ValueHead

        hidden_size = self._model.config.hidden_size
        self._value_head = ValueHead(hidden_size=hidden_size)
        device = next(self._model.parameters()).device
        self._value_head = self._value_head.to(device)

        # --- Create optimizer (policy + value head) ---
        trainable_params = [
            p for p in self._model.parameters() if p.requires_grad
        ]
        trainable_params += list(self._value_head.parameters())
        self._optimizer = torch.optim.Adam(
            trainable_params,
            lr=self._config.learning_rate,
        )

        # --- Training loop ---
        mean_rewards: list[float] = []
        targets = self._reward.targets
        sentence_index = 0

        for step in range(self._config.num_steps):
            # Get prompt
            if prompt_fn is not None:
                prompt_text = prompt_fn(targets)
            else:
                assert prompts is not None
                prompt_text = prompts[step % len(prompts)]

            # Encode prompt
            prompt_ids = self._tokenizer.encode(prompt_text)
            prompt_len = len(prompt_ids)

            # Generate response from current policy
            input_tensor = torch.tensor(
                [prompt_ids], device=device
            )
            with torch.no_grad():
                output_ids = self._model.generate(
                    input_tensor,
                    max_new_tokens=self._config.max_new_tokens,
                    temperature=self._config.temperature,
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            # Extract response token IDs
            full_ids = output_ids[0]  # [total_len]
            response_ids = full_ids[prompt_len:]
            if len(response_ids) == 0:
                mean_rewards.append(0.0)
                continue

            # Decode and compute sentence-level reward
            response_text = self._tokenizer.decode(
                response_ids, skip_special_tokens=True
            )
            phonemes = PhoneticReward._simple_char_phonemes(response_text)
            reward_result = self._reward.commit_sentence_reward(
                phonemes=phonemes,
                text=response_text,
                sentence_index=sentence_index,
            )
            sentence_index += 1
            step_reward = reward_result.composite_reward

            # --- PPO update ---
            policy_loss_val = self._ppo_step(
                full_ids=full_ids.unsqueeze(0),
                prompt_len=prompt_len,
                sentence_reward=step_reward,
            )

            mean_rewards.append(step_reward)

            # Callback
            if step_callback is not None:
                step_callback(
                    step=step,
                    mean_reward=step_reward,
                    policy_loss=policy_loss_val,
                )

            logger.info(
                "Step %d/%d: reward=%.4f, coverage=%.2f%%, policy_loss=%.4f",
                step + 1,
                self._config.num_steps,
                step_reward,
                targets.coverage * 100,
                policy_loss_val,
            )

        self._initialized = True

        return TrainingResult(
            mean_rewards=mean_rewards,
            total_steps=self._config.num_steps,
            final_coverage=targets.coverage,
            checkpoint_path=None,
        )

    # -------------------------------------------------------------------
    # PPO step (single update)
    # -------------------------------------------------------------------

    def _ppo_step(
        self,
        full_ids: Any,
        prompt_len: int,
        sentence_reward: float,
    ) -> float:
        """Execute a single PPO update step.

        Args:
            full_ids: Full token sequence [1, total_len] (prompt + response).
            prompt_len: Length of the prompt portion.
            sentence_reward: Scalar sentence-level reward.

        Returns:
            Policy loss value (float) for logging.
        """
        import torch

        response_len = full_ids.shape[1] - prompt_len
        if response_len <= 0:
            return 0.0

        response_ids = full_ids[:, prompt_len:]  # [1, response_len]

        # --- Forward pass: old policy log probs (detached) ---
        with torch.no_grad():
            old_outputs = self._model(
                full_ids, output_hidden_states=True
            )
            old_logits = old_outputs.logits[:, prompt_len - 1:-1, :]
            old_log_probs = compute_log_probs_from_logits(
                old_logits, response_ids
            )

            # Reference model log probs for KL penalty
            ref_outputs = self._ref_model(
                full_ids, output_hidden_states=True
            )
            ref_logits = ref_outputs.logits[:, prompt_len - 1:-1, :]
            ref_log_probs = compute_log_probs_from_logits(
                ref_logits, response_ids
            )

            # Value estimates from hidden states
            hidden = old_outputs.hidden_states[-1][:, prompt_len:, :]
            values = self._value_head(hidden).detach()

        # --- Compute per-token rewards ---
        # Distribute sentence reward to last token (terminal reward)
        per_token_rewards = torch.zeros(
            1, response_len, device=full_ids.device
        )
        per_token_rewards[0, -1] = sentence_reward

        # KL penalty
        kl = compute_kl_penalty(old_log_probs, ref_log_probs)
        per_token_rewards = per_token_rewards - self._config.kl_coeff * kl

        # --- GAE ---
        advantages, returns = compute_gae(
            per_token_rewards,
            values,
            gamma=self._config.gae_gamma,
            lam=self._config.gae_lambda,
        )
        # Normalize advantages
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )

        # --- PPO update (forward pass with gradients) ---
        self._optimizer.zero_grad()

        new_outputs = self._model(full_ids, output_hidden_states=True)
        new_logits = new_outputs.logits[:, prompt_len - 1:-1, :]
        new_log_probs = compute_log_probs_from_logits(
            new_logits, response_ids
        )

        # Policy loss (clipped surrogate)
        policy_loss = ppo_clip_loss(
            advantages,
            old_log_probs,
            new_log_probs,
            clip_epsilon=self._config.clip_epsilon,
        )

        # Value loss
        new_hidden = new_outputs.hidden_states[-1][:, prompt_len:, :]
        new_values = self._value_head(new_hidden)
        value_loss = ((new_values - returns) ** 2).mean()

        # Total loss
        total_loss = policy_loss + self._config.value_loss_coeff * value_loss
        total_loss.backward()
        self._optimizer.step()

        return policy_loss.item()

    # -------------------------------------------------------------------
    # Save checkpoint
    # -------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save the trained model and tokenizer to disk.

        Args:
            path: Directory path for the checkpoint.

        Raises:
            RuntimeError: If train() has not been called yet.
        """
        if not self._initialized:
            raise RuntimeError(
                "Cannot save checkpoint before training. "
                "Call train() first."
            )
        self._model.save_pretrained(path)
        self._tokenizer.save_pretrained(path)
        logger.info("Checkpoint saved to %s", path)

    # -------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------

    @staticmethod
    def _validate_prompt_args(
        prompts: list[str] | None,
        prompt_fn: Callable[[PhoneticTargetInventory], str] | None,
    ) -> None:
        """Validate that exactly one prompt source is provided."""
        if prompts is not None and prompt_fn is not None:
            raise ValueError(
                "Provide exactly one of 'prompts' or 'prompt_fn', not both."
            )
        if prompts is None and prompt_fn is None:
            raise ValueError(
                "Provide exactly one of 'prompts' or 'prompt_fn'."
            )
        if prompts is not None and len(prompts) == 0:
            raise ValueError("'prompts' must not be empty.")
