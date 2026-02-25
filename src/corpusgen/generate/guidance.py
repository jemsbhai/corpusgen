"""GuidanceStrategy ABC: inference-time guidance for local generation.

Defines the interface that Phon-DATG (logit steering) and Phon-RL
(policy-adjusted generation) must implement to plug into the
LocalBackend's generation pipeline.

The contract:
    1. ``prepare()`` is called once before generating a batch of
       candidates, allowing the strategy to precompute logit biases,
       attribute words, value estimates, etc.
    2. ``modify_logits()`` is called at each autoregressive step
       to adjust the model's raw logits before sampling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class GuidanceStrategy(ABC):
    """Abstract base class for inference-time guidance mechanisms.

    Subclasses must implement:
        - ``name``: identifier string for logging and results
        - ``prepare(target_units, model, tokenizer)``: one-time setup
          before a generation call
        - ``modify_logits(input_ids, logits)``: per-step logit adjustment

    Both Phon-DATG and Phon-RL will implement this interface, allowing
    the LocalBackend to use either (or neither) transparently.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier string."""

    @abstractmethod
    def prepare(
        self,
        target_units: list[str],
        model: Any,
        tokenizer: Any,
    ) -> None:
        """Prepare the strategy for a generation call.

        Called once before ``model.generate()`` to allow precomputation
        of logit biases, attribute word sets, value estimates, etc.

        Args:
            target_units: Phonetic units to target in this generation.
            model: The HuggingFace model instance.
            tokenizer: The HuggingFace tokenizer instance.
        """

    @abstractmethod
    def modify_logits(self, input_ids: Any, logits: Any) -> Any:
        """Modify logits at a single autoregressive generation step.

        Args:
            input_ids: Current token sequence (``torch.Tensor``
                of shape ``[batch, seq_len]``).
            logits: Raw logits from the model (``torch.Tensor``
                of shape ``[batch, vocab_size]``).

        Returns:
            Modified logits (``torch.Tensor``, same shape as input).
        """
