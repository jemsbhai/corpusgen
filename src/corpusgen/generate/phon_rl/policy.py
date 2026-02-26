"""PhonRLStrategy: GuidanceStrategy wrapper for RL-fine-tuned models.

After PPO training via PhonRLTrainer, the model itself generates
phonetically-targeted text — no inference-time logit modification is
needed. This strategy provides a thin GuidanceStrategy-compliant
wrapper so that RL-fine-tuned models plug into LocalBackend seamlessly.

The strategy optionally loads a PEFT/LoRA adapter checkpoint onto the
base model during ``prepare()``. Once loaded, ``modify_logits()`` is
identity (pass-through) since the model's policy already incorporates
the phonetic objective.

Usage:
    # No adapter (base model already fine-tuned in-place):
    strategy = PhonRLStrategy()

    # With PEFT adapter checkpoint:
    strategy = PhonRLStrategy(adapter_path="./checkpoints/phon_rl_adapter")

    # Plug into LocalBackend:
    backend = LocalBackend(model_name="...", guidance_strategy=strategy)
"""

from __future__ import annotations

import logging
from typing import Any

from corpusgen.generate.guidance import GuidanceStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Isolated helper (mockable in tests)
# ---------------------------------------------------------------------------


def _load_peft_adapter(model: Any, adapter_path: str) -> Any:
    """Load a PEFT adapter onto a HuggingFace model.

    Isolated for mockability in tests.

    Args:
        model: The base HuggingFace model instance.
        adapter_path: Path to the PEFT adapter checkpoint directory.

    Returns:
        The model with the adapter loaded and merged.

    Raises:
        ImportError: If peft is not installed.
        FileNotFoundError: If the adapter path does not exist.
    """
    from peft import PeftModel  # type: ignore[import-untyped]

    logger.info("Loading PEFT adapter from %s", adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    logger.info("PEFT adapter loaded and merged successfully.")
    return model


# ---------------------------------------------------------------------------
# PhonRLStrategy
# ---------------------------------------------------------------------------


class PhonRLStrategy(GuidanceStrategy):
    """GuidanceStrategy wrapper for Phon-RL fine-tuned models.

    After PPO training, the model's policy already incorporates the
    phonetic objective. This strategy:

    1. Optionally loads a PEFT/LoRA adapter during ``prepare()``
    2. Returns logits unchanged in ``modify_logits()`` (identity)

    This allows RL-fine-tuned models to be used with LocalBackend
    via the same GuidanceStrategy interface as Phon-DATG.

    Args:
        adapter_path: Optional path to a PEFT adapter checkpoint.
            If None, the base model is used as-is (assumed to be
            already fine-tuned or used without adaptation).
    """

    def __init__(self, adapter_path: str | None = None) -> None:
        self._adapter_path = adapter_path
        self._adapter_loaded = adapter_path is None  # trivially loaded if no adapter
        self._current_target_units: list[str] | None = None

    # -------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Strategy identifier."""
        return "phon_rl"

    @property
    def adapter_path(self) -> str | None:
        """Path to the PEFT adapter checkpoint, or None."""
        return self._adapter_path

    @property
    def is_adapter_loaded(self) -> bool:
        """Whether the adapter has been loaded (or no adapter is needed)."""
        return self._adapter_loaded

    @property
    def current_target_units(self) -> list[str] | None:
        """Target units from the most recent prepare() call, or None."""
        return self._current_target_units

    # -------------------------------------------------------------------
    # GuidanceStrategy interface
    # -------------------------------------------------------------------

    def prepare(
        self,
        target_units: list[str],
        model: Any,
        tokenizer: Any,
    ) -> None:
        """Prepare the strategy for a generation call.

        Loads the PEFT adapter onto the model on the first call (if
        an adapter_path was provided). Subsequent calls skip loading.
        Stores target_units for potential prompt formatting use.

        Args:
            target_units: Phonetic units to target in this generation.
            model: The HuggingFace model instance.
            tokenizer: The HuggingFace tokenizer instance (unused).
        """
        self._current_target_units = target_units

        if not self._adapter_loaded and self._adapter_path is not None:
            _load_peft_adapter(model, self._adapter_path)
            self._adapter_loaded = True
            logger.info(
                "PhonRLStrategy prepared with adapter from %s",
                self._adapter_path,
            )

    def modify_logits(self, input_ids: Any, logits: Any) -> Any:
        """Return logits unchanged (identity pass-through).

        The RL-fine-tuned model's policy already incorporates the
        phonetic objective, so no inference-time logit modification
        is needed.

        Args:
            input_ids: Current token sequence (unused).
            logits: Raw logits from the model.

        Returns:
            The same logits object, unmodified.
        """
        return logits
