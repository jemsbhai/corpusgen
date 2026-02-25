"""LogitModulator: additive logit adjustments for Phon-DATG.

Applies boost and penalty adjustments to logit tensors during
autoregressive generation. Attribute tokens (containing target
phonetic units) are boosted; anti-attribute tokens (containing only
already-covered units) are penalized.

This module is stateless — it performs pure tensor math. The
DATGStrategy in ``graph.py`` manages which tokens are attribute
vs anti-attribute and delegates the actual logit modification here.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class LogitModulator:
    """Applies additive logit adjustments to attribute/anti-attribute tokens.

    Args:
        boost_strength: Additive boost for attribute token logits.
            Must be >= 0.
        penalty_strength: Additive penalty for anti-attribute token logits.
            Must be <= 0.
    """

    def __init__(
        self,
        boost_strength: float = 5.0,
        penalty_strength: float = -5.0,
    ) -> None:
        if boost_strength < 0:
            raise ValueError(
                f"boost_strength must be >= 0, got {boost_strength}"
            )
        if penalty_strength > 0:
            raise ValueError(
                f"penalty_strength must be <= 0, got {penalty_strength}"
            )
        self._boost_strength = boost_strength
        self._penalty_strength = penalty_strength

    # -------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------

    @property
    def boost_strength(self) -> float:
        """Additive boost for attribute tokens."""
        return self._boost_strength

    @property
    def penalty_strength(self) -> float:
        """Additive penalty for anti-attribute tokens."""
        return self._penalty_strength

    # -------------------------------------------------------------------
    # Modulate
    # -------------------------------------------------------------------

    def modulate(
        self,
        logits: Any,
        attribute_token_ids: set[int],
        anti_attribute_token_ids: set[int],
    ) -> Any:
        """Apply additive logit adjustments.

        Does not mutate the input logits; returns a modified clone.

        Args:
            logits: Raw logits tensor with shape ``[batch, vocab_size]``.
                Must support ``.clone()`` and indexing via
                ``[batch_idx, token_id]``.
            attribute_token_ids: Token IDs to boost.
            anti_attribute_token_ids: Token IDs to penalize.

        Returns:
            Modified logits (same type and shape as input).
        """
        modified = logits.clone()
        vocab_size = modified.shape[-1]

        # Filter to valid token IDs
        valid_boost = [
            tid for tid in attribute_token_ids if 0 <= tid < vocab_size
        ]
        valid_penalty = [
            tid for tid in anti_attribute_token_ids if 0 <= tid < vocab_size
        ]

        # Apply adjustments across all batch entries
        for b in range(modified.shape[0]):
            for tid in valid_boost:
                modified[b, tid] = modified[b, tid] + self._boost_strength
            for tid in valid_penalty:
                modified[b, tid] = modified[b, tid] + self._penalty_strength

        return modified
