"""ValueHead: lightweight linear value estimator for PPO.

Maps hidden states from a language model to scalar value estimates.
Used in Phon-RL's PPO loop to compute advantages via Generalized
Advantage Estimation (GAE).

Architecture:
    hidden_states [batch, seq_len, hidden_size]
        → dropout (optional)
        → linear (hidden_size → 1)
        → squeeze → [batch, seq_len]

The value head is intentionally minimal — a single linear projection.
This keeps the trainable parameter count small and avoids interfering
with the LM's learned representations.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """Linear value head for PPO advantage estimation.

    Projects language model hidden states to scalar value estimates.
    Supports both 2D ``[batch, hidden]`` and 3D
    ``[batch, seq_len, hidden]`` inputs.

    Args:
        hidden_size: Dimensionality of the input hidden states.
            Must match the LM's hidden size.
        dropout: Dropout probability applied before the linear layer.
            Default 0.0 (no dropout).
    """

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.0,
    ) -> None:
        if hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be > 0, got {hidden_size}"
            )
        if dropout < 0.0 or dropout > 1.0:
            raise ValueError(
                f"dropout must be in [0.0, 1.0], got {dropout}"
            )

        super().__init__()

        self._hidden_size = hidden_size
        self._dropout_rate = dropout

        self._dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self._linear = nn.Linear(hidden_size, 1)

    # -------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------

    @property
    def hidden_size(self) -> int:
        """Dimensionality of input hidden states."""
        return self._hidden_size

    @property
    def dropout_rate(self) -> float:
        """Dropout probability."""
        return self._dropout_rate

    # -------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to scalar value estimates.

        Args:
            hidden_states: Input tensor of shape ``[batch, hidden]``
                or ``[batch, seq_len, hidden]``.

        Returns:
            Value estimates. Shape ``[batch]`` for 2D input,
            ``[batch, seq_len]`` for 3D input.
        """
        x = self._dropout(hidden_states)
        values = self._linear(x)  # [..., 1]
        return values.squeeze(-1)
