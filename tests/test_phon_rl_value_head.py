"""Tests for phon_rl.value_head — ValueHead linear module."""

from __future__ import annotations

import pytest

# Value head tests require torch
torch = pytest.importorskip("torch")

from corpusgen.generate.phon_rl.value_head import ValueHead


# -----------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------


class TestConstruction:
    """Tests for ValueHead construction and properties."""

    def test_basic_construction(self) -> None:
        head = ValueHead(hidden_size=768)
        assert head.hidden_size == 768

    def test_custom_hidden_size(self) -> None:
        head = ValueHead(hidden_size=1024)
        assert head.hidden_size == 1024

    def test_with_dropout(self) -> None:
        head = ValueHead(hidden_size=768, dropout=0.1)
        assert head.dropout_rate == 0.1

    def test_zero_hidden_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="hidden_size"):
            ValueHead(hidden_size=0)

    def test_negative_hidden_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="hidden_size"):
            ValueHead(hidden_size=-1)

    def test_negative_dropout_rejected(self) -> None:
        with pytest.raises(ValueError, match="dropout"):
            ValueHead(hidden_size=768, dropout=-0.1)

    def test_dropout_above_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="dropout"):
            ValueHead(hidden_size=768, dropout=1.5)

    def test_is_nn_module(self) -> None:
        head = ValueHead(hidden_size=768)
        assert isinstance(head, torch.nn.Module)

    def test_parameter_count(self) -> None:
        """Should have a small number of trainable parameters."""
        head = ValueHead(hidden_size=768)
        total_params = sum(p.numel() for p in head.parameters())
        # At minimum: one linear layer (768 weights + 1 bias = 769)
        assert total_params > 0
        # Should be small relative to an LM
        assert total_params < 100_000


# -----------------------------------------------------------------------
# Forward pass — shape correctness
# -----------------------------------------------------------------------


class TestForward:
    """Tests for forward pass shape and dtype behavior."""

    def test_output_shape_2d(self) -> None:
        """[batch, hidden] -> [batch, 1] squeezed to [batch]."""
        head = ValueHead(hidden_size=64)
        x = torch.randn(4, 64)
        out = head(x)
        assert out.shape == (4,)

    def test_output_shape_3d(self) -> None:
        """[batch, seq_len, hidden] -> [batch, seq_len]."""
        head = ValueHead(hidden_size=64)
        x = torch.randn(2, 10, 64)
        out = head(x)
        assert out.shape == (2, 10)

    def test_output_dtype_float(self) -> None:
        head = ValueHead(hidden_size=64)
        x = torch.randn(2, 10, 64)
        out = head(x)
        assert out.dtype == torch.float32

    def test_single_token(self) -> None:
        """[1, 1, hidden] -> [1, 1]."""
        head = ValueHead(hidden_size=32)
        x = torch.randn(1, 1, 32)
        out = head(x)
        assert out.shape == (1, 1)

    def test_batch_size_one(self) -> None:
        head = ValueHead(hidden_size=32)
        x = torch.randn(1, 5, 32)
        out = head(x)
        assert out.shape == (1, 5)

    def test_large_batch(self) -> None:
        head = ValueHead(hidden_size=32)
        x = torch.randn(128, 20, 32)
        out = head(x)
        assert out.shape == (128, 20)


# -----------------------------------------------------------------------
# Forward pass — value properties
# -----------------------------------------------------------------------


class TestForwardValues:
    """Tests for numerical properties of forward pass."""

    def test_output_is_finite(self) -> None:
        head = ValueHead(hidden_size=64)
        x = torch.randn(4, 10, 64)
        out = head(x)
        assert torch.all(torch.isfinite(out))

    def test_different_inputs_different_outputs(self) -> None:
        """Non-degenerate: different inputs should produce different values."""
        head = ValueHead(hidden_size=32)
        x1 = torch.randn(1, 5, 32)
        x2 = torch.randn(1, 5, 32)
        v1 = head(x1)
        v2 = head(x2)
        # Extremely unlikely to be identical for random inputs
        assert not torch.allclose(v1, v2)

    def test_deterministic_eval_mode(self) -> None:
        """Same input in eval mode should give same output."""
        head = ValueHead(hidden_size=32, dropout=0.1)
        head.eval()
        x = torch.randn(2, 5, 32)
        v1 = head(x)
        v2 = head(x)
        assert torch.allclose(v1, v2)

    def test_gradients_flow(self) -> None:
        """Gradients should flow back through the value head."""
        head = ValueHead(hidden_size=32)
        x = torch.randn(2, 5, 32, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# -----------------------------------------------------------------------
# Device handling
# -----------------------------------------------------------------------


class TestDevice:
    """Tests for device placement."""

    def test_cpu_by_default(self) -> None:
        head = ValueHead(hidden_size=32)
        x = torch.randn(1, 5, 32)
        out = head(x)
        assert out.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available",
    )
    def test_cuda_forward(self) -> None:
        head = ValueHead(hidden_size=32).cuda()
        x = torch.randn(1, 5, 32).cuda()
        out = head(x)
        assert out.device.type == "cuda"
