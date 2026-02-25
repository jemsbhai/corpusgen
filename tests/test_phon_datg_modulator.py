"""Tests for LogitModulator — additive logit adjustments for Phon-DATG."""

import pytest

from corpusgen.generate.phon_datg.modulator import LogitModulator


# ---------------------------------------------------------------------------
# Helpers — lightweight tensor stand-in (no torch dependency in tests)
# ---------------------------------------------------------------------------


class _FakeTensor:
    """Minimal tensor-like object for testing without torch."""

    def __init__(self, data: list[list[float]]) -> None:
        self.data = [row[:] for row in data]
        self.shape = (len(data), len(data[0]) if data else 0)

    def clone(self) -> "_FakeTensor":
        return _FakeTensor([row[:] for row in self.data])

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            row, col = key
            if isinstance(col, list):
                return [self.data[row][c] for c in col]
            return self.data[row][col]
        return self.data[key]

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key) == 2:
            row, col = key
            if isinstance(col, list):
                for c, v in zip(col, value if hasattr(value, "__iter__") else [value] * len(col)):
                    self.data[row][c] = v
            else:
                self.data[row][col] = value
        else:
            self.data[key] = value


def _make_logits(vocab_size: int = 10, batch_size: int = 1, fill: float = 0.0):
    """Create a fake logits tensor filled with a constant."""
    return _FakeTensor([[fill] * vocab_size for _ in range(batch_size)])


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """LogitModulator creation and configuration."""

    def test_basic_creation(self):
        mod = LogitModulator()
        assert isinstance(mod, LogitModulator)

    def test_default_boost_strength(self):
        mod = LogitModulator()
        assert mod.boost_strength == 5.0

    def test_default_penalty_strength(self):
        mod = LogitModulator()
        assert mod.penalty_strength == -5.0

    def test_custom_boost_strength(self):
        mod = LogitModulator(boost_strength=10.0)
        assert mod.boost_strength == 10.0

    def test_custom_penalty_strength(self):
        mod = LogitModulator(penalty_strength=-3.0)
        assert mod.penalty_strength == -3.0

    def test_positive_penalty_raises(self):
        with pytest.raises(ValueError, match="penalty_strength"):
            LogitModulator(penalty_strength=2.0)

    def test_negative_boost_raises(self):
        with pytest.raises(ValueError, match="boost_strength"):
            LogitModulator(boost_strength=-1.0)

    def test_zero_strengths_allowed(self):
        mod = LogitModulator(boost_strength=0.0, penalty_strength=0.0)
        assert mod.boost_strength == 0.0
        assert mod.penalty_strength == 0.0


# ---------------------------------------------------------------------------
# Modulate: boost
# ---------------------------------------------------------------------------


class TestModulateBoost:
    """Attribute token logits are boosted."""

    def test_boosts_attribute_tokens(self):
        mod = LogitModulator(boost_strength=5.0, penalty_strength=0.0)
        logits = _make_logits(vocab_size=5, fill=0.0)

        result = mod.modulate(
            logits=logits,
            attribute_token_ids={1, 3},
            anti_attribute_token_ids=set(),
        )

        assert result[0, 1] == pytest.approx(5.0)
        assert result[0, 3] == pytest.approx(5.0)

    def test_non_attribute_tokens_unchanged(self):
        mod = LogitModulator(boost_strength=5.0, penalty_strength=0.0)
        logits = _make_logits(vocab_size=5, fill=1.0)

        result = mod.modulate(
            logits=logits,
            attribute_token_ids={2},
            anti_attribute_token_ids=set(),
        )

        assert result[0, 0] == pytest.approx(1.0)
        assert result[0, 1] == pytest.approx(1.0)
        assert result[0, 3] == pytest.approx(1.0)
        assert result[0, 4] == pytest.approx(1.0)

    def test_boost_is_additive(self):
        mod = LogitModulator(boost_strength=3.0, penalty_strength=0.0)
        logits = _make_logits(vocab_size=5, fill=2.0)

        result = mod.modulate(
            logits=logits,
            attribute_token_ids={0},
            anti_attribute_token_ids=set(),
        )

        assert result[0, 0] == pytest.approx(5.0)  # 2.0 + 3.0


# ---------------------------------------------------------------------------
# Modulate: penalty
# ---------------------------------------------------------------------------


class TestModulatePenalty:
    """Anti-attribute token logits are penalized."""

    def test_penalizes_anti_attribute_tokens(self):
        mod = LogitModulator(boost_strength=0.0, penalty_strength=-5.0)
        logits = _make_logits(vocab_size=5, fill=0.0)

        result = mod.modulate(
            logits=logits,
            attribute_token_ids=set(),
            anti_attribute_token_ids={0, 4},
        )

        assert result[0, 0] == pytest.approx(-5.0)
        assert result[0, 4] == pytest.approx(-5.0)

    def test_non_anti_attribute_unchanged(self):
        mod = LogitModulator(boost_strength=0.0, penalty_strength=-5.0)
        logits = _make_logits(vocab_size=5, fill=1.0)

        result = mod.modulate(
            logits=logits,
            attribute_token_ids=set(),
            anti_attribute_token_ids={2},
        )

        assert result[0, 0] == pytest.approx(1.0)
        assert result[0, 1] == pytest.approx(1.0)
        assert result[0, 3] == pytest.approx(1.0)

    def test_penalty_is_additive(self):
        mod = LogitModulator(boost_strength=0.0, penalty_strength=-4.0)
        logits = _make_logits(vocab_size=5, fill=3.0)

        result = mod.modulate(
            logits=logits,
            attribute_token_ids=set(),
            anti_attribute_token_ids={1},
        )

        assert result[0, 1] == pytest.approx(-1.0)  # 3.0 + (-4.0)


# ---------------------------------------------------------------------------
# Modulate: combined boost + penalty
# ---------------------------------------------------------------------------


class TestModulateCombined:
    """Boost and penalty applied simultaneously."""

    def test_boost_and_penalty_together(self):
        mod = LogitModulator(boost_strength=5.0, penalty_strength=-5.0)
        logits = _make_logits(vocab_size=5, fill=0.0)

        result = mod.modulate(
            logits=logits,
            attribute_token_ids={1},
            anti_attribute_token_ids={3},
        )

        assert result[0, 1] == pytest.approx(5.0)   # boosted
        assert result[0, 3] == pytest.approx(-5.0)   # penalized
        assert result[0, 0] == pytest.approx(0.0)    # neutral
        assert result[0, 2] == pytest.approx(0.0)    # neutral

    def test_overlap_gets_both(self):
        """A token in both sets gets boost + penalty."""
        mod = LogitModulator(boost_strength=5.0, penalty_strength=-3.0)
        logits = _make_logits(vocab_size=5, fill=0.0)

        result = mod.modulate(
            logits=logits,
            attribute_token_ids={2},
            anti_attribute_token_ids={2},
        )

        # Both adjustments applied: 0 + 5 + (-3) = 2.0
        assert result[0, 2] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Modulate: batch dimension
# ---------------------------------------------------------------------------


class TestModulateBatch:
    """Modulation applies across all batch entries."""

    def test_batch_size_two(self):
        mod = LogitModulator(boost_strength=3.0, penalty_strength=-2.0)
        logits = _make_logits(vocab_size=4, batch_size=2, fill=1.0)

        result = mod.modulate(
            logits=logits,
            attribute_token_ids={0},
            anti_attribute_token_ids={3},
        )

        for b in range(2):
            assert result[b, 0] == pytest.approx(4.0)   # 1 + 3
            assert result[b, 3] == pytest.approx(-1.0)   # 1 + (-2)
            assert result[b, 1] == pytest.approx(1.0)    # unchanged


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary conditions."""

    def test_empty_sets_no_change(self):
        mod = LogitModulator(boost_strength=5.0, penalty_strength=-5.0)
        logits = _make_logits(vocab_size=5, fill=1.0)

        result = mod.modulate(
            logits=logits,
            attribute_token_ids=set(),
            anti_attribute_token_ids=set(),
        )

        for i in range(5):
            assert result[0, i] == pytest.approx(1.0)

    def test_zero_strengths_no_change(self):
        mod = LogitModulator(boost_strength=0.0, penalty_strength=0.0)
        logits = _make_logits(vocab_size=5, fill=2.0)

        result = mod.modulate(
            logits=logits,
            attribute_token_ids={0, 1, 2},
            anti_attribute_token_ids={3, 4},
        )

        for i in range(5):
            assert result[0, i] == pytest.approx(2.0)

    def test_out_of_range_token_ids_ignored(self):
        """Token IDs beyond vocab_size should not cause errors."""
        mod = LogitModulator(boost_strength=5.0, penalty_strength=-5.0)
        logits = _make_logits(vocab_size=5, fill=0.0)

        result = mod.modulate(
            logits=logits,
            attribute_token_ids={1, 999},   # 999 out of range
            anti_attribute_token_ids={3, 888},
        )

        assert result[0, 1] == pytest.approx(5.0)
        assert result[0, 3] == pytest.approx(-5.0)

    def test_does_not_mutate_input(self):
        """Modulate should return new logits, not mutate the input."""
        mod = LogitModulator(boost_strength=5.0, penalty_strength=-5.0)
        logits = _make_logits(vocab_size=5, fill=0.0)
        original_val = logits[0, 1]

        mod.modulate(
            logits=logits,
            attribute_token_ids={1},
            anti_attribute_token_ids=set(),
        )

        assert logits[0, 1] == pytest.approx(original_val)
