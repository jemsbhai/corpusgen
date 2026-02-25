"""Tests for GuidanceStrategy ABC — inference-time guidance interface."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from corpusgen.generate.guidance import GuidanceStrategy


# ---------------------------------------------------------------------------
# Concrete stub for testing the ABC
# ---------------------------------------------------------------------------


class _StubStrategy(GuidanceStrategy):
    """Minimal concrete implementation for testing the ABC contract."""

    def __init__(self, strategy_name: str = "stub") -> None:
        self._name = strategy_name
        self.prepare_called = False
        self.prepare_args: tuple = ()
        self.modify_called = False

    @property
    def name(self) -> str:
        return self._name

    def prepare(
        self,
        target_units: list[str],
        model: Any,
        tokenizer: Any,
    ) -> None:
        self.prepare_called = True
        self.prepare_args = (target_units, model, tokenizer)

    def modify_logits(self, input_ids: Any, logits: Any) -> Any:
        self.modify_called = True
        return logits


# ---------------------------------------------------------------------------
# ABC enforcement
# ---------------------------------------------------------------------------


class TestABCEnforcement:
    """GuidanceStrategy cannot be instantiated without implementing all methods."""

    def test_cannot_instantiate_abc_directly(self):
        with pytest.raises(TypeError):
            GuidanceStrategy()  # type: ignore[abstract]

    def test_missing_name_raises(self):
        class _NoName(GuidanceStrategy):
            def prepare(self, target_units, model, tokenizer):
                pass

            def modify_logits(self, input_ids, logits):
                return logits

        with pytest.raises(TypeError):
            _NoName()  # type: ignore[abstract]

    def test_missing_prepare_raises(self):
        class _NoPrepare(GuidanceStrategy):
            @property
            def name(self):
                return "x"

            def modify_logits(self, input_ids, logits):
                return logits

        with pytest.raises(TypeError):
            _NoPrepare()  # type: ignore[abstract]

    def test_missing_modify_logits_raises(self):
        class _NoModify(GuidanceStrategy):
            @property
            def name(self):
                return "x"

            def prepare(self, target_units, model, tokenizer):
                pass

        with pytest.raises(TypeError):
            _NoModify()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Concrete implementation works
# ---------------------------------------------------------------------------


class TestConcreteImplementation:
    """A fully-implemented subclass satisfies the ABC contract."""

    def test_instantiation(self):
        strategy = _StubStrategy()
        assert isinstance(strategy, GuidanceStrategy)

    def test_name_property(self):
        strategy = _StubStrategy(strategy_name="test_datg")
        assert strategy.name == "test_datg"

    def test_prepare_receives_args(self):
        strategy = _StubStrategy()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        targets = ["ʃ", "θ"]

        strategy.prepare(targets, mock_model, mock_tokenizer)

        assert strategy.prepare_called
        assert strategy.prepare_args == (targets, mock_model, mock_tokenizer)

    def test_modify_logits_returns_logits(self):
        strategy = _StubStrategy()
        mock_ids = MagicMock()
        mock_logits = MagicMock()

        result = strategy.modify_logits(mock_ids, mock_logits)

        assert strategy.modify_called
        assert result is mock_logits

    def test_prepare_with_empty_targets(self):
        strategy = _StubStrategy()
        strategy.prepare([], MagicMock(), MagicMock())
        assert strategy.prepare_called


# ---------------------------------------------------------------------------
# isinstance checks
# ---------------------------------------------------------------------------


class TestTypeChecks:
    """GuidanceStrategy works with isinstance for runtime dispatch."""

    def test_isinstance_check(self):
        strategy = _StubStrategy()
        assert isinstance(strategy, GuidanceStrategy)

    def test_non_strategy_fails_isinstance(self):
        assert not isinstance("not a strategy", GuidanceStrategy)
